"""Request-aware chunk lifecycle, fresh-arrival reanchor, and overlap orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence
import time

import numpy as np

from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.observation import ObservationSnapshot
from .interpolate import sample_trajectory_at_time
from .overlap import overlap_ensemble
from .reanchor import reanchor_trajectory


class SchedulerState(str, Enum):
    BEFORE_START = "BEFORE_START"
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    FAULT = "FAULT"


@dataclass(frozen=True)
class RequestContext:
    request_id: int
    t_obs: float
    s_obs: np.ndarray
    q_obs: np.ndarray
    t_frame: float
    t_q_obs: float
    frame_sequence: int
    frame_metadata: Mapping[str, object]

    @classmethod
    def from_observation(cls, snapshot: ObservationSnapshot) -> "RequestContext":
        state = snapshot.s_obs.copy()
        joints = snapshot.q_obs_degrees.copy()
        state.setflags(write=False)
        joints.setflags(write=False)
        return cls(
            snapshot.request_id,
            snapshot.t_obs,
            state,
            joints,
            snapshot.t_frame,
            snapshot.t_q_obs,
            snapshot.frame_sequence,
            MappingProxyType(dict(snapshot.frame_metadata)),
        )


class ChunkScheduler:
    """Own trajectory/request state without queues or implicit action popping."""

    def __init__(self, request_stride_s: float = 0.9, transition_overlap_s: float = 0.2, action_dt_s: float = 0.1):
        self.request_stride_s = float(request_stride_s)
        self.transition_overlap_s = float(transition_overlap_s)
        self.action_dt_s = float(action_dt_s)
        if min(self.request_stride_s, self.transition_overlap_s, self.action_dt_s) <= 0.0:
            raise ValueError("scheduler periods must be positive")
        self.pending: dict[int, RequestContext] = {}
        self.completed: set[int] = set()
        self.active_times = np.empty((0,), dtype=np.float64)
        self.active_states = np.empty((0, 6), dtype=np.float64)
        self.last_request_time: float | None = None
        self.last_integrated_request_id: int | None = None
        self.state = SchedulerState.EXPIRED

    @property
    def in_flight(self) -> bool:
        return bool(self.pending)

    def should_request(self, now: float) -> bool:
        if not np.isfinite(now):
            return False
        if self.in_flight:
            return False
        return self.last_request_time is None or now - self.last_request_time >= self.request_stride_s - 1e-9

    def register_request(self, snapshot: ObservationSnapshot) -> RequestContext:
        request_id = snapshot.request_id
        if request_id in self.pending or request_id in self.completed:
            raise ValueError(f"duplicate request ID {request_id}")
        context = RequestContext.from_observation(snapshot)
        self.pending[request_id] = context
        self.last_request_time = context.t_obs
        return context

    def handle_response(
        self,
        request_id: int,
        action: Sequence[Sequence[float]],
        t_arr: float,
        s_actual_arrival: Sequence[float],
    ) -> dict:
        if request_id in self.completed:
            return {"status": "DUPLICATE_RESPONSE", "request_id": request_id}
        context = self.pending.get(request_id)
        if context is None:
            return {"status": "UNKNOWN_RESPONSE", "request_id": request_id}
        if self.last_integrated_request_id is not None and request_id < self.last_integrated_request_id:
            del self.pending[request_id]
            self.completed.add(request_id)
            return {"status": "OUT_OF_ORDER_RESPONSE", "request_id": request_id}

        started = time.perf_counter()
        actions = validate_postprocessed_action_chunk(action)
        reanchor_started = time.perf_counter()
        result = reanchor_trajectory(
            actions,
            context.s_obs,
            s_actual_arrival,
            context.t_obs,
            float(t_arr),
            self.action_dt_s,
        )
        reanchor_elapsed = time.perf_counter() - reanchor_started
        del self.pending[request_id]
        self.completed.add(request_id)
        if result["status"] == "EXPIRED":
            self.state = SchedulerState.EXPIRED if not self._active_valid(t_arr) else SchedulerState.VALID
            return {
                "status": "EXPIRED_RESPONSE",
                "request_id": request_id,
                "reanchor": result,
                "timing_s": {
                    "reanchor": reanchor_elapsed,
                    "overlap": 0.0,
                    "total": time.perf_counter() - started,
                },
            }

        new_times = result["absolute_times"]
        new_states = result["states"]
        overlap_started = time.perf_counter()
        if self.active_times.size:
            ensemble = overlap_ensemble(
                self.active_times,
                self.active_states,
                new_times,
                new_states,
                float(t_arr),
                dt=self.action_dt_s,
                ensemble_req_dur=self.transition_overlap_s,
            )
            self.active_times = np.asarray(ensemble["merged_times"], dtype=np.float64)
            self.active_states = np.asarray(ensemble["merged_states"], dtype=np.float64)
        else:
            ensemble = {"status": "FIRST_CHUNK", "usable_overlap_duration": 0.0, "source_mode": ["NEW"] * len(new_times)}
            self.active_times = np.asarray(new_times, dtype=np.float64).copy()
            self.active_states = np.asarray(new_states, dtype=np.float64).copy()
        overlap_elapsed = time.perf_counter() - overlap_started
        self.last_integrated_request_id = request_id
        self.state = SchedulerState.VALID
        return {
            "status": "INTEGRATED",
            "request_id": request_id,
            "context": context,
            "reanchor": result,
            "ensemble": ensemble,
            "timing_s": {
                "reanchor": reanchor_elapsed,
                "overlap": overlap_elapsed,
                "total": time.perf_counter() - started,
            },
        }

    def _active_valid(self, now: float) -> bool:
        return bool(self.active_times.size and now <= self.active_times[-1] + 1e-9)

    def sample(self, now: float) -> dict:
        sample = sample_trajectory_at_time(self.active_times, self.active_states, float(now))
        status = sample["status"]
        if status == "VALID":
            self.state = SchedulerState.VALID
        elif status == "BEFORE_START":
            self.state = SchedulerState.BEFORE_START
        elif status in {"EXPIRED", "EMPTY"}:
            self.state = SchedulerState.EXPIRED
        else:
            self.state = SchedulerState.FAULT
        return sample


__all__ = ["ChunkScheduler", "RequestContext", "SchedulerState"]
