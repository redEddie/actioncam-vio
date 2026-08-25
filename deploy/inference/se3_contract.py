"""SE(3) relative-waypoint action contract (`se3_relwp_v1`) helpers.

Model I/O (matches training variant V6, `~/work/smolvla/variants/`):
  state  (10,)   = [ inv(T_t) @ T_{t-dt} -> pos(3) + rot6d(first two rows, 6), gripper_{t-dt}(1) ]
  action (15,10) = [ inv(T_t) @ T_{t+k}  -> pos(3) + rot6d(6),                gripper_{t+k}(1) ]  (k=1..15)

Deployment adapter: convert the SE(3) chunk into the legacy yaw-free 6D incremental actions
(A6[k] = S6[k+1] - S6[k]) so the downstream scheduler / reanchor / interpolation / IK path
(`reconstruct_incremental_states`) reproduces the SE(3) waypoints exactly. Yaw from the model is
intentionally discarded at execution time (SO-101 is 5-DOF; IK keeps using current FK yaw).
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SE3_ACTION_SHAPE = (15, 10)
SE3_WIRE_BATCH_SHAPE = (1, 15, 10)
SE3_STATE_DIM = 10
STATE_HISTORY_LAG_S = 0.1


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a first-two-rows 6D rotation back to a proper rotation matrix."""
    a1, a2 = np.asarray(d6[:3], dtype=np.float64), np.asarray(d6[3:6], dtype=np.float64)
    n1 = np.linalg.norm(a1)
    if not np.isfinite(n1) or n1 < 1e-8:
        raise ValueError("rot6d first row is degenerate")
    b1 = a1 / n1
    a2p = a2 - (b1 @ a2) * b1
    n2 = np.linalg.norm(a2p)
    if not np.isfinite(n2) or n2 < 1e-8:
        raise ValueError("rot6d second row is degenerate")
    b2 = a2p / n2
    return np.stack([b1, b2, np.cross(b1, b2)])


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:2, :3].reshape(6)


def transform_to_10d(transform: np.ndarray, gripper: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    return np.concatenate([transform[:3, 3], matrix_to_rot6d(transform[:3, :3]), [float(gripper)]])


def relative_state10(t_current: np.ndarray, t_previous: np.ndarray, gripper_previous: float) -> np.ndarray:
    """state = inv(T_t) @ T_{t-dt} with the previous gripper width (identity at episode start)."""
    rel = np.linalg.solve(np.asarray(t_current, dtype=np.float64), np.asarray(t_previous, dtype=np.float64))
    return transform_to_10d(rel, gripper_previous)


def validate_se3_action_chunk(value: Any) -> np.ndarray:
    result = np.asarray(value)
    if result.shape == SE3_WIRE_BATCH_SHAPE:
        result = result[0]
    elif result.shape != SE3_ACTION_SHAPE:
        raise ValueError(
            f"se3 action must have shape {SE3_WIRE_BATCH_SHAPE} or {SE3_ACTION_SHAPE}, got {result.shape}"
        )
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("se3 action contains NaN or Inf")
    return result.copy()


def _wrap_to(reference: float, angle: float) -> float:
    return float(reference + np.angle(np.exp(1j * (angle - reference))))


def chunk10_to_incremental6(t_obs: np.ndarray, s_obs6: Sequence[float], chunk10: Any) -> np.ndarray:
    """Convert an SE(3) relative-waypoint chunk into legacy yaw-free incremental actions [15,6].

    Guarantee: `reconstruct_incremental_states(s_obs6, result)` returns exactly the yaw-free
    projection of the absolute SE(3) waypoints (gripper clipped to [0,1])."""
    chunk = validate_se3_action_chunk(chunk10)
    t_obs = np.asarray(t_obs, dtype=np.float64)
    anchor = np.asarray(s_obs6, dtype=np.float64)
    if t_obs.shape != (4, 4) or anchor.shape != (6,):
        raise ValueError("t_obs must be 4x4 and s_obs6 must have shape (6,)")
    if not np.isfinite(t_obs).all() or not np.isfinite(anchor).all():
        raise ValueError("t_obs and s_obs6 must be finite")

    states = np.empty((chunk.shape[0] + 1, 6), dtype=np.float64)
    states[0] = anchor
    states[0, 5] = np.clip(states[0, 5], 0.0, 1.0)
    ref_roll, ref_pitch = float(anchor[3]), float(anchor[4])
    for k, row in enumerate(chunk):
        rel = np.eye(4)
        rel[:3, :3] = rot6d_to_matrix(row[3:9])
        rel[:3, 3] = row[:3]
        absolute = t_obs @ rel
        yaw, pitch, roll = Rotation.from_matrix(absolute[:3, :3]).as_euler("ZYX")
        states[k + 1, :3] = absolute[:3, 3]
        states[k + 1, 3] = ref_roll = _wrap_to(ref_roll, roll)
        states[k + 1, 4] = ref_pitch = _wrap_to(ref_pitch, pitch)
        states[k + 1, 5] = np.clip(row[9], 0.0, 1.0)
    return np.diff(states, axis=0)


class StateHistory:
    """Ring buffer of (timestamp, transform, gripper, state6) with time interpolation.

    Supports (a) the lagged relative state for se3_relwp_v1 and (b) UMI-style observation
    alignment: interpolate the robot state at the (back-dated) image capture time so the
    observation tuple is temporally consistent, like the training dataset."""

    def __init__(self, horizon_s: float = 2.0):
        self.horizon_s = float(horizon_s)
        self._times: list[float] = []
        self._entries: list[tuple[np.ndarray, float, np.ndarray | None]] = []

    def push(self, timestamp: float, transform: np.ndarray, gripper: float, state6=None) -> None:
        transform = np.asarray(transform, dtype=np.float64).copy()
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("history transform must be a finite 4x4 matrix")
        timestamp = float(timestamp)
        if self._times and timestamp < self._times[-1]:
            return
        state6_arr = None
        if state6 is not None:
            state6_arr = np.asarray(state6, dtype=np.float64).copy()
            if state6_arr.shape != (6,) or not np.isfinite(state6_arr).all():
                raise ValueError("history state6 must be a finite 6-vector")
        self._times.append(timestamp)
        self._entries.append((transform, float(gripper), state6_arr))
        cutoff = timestamp - self.horizon_s
        while len(self._times) > 2 and self._times[0] < cutoff:
            self._times.pop(0)
            self._entries.pop(0)

    def interpolate(self, timestamp: float) -> tuple[np.ndarray, float, np.ndarray | None]:
        """(transform, gripper, state6) linearly interpolated at ``timestamp`` (rotation via slerp,
        roll/pitch wrap-aware; clamped to the buffer's ends)."""
        if not self._times:
            raise ValueError("state history is empty")
        target = float(timestamp)
        index = bisect_left(self._times, target)
        if index <= 0 or len(self._times) == 1:
            transform, gripper, state6 = self._entries[0 if index <= 0 else -1]
            return transform.copy(), gripper, None if state6 is None else state6.copy()
        if index >= len(self._times):
            transform, gripper, state6 = self._entries[-1]
            return transform.copy(), gripper, None if state6 is None else state6.copy()
        t0, t1 = self._times[index - 1], self._times[index]
        (T0, g0, s0), (T1, g1, s1) = self._entries[index - 1], self._entries[index]
        alpha = 0.0 if t1 <= t0 else float(np.clip((target - t0) / (t1 - t0), 0.0, 1.0))
        transform = np.eye(4)
        transform[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
        rotations = Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
        transform[:3, :3] = Slerp([0.0, 1.0], rotations)(alpha).as_matrix()
        gripper = (1.0 - alpha) * g0 + alpha * g1
        state6 = None
        if s0 is not None and s1 is not None:
            state6 = (1.0 - alpha) * s0 + alpha * s1
            for axis in (3, 4):  # roll/pitch wrap-aware
                delta = np.angle(np.exp(1j * (s1[axis] - s0[axis])))
                state6[axis] = s0[axis] + alpha * delta
            state6[5] = np.clip(gripper, 0.0, 1.0)
        return transform, float(gripper), state6

    def lagged(self, timestamp: float, lag_s: float = STATE_HISTORY_LAG_S) -> tuple[np.ndarray, float, np.ndarray | None]:
        """Entry closest to ``timestamp - lag_s`` (falls back to the oldest/newest available)."""
        if not self._times:
            raise ValueError("state history is empty")
        target = float(timestamp) - float(lag_s)
        index = bisect_left(self._times, target)
        if index <= 0:
            best = 0
        elif index >= len(self._times):
            best = len(self._times) - 1
        else:
            best = index if abs(self._times[index] - target) < abs(self._times[index - 1] - target) else index - 1
        transform, gripper, state6 = self._entries[best]
        return transform.copy(), gripper, None if state6 is None else state6.copy()


__all__ = [
    "SE3_ACTION_SHAPE",
    "SE3_WIRE_BATCH_SHAPE",
    "SE3_STATE_DIM",
    "STATE_HISTORY_LAG_S",
    "StateHistory",
    "chunk10_to_incremental6",
    "matrix_to_rot6d",
    "relative_state10",
    "rot6d_to_matrix",
    "transform_to_10d",
    "validate_se3_action_chunk",
]
