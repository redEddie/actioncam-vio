#!/usr/bin/env python3
"""Bounded real STEP 4 shadow validation through q_cmd/gripper preview.

The process uses the canonical camera, remote policy, scheduler, IK, gravity,
controller, mapping, and safety modules.  The motor boundary is connected with
``read_only=True`` and every write-attempt counter must remain exactly zero.
No preview value is passed to ``MotorIO.write_goal_ticks``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
DEPLOY_DIR = HERE.parents[2]
PROJECT_ROOT = HERE.parents[3]
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ARM_JOINTS, MotorIO
from control.safety import (
    SafetyError,
    finite_vector,
    require_ik_success,
    require_joint_limits,
    require_raw_goal_ticks,
)
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.camera import CameraError, LatestFrameCamera
from inference.observation import ActualStateSnapshot, ObservationSnapshot, build_observation, state_from_motor_sample
from inference.remote_smolvla import RemoteRequest, RemoteResponse, RemoteSmolVLAClient, SubprocessSSHTransport
from trajectory.chunk_scheduler import ChunkScheduler


EXPECTED_PYTHON = Path("/home/kimminje/miniconda3/envs/lerobot_env2/bin/python")
DEFAULT_CONFIG = DEPLOY_DIR / "config" / "deployment.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _stats_ms(values_s: Iterable[float]) -> dict[str, float] | None:
    values = np.asarray(list(values_s), dtype=np.float64)
    if values.size == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError("timing samples must be finite")
    values *= 1000.0
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _zero_write_audit(motor: MotorIO) -> dict[str, int]:
    audit = asdict(motor.write_audit)
    if any(audit.values()):
        raise RuntimeError(f"STEP 4 zero-write invariant violated: {audit}")
    return audit


def _read_actual(
    motor: MotorIO,
    mapping: JointMapping,
    ik: DLSInverseKinematicsV7,
) -> tuple[ActualStateSnapshot, float, float]:
    started = time.perf_counter()
    degree_state = motor.read_present_positions(normalized=True)
    tick_state = motor.read_present_positions(normalized=False)
    elapsed = time.perf_counter() - started
    positions = {name: float(degree_state.positions[name]) for name in ARM_JOINTS}
    positions["gripper"] = float(tick_state.positions["gripper"])
    snapshot = state_from_motor_sample(
        positions,
        max(degree_state.timestamp, tick_state.timestamp),
        mapping,
        ik,
    )
    return snapshot, elapsed, positions["gripper"]


def _limit_status(actual: ActualStateSnapshot, ik: DLSInverseKinematicsV7) -> dict[str, Any]:
    q = np.deg2rad(actual.q_urdf_degrees)
    physical = {
        name: bool(lo <= value <= hi)
        for name, value in zip(ik.joint_names, q, strict=True)
        for lo, hi in [ik.PHYSICAL_JOINT_LIMITS_RAD[name]]
    }
    safe = {
        name: bool(lo <= value <= hi)
        for name, value in zip(ik.joint_names, q, strict=True)
        for lo, hi in [ik.safe_limits[name]]
    }
    return {
        "physical": physical,
        "safe": safe,
        "physical_pass": all(physical.values()),
        "safe_pass": all(safe.values()),
    }


def _wait_camera(camera: LatestFrameCamera, timeout_s: float = 8.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return camera.latest()
        except CameraError as exc:
            if "has not produced" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


class ShadowValidation:
    def __init__(self, config_path: Path, output_dir: Path, warmup: int, measured: int):
        self.config = load_runtime_config(config_path)
        self.output_dir = output_dir
        self.warmup_target = warmup
        self.measured_target = measured
        self.mapping = JointMapping(self.config.motor_mapping_path, self.config.motor_calibration)
        self.ik = DLSInverseKinematicsV7(
            str(DEPLOY_DIR / "ik" / "urdf" / "so_arm_with_gopro_final.urdf")
        )
        self.gravity = GravityCompensator.from_runtime_config(self.config)
        self.controller = BPDeltaController(
            self.config.k_ext, self.config.q_corr_clamp_deg, self.gravity
        )
        scheduler_config = self.config.deployment["scheduler"]
        self.scheduler = ChunkScheduler(
            scheduler_config["request_stride_s"],
            scheduler_config["transition_overlap_s"],
            self.config.action_period_s,
        )
        motor_config = self.config.deployment["motor"]
        self.motor = MotorIO(
            motor_config["port"],
            self.config.motor_calibration,
            motor_config["pid"],
            disable_torque_on_disconnect=False,
        )
        camera_config = self.config.deployment["camera"]
        self.camera = LatestFrameCamera(
            camera_config["device"],
            camera_config["width"],
            camera_config["height"],
            camera_config["stale_timeout_s"],
        )
        remote_config = self.config.deployment["remote"]
        self.remote = RemoteSmolVLAClient(
            SubprocessSSHTransport(remote_config),
            remote_config["request_timeout_s"],
            remote_config["request_queue_size"],
            remote_config["response_queue_size"],
        )
        self.remote_config = remote_config
        self.pending: dict[int, tuple[str, ObservationSnapshot, float]] = {}
        self.request_records: list[dict[str, Any]] = []
        self.tick_records: list[dict[str, Any]] = []
        self.present_read_s: list[float] = []
        self.request_id = 0
        self.q_seed: np.ndarray | None = None
        self.q_cmd_hold: np.ndarray | None = None
        self.origin_identity: int | None = None
        self.hold_capture_count = 0
        self.tick0_errors: list[float] = []
        self.missed_tick_replay = 0
        self.errors = {
            "camera": 0,
            "robot_read": 0,
            "ssh": 0,
            "remote": 0,
            "shape": 0,
            "nonfinite": 0,
            "request_id": 0,
            "reanchor": 0,
            "ik": 0,
            "safety_preview": 0,
        }

    def read_actual(self) -> tuple[ActualStateSnapshot, float]:
        try:
            actual, elapsed, gripper_raw = _read_actual(self.motor, self.mapping, self.ik)
        except Exception:
            self.errors["robot_read"] += 1
            raise
        self.present_read_s.append(elapsed)
        return actual, gripper_raw

    def submit(self, phase: str) -> dict[str, Any]:
        assembly_started = time.perf_counter()
        actual, gripper_raw = self.read_actual()
        frame = self.camera.latest()
        snapshot = build_observation(self.request_id, frame, actual)
        request = RemoteRequest.from_observation(snapshot)
        immutable = all(
            not value.flags.writeable
            for value in (
                snapshot.image_rgb,
                snapshot.raw_degrees,
                snapshot.q_obs_degrees,
                snapshot.s_obs,
                request.image_rgb,
                request.state,
            )
        )
        if not immutable:
            raise RuntimeError("request context is mutable")
        self.scheduler.register_request(snapshot)
        submitted = time.monotonic()
        self.remote.submit(request)
        self.pending[self.request_id] = (phase, snapshot, submitted)
        record = {
            "phase": phase,
            "request_id": self.request_id,
            "submitted": submitted,
            "t_obs": snapshot.t_obs,
            "t_frame": snapshot.t_frame,
            "t_q_obs": snapshot.t_q_obs,
            "frame_sequence": snapshot.frame_sequence,
            "source_shape": snapshot.frame_metadata["source_shape"],
            "frame_age_s": submitted - snapshot.t_frame,
            "image_state_gap_s": abs(snapshot.t_frame - snapshot.t_q_obs),
            "observation_assembly_s": time.perf_counter() - assembly_started,
            "q_obs_raw": snapshot.raw_degrees,
            "q_obs_urdf": snapshot.q_obs_degrees,
            "s_obs": snapshot.s_obs,
            "gripper_raw": gripper_raw,
            "immutable": immutable,
        }
        self.request_records.append(record)
        self.request_id += 1
        _zero_write_audit(self.motor)
        return record

    def process_response(self, response: RemoteResponse) -> dict[str, Any]:
        if response.request_id not in self.pending:
            self.errors["request_id"] += 1
            raise RuntimeError(f"unknown or duplicate response ID {response.request_id}")
        phase, snapshot, submitted = self.pending.pop(response.request_id)
        record = next(item for item in self.request_records if item["request_id"] == response.request_id)
        if response.error is not None or response.action is None:
            self.errors["remote"] += 1
            raise RuntimeError(f"remote inference failed for {response.request_id}: {response.error}")
        raw = np.asarray(response.action)
        if raw.shape != (1, 15, 6):
            self.errors["shape"] += 1
            raise RuntimeError(f"raw model response must be [1,15,6], got {raw.shape}")
        try:
            action = validate_postprocessed_action_chunk(raw)
        except Exception:
            self.errors["shape"] += 1
            raise
        arrival_started = time.perf_counter()
        arrival, arrival_gripper_raw = self.read_actual()
        arrival_read_s = time.perf_counter() - arrival_started
        if arrival.timestamp < response.t_arr:
            raise RuntimeError("fresh arrival state timestamp precedes response arrival")
        scheduler_started = time.perf_counter()
        integrated = self.scheduler.handle_response(
            response.request_id,
            action,
            response.t_arr,
            arrival.state,
        )
        scheduler_elapsed = time.perf_counter() - scheduler_started
        if integrated["status"] not in {"INTEGRATED", "EXPIRED_RESPONSE"}:
            self.errors["reanchor"] += 1
            raise RuntimeError(f"response was not integrated: {integrated['status']}")
        timing = integrated.get("timing_s", {})
        response_expired = integrated["status"] == "EXPIRED_RESPONSE"
        if not response_expired:
            context = integrated["context"]
            if not (
                context.request_id == snapshot.request_id
                and context.t_obs == snapshot.t_obs
                and np.array_equal(context.s_obs, snapshot.s_obs)
                and np.array_equal(context.q_obs, snapshot.q_obs_degrees)
            ):
                raise RuntimeError("scheduler request context changed")
            if not np.array_equal(integrated["reanchor"]["actual_arrival_state"], arrival.state):
                raise RuntimeError("scheduler did not use the fresh arrival state")
            ensemble = integrated["ensemble"]
        else:
            ensemble = {
                "status": "NOT_APPLICABLE_EXPIRED",
                "usable_overlap_duration": 0.0,
                "source_mode": [],
            }
        record.update(
            {
                "t_arr": response.t_arr,
                "arrival_q_timestamp": arrival.timestamp,
                "remote_s": response.t_arr - submitted,
                "arrival_read_s": arrival_read_s,
                "raw_shape": list(raw.shape),
                "postprocessed_shape": list(action.shape),
                "finite": bool(np.isfinite(action).all()),
                "a0": action[0],
                "a14": action[14],
                "arrival_q_raw": arrival.raw_degrees,
                "arrival_q_urdf": arrival.q_urdf_degrees,
                "arrival_state": arrival.state,
                "arrival_gripper_raw": arrival_gripper_raw,
                "arrival_state_source": "FRESH_REAL_PRESENT_POSITION",
                "reanchor_status": integrated["reanchor"]["status"],
                "scheduler_status": integrated["status"],
                "reanchor_tau_s": integrated["reanchor"]["tau"],
                "discarded_past_actions": integrated["reanchor"].get("first_future_index"),
                "reanchor_s": float(timing.get("reanchor", scheduler_elapsed)),
                "overlap_s": float(timing.get("overlap", 0.0)),
                "scheduler_response_s": scheduler_elapsed,
                "ensemble_status": ensemble["status"],
                "usable_overlap_s": float(ensemble["usable_overlap_duration"]),
                "source_mode": list(ensemble["source_mode"]),
                "request_response_match": True,
                "s_obs_preserved": True,
            }
        )
        _zero_write_audit(self.motor)
        return record

    def wait_preflight_response(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = self.remote.poll()
            if response is not None:
                return self.process_response(response)
            _zero_write_audit(self.motor)
            time.sleep(0.001)
        raise TimeoutError("single-request preflight timed out")

    def control_tick(self, now: float, measured_started: bool) -> dict[str, Any]:
        tick_started = time.perf_counter()
        sample_started = time.perf_counter()
        sampled = self.scheduler.sample(now)
        scheduler_sample_s = time.perf_counter() - sample_started
        record: dict[str, Any] = {
            "timestamp": now,
            "measured": measured_started,
            "status": sampled["status"],
            "scheduler_interpolation_s": scheduler_sample_s,
            "write_status": "LOG_ONLY",
        }
        if sampled["status"] != "VALID":
            record["full_tick_s"] = time.perf_counter() - tick_started
            self.tick_records.append(record)
            _zero_write_audit(self.motor)
            return record

        target = finite_vector(sampled["action"], 6, "sampled Cartesian target")
        actual, gripper_raw = self.read_actual()
        rotation = Rotation.from_euler(
            "ZYX", [actual.yaw_radians, target[4], target[3]]
        ).as_matrix()
        ik_started = time.perf_counter()
        seed = np.deg2rad(actual.q_urdf_degrees) if self.q_seed is None else self.q_seed
        q_nom = self.ik.solve(target[:3], rotation, seed)
        ik_s = time.perf_counter() - ik_started
        try:
            require_ik_success(q_nom, self.ik.last_converged)
            require_joint_limits(q_nom, self.ik.joint_names, self.ik.PHYSICAL_JOINT_LIMITS_RAD)
            require_joint_limits(q_nom, self.ik.joint_names, self.ik.safe_limits)
        except Exception:
            self.errors["ik"] += 1
            raise
        q_nom_deg = np.rad2deg(q_nom)

        gravity_started = time.perf_counter()
        gravity_eval = self.gravity.evaluate(np.deg2rad(actual.q_urdf_degrees))
        gravity_s = time.perf_counter() - gravity_started
        controller_started = time.perf_counter()
        began = False
        if self.controller.origin is None:
            if self.q_cmd_hold is None:
                raise RuntimeError("HOLD command is missing at trajectory onset")
            self.controller.begin_trajectory(actual.q_urdf_degrees, q_nom_deg, self.q_cmd_hold)
            self.origin_identity = id(self.controller.origin)
            self.hold_capture_count += 1
            began = True
        elif id(self.controller.origin) != self.origin_identity:
            raise RuntimeError("controller origin changed without a HOLD transition")
        q_cmd = self.controller.compute_trajectory(q_nom_deg, actual.q_urdf_degrees)
        diagnostics = self.controller.diagnostics()
        if began:
            tick0_error = float(np.max(np.abs(q_cmd - self.q_cmd_hold)))
            self.tick0_errors.append(tick0_error)
            if tick0_error > np.finfo(np.float64).eps * 16:
                raise RuntimeError(f"BP+delta TICK0 discontinuity: {tick0_error}")
        try:
            require_joint_limits(np.deg2rad(q_cmd), self.ik.joint_names, self.ik.PHYSICAL_JOINT_LIMITS_RAD)
            require_joint_limits(np.deg2rad(q_cmd), self.ik.joint_names, self.ik.safe_limits)
            raw_deg = self.mapping.urdf_degrees_to_raw_degrees(q_cmd)
            arm_ticks = self.mapping.arm_degrees_to_raw_ticks(raw_deg)
            require_raw_goal_ticks(arm_ticks, ARM_JOINTS)
            gripper_width = float(np.clip(target[5], 0.0, 1.0))
            gripper_tick = self.mapping.gripper_raw_from_width(gripper_width)
            if not self.mapping.gripper_open_raw <= gripper_tick <= self.mapping.gripper_closed_raw:
                raise SafetyError("gripper preview is outside canonical endpoints")
        except Exception:
            self.errors["safety_preview"] += 1
            raise
        controller_s = time.perf_counter() - controller_started
        self.q_seed = q_nom.copy()
        record.update(
            {
                "target": target,
                "q_actual": actual.q_urdf_degrees,
                "actual_gripper_raw": gripper_raw,
                "q_nom": q_nom_deg,
                "q_cmd": q_cmd,
                "arm_raw_preview": arm_ticks,
                "gripper_width_preview": gripper_width,
                "gripper_raw_preview": gripper_tick,
                "ik_s": ik_s,
                "ik_position_error": self.ik.last_position_error,
                "ik_orientation_error": self.ik.last_orientation_error,
                "gravity_s": gravity_s,
                "controller_safety_s": controller_s,
                "tau_g_nm": gravity_eval.tau_g_nm,
                "gravity_ratio_unclipped": gravity_eval.ratio_unclipped,
                "gravity_ratio_clipped": gravity_eval.ratio_clipped,
                "gravity_bias_deg": gravity_eval.gravity_bias_deg,
                "gravity_clamp_active": gravity_eval.clamp_active,
                "q_corr_deg": diagnostics["q_corr_deg"],
                "delta_support_bias_deg": diagnostics["delta_support_bias_deg"],
                "preview_allowed": True,
                "origin_identity": self.origin_identity,
                "tick0": began,
            }
        )
        record["full_tick_s"] = time.perf_counter() - tick_started
        self.tick_records.append(record)
        _zero_write_audit(self.motor)
        return record

    def summary(self, initial: dict[str, Any]) -> dict[str, Any]:
        measured_requests = [r for r in self.request_records if r["phase"] == "measured" and "t_arr" in r]
        measured_ticks = [r for r in self.tick_records if r["measured"]]
        valid_ticks = [r for r in measured_ticks if r["status"] == "VALID"]
        all_valid = [r for r in self.tick_records if r["status"] == "VALID"]
        if len(measured_requests) != self.measured_target:
            raise RuntimeError("measured request count is incomplete")
        if not valid_ticks:
            raise RuntimeError("no measured valid preview ticks")

        def timing(field: str, rows: list[dict[str, Any]]) -> dict[str, float] | None:
            return _stats_ms(row[field] for row in rows if field in row)

        targets = np.asarray([row["target"] for row in all_valid])
        q_nom = np.asarray([row["q_nom"] for row in all_valid])
        q_cmd = np.asarray([row["q_cmd"] for row in all_valid])
        gripper = np.asarray([row["gripper_width_preview"] for row in all_valid])
        gravity_tau = np.asarray([row["tau_g_nm"] for row in valid_ticks])
        gravity_ratio = np.asarray([row["gravity_ratio_unclipped"] for row in valid_ticks])
        gravity_clipped = np.asarray([row["gravity_ratio_clipped"] for row in valid_ticks])
        gravity_bias = np.asarray([row["gravity_bias_deg"] for row in valid_ticks])
        gravity_clamps = np.asarray([row["gravity_clamp_active"] for row in valid_ticks], dtype=bool)
        corrections = np.asarray([row["q_corr_deg"] for row in valid_ticks])

        def max_jump(values: np.ndarray) -> Any:
            if len(values) < 2:
                return 0.0
            diff = np.diff(values, axis=0)
            if diff.ndim == 1:
                return float(np.max(np.abs(diff)))
            return np.max(np.abs(diff), axis=0)

        integrated_requests = [
            r for r in self.request_records if r.get("scheduler_status") == "INTEGRATED"
        ]
        ensembles = [r["ensemble_status"] for r in integrated_requests]
        overlap_records = [
            r for r in integrated_requests if r.get("usable_overlap_s", 0.0) > 1e-6
        ]
        phases = {phase: sum(r["phase"] == phase and "t_arr" in r for r in self.request_records) for phase in ("first", "warmup", "measured")}
        status_counts = {
            status: sum(row["status"] == status for row in self.tick_records)
            for status in ("BEFORE_START", "VALID", "EXPIRED", "EMPTY")
        }
        return {
            "initial": initial,
            "requests": {
                "sent": len(self.request_records),
                "received": sum("t_arr" in row for row in self.request_records),
                "phases": phases,
                "raw_shapes": sorted({tuple(r["raw_shape"]) for r in measured_requests}),
                "postprocessed_shapes": sorted({tuple(r["postprocessed_shape"]) for r in measured_requests}),
                "all_finite": all(r["finite"] for r in measured_requests),
                "all_immutable": all(r["immutable"] for r in measured_requests),
                "all_ids_match": all(r["request_response_match"] for r in measured_requests),
                "all_fresh_arrival": all(
                    r["arrival_state_source"] == "FRESH_REAL_PRESENT_POSITION"
                    and r["arrival_q_timestamp"] >= r["t_arr"]
                    for r in measured_requests
                ),
                "expired": sum(r.get("scheduler_status") == "EXPIRED_RESPONSE" for r in measured_requests),
            },
            "camera": {
                "device": self.camera.device,
                "source_shapes": sorted({tuple(r["source_shape"]) for r in self.request_records}),
                "runtime_shape": [256, 256, 3],
                "unique_frame_sequences": len({r["frame_sequence"] for r in self.request_records}),
                "frame_age_ms": timing("frame_age_s", measured_requests),
                "image_state_gap_ms": timing("image_state_gap_s", measured_requests),
            },
            "remote_latency_ms": timing("remote_s", measured_requests),
            "local_timing_ms": {
                "present_position_read": _stats_ms(self.present_read_s),
                "observation_assembly": timing("observation_assembly_s", measured_requests),
                "reanchor": timing("reanchor_s", measured_requests),
                "overlap": timing("overlap_s", measured_requests),
                "scheduler_response": timing("scheduler_response_s", measured_requests),
                "scheduler_interpolation": timing("scheduler_interpolation_s", valid_ticks),
                "ik": timing("ik_s", valid_ticks),
                "gravity": timing("gravity_s", valid_ticks),
                "bp_delta_safety": timing("controller_safety_s", valid_ticks),
                "full_30hz_tick": timing("full_tick_s", valid_ticks),
            },
            "scheduler": {
                "request_stride_s": self.scheduler.request_stride_s,
                "horizon_s": self.config.deployment["model"]["horizon_s"],
                "chunk_transitions": max(0, len(ensembles) - 1),
                "ensemble_status_counts": {value: ensembles.count(value) for value in sorted(set(ensembles))},
                "usable_overlap_count": len(overlap_records),
                "zero_overlap_count": len(ensembles) - 1 - len(overlap_records),
                "usable_overlap_s": [r["usable_overlap_s"] for r in overlap_records],
                "tick_status_counts": status_counts,
                "missed_tick_replay": self.missed_tick_replay,
            },
            "ik": {
                "attempted": len(all_valid),
                "success": len(all_valid),
                "fail": self.errors["ik"],
                "max_position_error": float(max(r["ik_position_error"] for r in all_valid)),
                "max_orientation_error": float(max(r["ik_orientation_error"] for r in all_valid)),
            },
            "gravity": {
                "tau_min": gravity_tau.min(axis=0),
                "tau_max": gravity_tau.max(axis=0),
                "ratio_unclipped_min": gravity_ratio.min(axis=0),
                "ratio_unclipped_max": gravity_ratio.max(axis=0),
                "ratio_clipped_min": gravity_clipped.min(axis=0),
                "ratio_clipped_max": gravity_clipped.max(axis=0),
                "bias_min": gravity_bias.min(axis=0),
                "bias_max": gravity_bias.max(axis=0),
                "clamp_hits": gravity_clamps.sum(axis=0),
                "clamp_percent": gravity_clamps.mean(axis=0) * 100.0,
            },
            "controller": {
                "hold_capture_count": self.hold_capture_count,
                "tick0_max_error": float(max(self.tick0_errors, default=float("nan"))),
                "q_corr_clamp_hits": int(np.sum(np.isclose(np.abs(corrections), self.config.q_corr_clamp_deg))),
                "origin_identity_count": len({r["origin_identity"] for r in all_valid}),
            },
            "preview": {
                "count": len(all_valid),
                "finite": all(
                    np.isfinite(r["q_cmd"]).all()
                    and np.isfinite(r["gripper_width_preview"])
                    for r in all_valid
                ),
                "safe": all(r["preview_allowed"] for r in all_valid),
                "unsafe_count": self.errors["safety_preview"],
                "gripper_min": float(np.min(gripper)),
                "gripper_max": float(np.max(gripper)),
                "gripper_raw_min": int(min(r["gripper_raw_preview"] for r in all_valid)),
                "gripper_raw_max": int(max(r["gripper_raw_preview"] for r in all_valid)),
                "gripper_clipping_count": int(np.sum((gripper <= 0.0) | (gripper >= 1.0))),
            },
            "continuity": {
                "max_cartesian_single_tick_jump": max_jump(targets),
                "max_q_nom_single_tick_jump_deg": max_jump(q_nom),
                "max_q_cmd_single_tick_jump_deg": max_jump(q_cmd),
                "max_gripper_single_tick_jump": max_jump(gripper),
            },
            "errors": self.errors.copy(),
            "write_audit": _zero_write_audit(self.motor),
            "present_position_reads": self.motor.present_position_reads,
            "present_position_read_failures": self.motor.present_position_read_failures,
        }

    def run(self) -> dict[str, Any]:
        motor_started = camera_started = remote_started = False
        report: dict[str, Any] = {
            "status": "RUNNING",
            "started_wall_time": datetime.now().astimezone().isoformat(),
            "python": sys.executable,
            "project_root": PROJECT_ROOT,
            "checkpoint": self.remote_config["checkpoint"],
            "remote_target": f"{self.remote_config['user']}@{self.remote_config['host']}:{self.remote_config['port']}",
            "output_dir": self.output_dir,
        }
        self.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            self.motor.connect(permit_motion=False, read_only=True)
            motor_started = True
            _zero_write_audit(self.motor)
            initial_actual, initial_gripper_raw = self.read_actual()
            initial_limits = _limit_status(initial_actual, self.ik)
            initial = {
                "timestamp": initial_actual.timestamp,
                "arm_raw": initial_actual.raw_degrees,
                "arm_urdf": initial_actual.q_urdf_degrees,
                "gripper_raw": initial_gripper_raw,
                "gripper_normalized": initial_actual.state[5],
                "state": initial_actual.state,
                "yaw": initial_actual.yaw_radians,
                "limits": initial_limits,
            }
            report["initial"] = initial
            if not initial_limits["physical_pass"] or not initial_limits["safe_pass"]:
                raise RuntimeError(f"initial actual pose is outside a physical/safe limit: {initial_limits}")
            self.q_seed = np.deg2rad(initial_actual.q_urdf_degrees)
            self.q_cmd_hold = self.controller.compute_hold(
                self.config.start_urdf_deg,
                initial_actual.q_urdf_degrees,
            )

            self.camera.start()
            camera_started = True
            _wait_camera(self.camera)
            self.remote.start()
            remote_started = True

            self.submit("first")
            preflight = self.wait_preflight_response(float(self.remote_config["request_timeout_s"]) + 3.0)
            if preflight["raw_shape"] != [1, 15, 6] or preflight["postprocessed_shape"] != [15, 6]:
                raise RuntimeError("single-request preflight model contract failed")

            total_requests = 1 + self.warmup_target + self.measured_target
            next_tick = time.monotonic()
            measured_started = False
            final_response_time: float | None = None
            deadline = time.monotonic() + max(90.0, total_requests * self.scheduler.request_stride_s + 45.0)
            while time.monotonic() < deadline:
                response = self.remote.poll()
                while response is not None:
                    integrated = self.process_response(response)
                    if integrated["phase"] == "measured":
                        measured_started = True
                    if sum(r["phase"] == "measured" and "t_arr" in r for r in self.request_records) == self.measured_target:
                        final_response_time = time.monotonic()
                    response = self.remote.poll()

                now = time.monotonic()
                if self.request_id < total_requests and self.scheduler.should_request(now):
                    if self.request_id <= self.warmup_target:
                        phase = "warmup"
                    else:
                        phase = "measured"
                    self.submit(phase)

                if now >= next_tick:
                    self.control_tick(now, measured_started)
                    next_tick += self.config.control_period_s
                    if next_tick <= now:
                        # Skip lateness; never replay missed 30 Hz ticks.
                        next_tick = now + self.config.control_period_s

                if (
                    final_response_time is not None
                    and not self.pending
                    and time.monotonic() - final_response_time >= 0.35
                ):
                    break
                time.sleep(min(0.001, max(0.0, next_tick - time.monotonic())))
            else:
                raise TimeoutError("bounded full-shadow run did not finish")

            report["summary"] = self.summary(initial)
            report["status"] = "PASS"
        except Exception as exc:
            report["status"] = "FAIL"
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"] = self.errors.copy()
            report["write_audit"] = asdict(self.motor.write_audit)
            raise
        finally:
            if remote_started:
                self.remote.close()
            if camera_started:
                self.camera.stop()
            if motor_started:
                self.motor.disconnect()
            report["finished_wall_time"] = datetime.now().astimezone().isoformat()
            report["write_audit_after_cleanup"] = asdict(self.motor.write_audit)
            report["present_position_reads"] = self.motor.present_position_reads
            report["present_position_read_failures"] = self.motor.present_position_read_failures
            (self.output_dir / "summary.json").write_text(
                json.dumps(report, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )
            with (self.output_dir / "request_telemetry.jsonl").open("x", encoding="utf-8") as stream:
                for row in self.request_records:
                    stream.write(json.dumps(row, default=_json_default, separators=(",", ":")) + "\n")
            with (self.output_dir / "preview_telemetry.jsonl").open("x", encoding="utf-8") as stream:
                for row in self.tick_records:
                    stream.write(json.dumps(row, default=_json_default, separators=(",", ":")) + "\n")
            print(json.dumps(report, indent=2, default=_json_default))
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-live-io", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise SystemExit(f"wrong Python interpreter: {sys.executable}")
    if not args.confirm_live_io:
        raise SystemExit("STEP 4 live I/O is blocked without --confirm-live-io")
    if args.warmup_requests < 0 or args.measured_requests <= 0:
        raise SystemExit("request counts are invalid")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "archive" / "generated_results" / "deployment" / f"step4_full_shadow_{timestamp}"
    )
    validation = ShadowValidation(
        args.config.resolve(), output_dir.resolve(), args.warmup_requests, args.measured_requests
    )
    validation.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
