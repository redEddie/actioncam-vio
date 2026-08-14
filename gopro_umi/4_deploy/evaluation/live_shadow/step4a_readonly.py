#!/usr/bin/env python3
"""Bounded STEP 4A read/observe/infer/read/reanchor validation.

This evaluation intentionally does not instantiate the chunk scheduler,
BP+delta controller, or any motor-command construction path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve()
DEPLOY_DIR = HERE.parents[2]
PROJECT_ROOT = HERE.parents[3]
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ARM_JOINTS, MotorIO
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.camera import CameraError, LatestFrameCamera
from inference.observation import ActualStateSnapshot, build_observation, state_from_motor_sample
from inference.remote_smolvla import RemoteRequest, RemoteSmolVLAClient, SubprocessSSHTransport
from trajectory.reanchor import reanchor_trajectory


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


def _stats_ms(values_s: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values_s), dtype=np.float64) * 1000.0
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("timing statistics require finite samples")
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
        raise RuntimeError(f"STEP 4A write invariant violated: {audit}")
    return audit


def _read_actual(
    motor: MotorIO,
    mapping: JointMapping,
    ik: DLSInverseKinematicsV7,
) -> tuple[ActualStateSnapshot, float, dict[str, float]]:
    started = time.monotonic()
    degree_state = motor.read_present_positions(normalized=True)
    tick_state = motor.read_present_positions(normalized=False)
    finished = time.monotonic()
    positions = {name: float(degree_state.positions[name]) for name in ARM_JOINTS}
    positions["gripper"] = float(tick_state.positions["gripper"])
    actual = state_from_motor_sample(
        positions,
        max(degree_state.timestamp, tick_state.timestamp),
        mapping,
        ik,
    )
    return actual, finished - started, {
        "degree_read_timestamp": degree_state.timestamp,
        "tick_read_timestamp": tick_state.timestamp,
        "gripper_raw": positions["gripper"],
    }


def _hard_limit_status(actual: ActualStateSnapshot, ik: DLSInverseKinematicsV7) -> dict[str, Any]:
    q_rad = np.deg2rad(actual.q_urdf_degrees)
    hard = {
        name: bool(lo <= value <= hi)
        for name, value in zip(ik.joint_names, q_rad, strict=True)
        for lo, hi in [ik.PHYSICAL_JOINT_LIMITS_RAD[name]]
    }
    safe = {
        name: bool(lo <= value <= hi)
        for name, value in zip(ik.joint_names, q_rad, strict=True)
        for lo, hi in [ik.safe_limits[name]]
    }
    return {"hard": hard, "safe": safe, "hard_pass": all(hard.values()), "safe_pass": all(safe.values())}


def _wait_for_camera(camera: LatestFrameCamera, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return camera.latest()
        except CameraError as exc:
            if "has not produced" not in str(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _camera_rate_sample(camera: LatestFrameCamera, unique_frames: int = 30) -> dict[str, Any]:
    samples = []
    deadline = time.monotonic() + 8.0
    last_sequence = None
    while len(samples) < unique_frames and time.monotonic() < deadline:
        frame = camera.latest()
        if frame.sequence != last_sequence:
            samples.append(frame)
            last_sequence = frame.sequence
        time.sleep(0.001)
    if len(samples) < 2:
        raise CameraError("insufficient unique camera frames for rate validation")
    timestamps = np.asarray([item.timestamp for item in samples], dtype=np.float64)
    intervals = np.diff(timestamps)
    return {
        "frames": len(samples),
        "source_shape": samples[-1].source_shape,
        "runtime_shape": samples[-1].image_rgb.shape,
        "observed_fps": float(1.0 / np.mean(intervals)),
        "interval_stats_ms": _stats_ms(intervals),
        "capture_failures": 0,
    }


def _wait_response(client: RemoteSmolVLAClient, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.poll()
        if response is not None:
            return response
        time.sleep(0.001)
    raise TimeoutError("local response polling deadline expired")


def _gravity_record(gravity: GravityCompensator, actual: ActualStateSnapshot) -> dict[str, Any]:
    evaluated = gravity.evaluate(np.deg2rad(actual.q_urdf_degrees))
    return {
        "tau_g_nm": evaluated.tau_g_nm,
        "ratio_unclipped": evaluated.ratio_unclipped,
        "ratio_clipped": evaluated.ratio_clipped,
        "gravity_bias_deg": evaluated.gravity_bias_deg,
        "static_bias_deg": evaluated.static_bias_deg,
        "total_support_bias_deg": evaluated.total_support_bias_deg,
        "clamp_flags": evaluated.clamp_active,
    }


def _run_request(
    request_id: int,
    phase: str,
    motor: MotorIO,
    mapping: JointMapping,
    ik: DLSInverseKinematicsV7,
    gravity: GravityCompensator,
    camera: LatestFrameCamera,
    remote: RemoteSmolVLAClient,
    response_timeout_s: float,
) -> dict[str, Any]:
    actual_obs, obs_read_s, obs_read_meta = _read_actual(motor, mapping, ik)
    frame = camera.latest()
    assembly_started = time.monotonic()
    snapshot = build_observation(request_id, frame, actual_obs)
    assembly_finished = time.monotonic()
    request = RemoteRequest.from_observation(snapshot)
    immutable = all(
        not values.flags.writeable
        for values in (
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

    submitted_at = time.monotonic()
    remote.submit(request)
    response = _wait_response(remote, response_timeout_s + 2.0)
    if response.request_id != request_id:
        raise RuntimeError("response request ID mismatch")
    if response.error is not None or response.action is None:
        raise RuntimeError(f"remote inference failed: {response.error}")
    raw_action = np.asarray(response.action)
    action = validate_postprocessed_action_chunk(raw_action)

    arrival_actual, arrival_read_s, arrival_meta = _read_actual(motor, mapping, ik)
    if arrival_actual.timestamp < response.t_arr:
        raise RuntimeError("arrival state was not read after response arrival")
    reanchor_started = time.monotonic()
    reanchor = reanchor_trajectory(
        action,
        snapshot.s_obs,
        arrival_actual.state,
        snapshot.t_obs,
        response.t_arr,
    )
    reanchor_finished = time.monotonic()
    _zero_write_audit(motor)
    return {
        "phase": phase,
        "request_id": request_id,
        "t_frame": snapshot.t_frame,
        "t_q_obs": snapshot.t_q_obs,
        "t_obs": snapshot.t_obs,
        "submitted_at": submitted_at,
        "t_arr": response.t_arr,
        "arrival_q_timestamp": arrival_actual.timestamp,
        "frame_sequence": snapshot.frame_sequence,
        "frame_metadata": dict(snapshot.frame_metadata),
        "frame_age_s": assembly_finished - snapshot.t_frame,
        "image_state_abs_gap_s": abs(snapshot.t_frame - snapshot.t_q_obs),
        "observation_read_s": obs_read_s,
        "observation_assembly_s": assembly_finished - assembly_started,
        "remote_request_response_s": response.t_arr - submitted_at,
        "t_obs_to_t_arr_s": response.t_arr - snapshot.t_obs,
        "arrival_read_s": arrival_read_s,
        "reanchor_s": reanchor_finished - reanchor_started,
        "t_obs_to_reanchor_ready_s": reanchor_finished - snapshot.t_obs,
        "raw_shape": list(raw_action.shape),
        "postprocessed_shape": list(action.shape),
        "response_finite": bool(np.isfinite(action).all()),
        "a0": action[0],
        "a14": action[14],
        "q_obs_raw_deg": snapshot.raw_degrees,
        "q_obs_urdf_deg": snapshot.q_obs_degrees,
        "s_obs": snapshot.s_obs,
        "obs_read_metadata": obs_read_meta,
        "arrival_q_raw_deg": arrival_actual.raw_degrees,
        "arrival_q_urdf_deg": arrival_actual.q_urdf_degrees,
        "s_actual_arrival": arrival_actual.state,
        "arrival_read_metadata": arrival_meta,
        "arrival_state_source": "FRESH_REAL_PRESENT_POSITION",
        "request_context_immutable": immutable,
        "reanchor_status": reanchor["status"],
        "reanchor_tau_s": reanchor["tau"],
        "reanchor_interval": reanchor.get("current_interval_index"),
        "reanchor_alpha": reanchor.get("alpha"),
        "first_future_index": reanchor.get("first_future_index"),
        "discarded_past_actions": reanchor.get("first_future_index"),
        "gravity": _gravity_record(gravity, arrival_actual),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in records if item["phase"] == "measured"]
    if not measured:
        raise RuntimeError("no measured requests completed")
    timing_fields = (
        "frame_age_s",
        "observation_assembly_s",
        "remote_request_response_s",
        "arrival_read_s",
        "reanchor_s",
        "t_obs_to_reanchor_ready_s",
        "image_state_abs_gap_s",
    )
    gravity = np.asarray([item["gravity"]["tau_g_nm"] for item in measured])
    ratio = np.asarray([item["gravity"]["ratio_unclipped"] for item in measured])
    bias = np.asarray([item["gravity"]["gravity_bias_deg"] for item in measured])
    clamps = np.asarray([item["gravity"]["clamp_flags"] for item in measured], dtype=bool)
    return {
        "requests_measured": len(measured),
        "timing_ms": {field: _stats_ms(item[field] for item in measured) for field in timing_fields},
        "raw_shapes": sorted({tuple(item["raw_shape"]) for item in measured}),
        "postprocessed_shapes": sorted({tuple(item["postprocessed_shape"]) for item in measured}),
        "reanchor_status_counts": {
            status: sum(item["reanchor_status"] == status for item in measured)
            for status in sorted({item["reanchor_status"] for item in measured})
        },
        "gravity": {
            "tau_min": gravity.min(axis=0),
            "tau_max": gravity.max(axis=0),
            "ratio_unclipped_min": ratio.min(axis=0),
            "ratio_unclipped_max": ratio.max(axis=0),
            "bias_min": bias.min(axis=0),
            "bias_max": bias.max(axis=0),
            "clamp_hit_count": clamps.sum(axis=0),
            "clamp_percent": clamps.mean(axis=0) * 100.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-live-io", action="store_true")
    parser.add_argument("--read-timing-samples", type=int, default=30)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--measured-requests", type=int, default=30)
    parser.add_argument("--stop-after-robot-read", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise SystemExit(f"wrong Python interpreter: {sys.executable}")
    if not args.confirm_live_io:
        raise SystemExit("STEP 4A live I/O is blocked without --confirm-live-io")
    if min(args.read_timing_samples, args.measured_requests) <= 0 or args.warmup_requests < 0:
        raise SystemExit("sample counts must be positive")

    config = load_runtime_config(args.config)
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    ik = DLSInverseKinematicsV7(str(DEPLOY_DIR / "ik" / "urdf" / "so_arm_with_gopro_final.urdf"))
    gravity = GravityCompensator.from_runtime_config(config)
    motor = MotorIO(
        config.deployment["motor"]["port"],
        config.motor_calibration,
        config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=False,
    )
    camera_config = config.deployment["camera"]
    camera = LatestFrameCamera(
        camera_config["device"],
        camera_config["width"],
        camera_config["height"],
        camera_config["stale_timeout_s"],
    )
    remote_config = config.deployment["remote"]
    remote = RemoteSmolVLAClient(
        SubprocessSSHTransport(remote_config),
        remote_config["request_timeout_s"],
        remote_config["request_queue_size"],
        remote_config["response_queue_size"],
    )
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "archive" / "generated_results" / "deployment" / f"step4a_readonly_shadow_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "status": "RUNNING",
        "started_wall_time": datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "project_root": PROJECT_ROOT,
        "device": motor.port,
        "camera_device": camera.device,
        "remote_target": f"{remote_config['user']}@{remote_config['host']}:{remote_config['port']}",
        "checkpoint": remote_config["checkpoint"],
        "output_dir": output_dir,
    }
    records: list[dict[str, Any]] = []
    motor_started = camera_started = remote_started = False
    try:
        motor.connect(permit_motion=False, read_only=True)
        motor_started = True
        _zero_write_audit(motor)
        initial, initial_read_s, initial_meta = _read_actual(motor, mapping, ik)
        limits = _hard_limit_status(initial, ik)
        report["initial_robot"] = {
            "q_raw_deg": initial.raw_degrees,
            "q_urdf_deg": initial.q_urdf_degrees,
            "gripper_raw": initial_meta["gripper_raw"],
            "gripper_normalized": initial.state[5],
            "state": initial.state,
            "yaw_radians": initial.yaw_radians,
            "read_s": initial_read_s,
            "limits": limits,
            "gravity": _gravity_record(gravity, initial),
        }
        if not limits["hard_pass"]:
            raise RuntimeError(f"actual robot pose violates a physical joint limit: {limits['hard']}")

        read_timings = []
        for _ in range(args.read_timing_samples):
            _actual, duration, _metadata = _read_actual(motor, mapping, ik)
            read_timings.append(duration)
        report["present_position_timing_ms"] = _stats_ms(read_timings)
        _zero_write_audit(motor)
        if args.stop_after_robot_read:
            report["present_position_reads"] = motor.present_position_reads
            report["present_position_read_failures"] = motor.present_position_read_failures
            report["write_audit"] = _zero_write_audit(motor)
            report["status"] = "PASS"
            return 0

        camera.start()
        camera_started = True
        _wait_for_camera(camera)
        report["camera"] = _camera_rate_sample(camera)

        remote_started = True
        remote.start()
        response_timeout_s = float(remote_config["request_timeout_s"])
        stride_s = float(config.deployment["scheduler"]["request_stride_s"])
        next_request_at = time.monotonic()
        phases = ["first"] + ["warmup"] * args.warmup_requests + ["measured"] * args.measured_requests
        for request_id, phase in enumerate(phases):
            delay = next_request_at - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            record = _run_request(
                request_id,
                phase,
                motor,
                mapping,
                ik,
                gravity,
                camera,
                remote,
                response_timeout_s,
            )
            records.append(record)
            if record["reanchor_status"] != "VALID":
                raise RuntimeError(
                    f"request {request_id} reanchor was {record['reanchor_status']}"
                )
            next_request_at = max(next_request_at + stride_s, time.monotonic())

        report["summary"] = _summarize(records)
        report["requests_sent"] = len(records)
        report["responses_received"] = len(records)
        report["present_position_reads"] = motor.present_position_reads
        report["present_position_read_failures"] = motor.present_position_read_failures
        report["write_audit"] = _zero_write_audit(motor)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["requests_sent"] = len(records)
        report["responses_received"] = len(records)
        report["present_position_reads"] = motor.present_position_reads
        report["present_position_read_failures"] = motor.present_position_read_failures
        report["write_audit"] = asdict(motor.write_audit)
        raise
    finally:
        if remote_started:
            remote.close()
        if camera_started:
            camera.stop()
        if motor_started:
            motor.disconnect()
        report["finished_wall_time"] = datetime.now().astimezone().isoformat()
        report["write_audit_after_cleanup"] = asdict(motor.write_audit)
        (output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        with (output_dir / "request_telemetry.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, default=_json_default, separators=(",", ":")) + "\n")
        if records:
            with (output_dir / "timing.csv").open("x", newline="", encoding="utf-8") as stream:
                fields = [
                    "phase",
                    "request_id",
                    "t_frame",
                    "t_q_obs",
                    "t_obs",
                    "submitted_at",
                    "t_arr",
                    "arrival_q_timestamp",
                    "frame_age_s",
                    "image_state_abs_gap_s",
                    "observation_assembly_s",
                    "remote_request_response_s",
                    "arrival_read_s",
                    "reanchor_s",
                    "t_obs_to_reanchor_ready_s",
                    "reanchor_status",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    writer.writerow({field: record[field] for field in fields})
        print(json.dumps(report, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
