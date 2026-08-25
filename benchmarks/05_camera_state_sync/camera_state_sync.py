#!/usr/bin/env python3
"""
Benchmark 05 — Camera / State Sync Validation
==============================================
Evaluates timestamp synchronization and age between GoPro RGB frames and physical
SO-101 Present_Position readings during observation assembly.

Key Measurements:
  - Absolute Image-State Gap (|t_frame - t_state|)
  - Signed Image-State Gap (t_state - t_frame)
  - Frame Age (t_obs_done - t_frame)
  - State Age (t_obs_done - t_state)
  - Camera Read / Fetch Latency
  - Present_Position Read Latency
  - Full Observation Assembly Latency

Safety constraints:
  - MOTOR ACCESS = YES (READ ONLY)
  - MOTOR WRITE = NO (Goal_Position=0, Torque=0, PID=0, Calibration=0)
  - CAMERA ACCESS = YES (Read-only capture)
  - SSH / REMOTE INFERENCE = NO
  - Robot does NOT move (stationary follower)
  - READ-ONLY on existing project source files
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping
from control.motor_io import ARM_JOINTS, MotorIO
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.camera import CameraError, LatestFrameCamera
from inference.observation import build_observation, state_from_motor_sample


WARMUP_SAMPLES = 60
MEASURED_SAMPLES = 900
CADENCE_HZ = 30.0
CADENCE_PERIOD_S = 1.0 / CADENCE_HZ


@dataclass
class MetricStats:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max_val: float
    min_val: float


def compute_stats(values: Sequence[float]) -> MetricStats:
    if len(values) == 0:
        return MetricStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    return MetricStats(
        count=len(arr),
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max_val=float(np.max(arr)),
        min_val=float(np.min(arr)),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark 05: Camera / State Sync Validation")
    parser.add_argument("--warmup-samples", type=int, default=WARMUP_SAMPLES, help="Warmup iterations (default 60)")
    parser.add_argument("--measured-samples", type=int, default=MEASURED_SAMPLES, help="Measured iterations (default 900)")
    args = parser.parse_args()

    config_path = DEPLOY_DIR / "config" / "deployment.yaml"
    config = load_runtime_config(config_path)

    # 1. Initialize Kinematics & Mapping
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    urdf_path = str(DEPLOY_DIR / "urdf" / "so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path)

    # 2. Initialize Camera
    cam_cfg = config.deployment["camera"]
    camera = LatestFrameCamera(
        device=cam_cfg["device"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        stale_timeout_s=cam_cfg["stale_timeout_s"],
    )

    # 3. Initialize MotorIO in strict read-only mode
    motor = MotorIO(
        port=config.deployment["motor"]["port"],
        calibration_path=config.motor_calibration,
        pid=config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=False,
    )

    # Failure counters
    counters = {
        "camera_open_failures": 0,
        "camera_read_failures": 0,
        "empty_frames": 0,
        "invalid_frame_shape": 0,
        "duplicate_stale_candidates": 0,
        "present_position_read_failures": 0,
        "invalid_joint_reads": 0,
        "observation_assembly_exceptions": 0,
        "freshness_failures": 0,
    }

    print("=" * 70)
    print("BENCHMARK 05 — CAMERA / STATE SYNC VALIDATION")
    print("=" * 70)
    print(f"Connecting to motor bus on {motor.port} in strict READ-ONLY mode...")
    motor.connect(permit_motion=False, read_only=True)
    print(f"Motor connected: read_only={motor.read_only}, motion_permitted={motor.motion_permitted}")

    print(f"Starting camera (device={cam_cfg['device']}, {cam_cfg['width']}x{cam_cfg['height']} {cam_cfg['backend']})...")
    try:
        camera.start()
    except Exception as e:
        counters["camera_open_failures"] += 1
        print(f"CRITICAL: Failed to open camera: {e}")
        motor.disconnect()
        return

    # Let camera stream warm up and acquire initial frames
    time.sleep(1.0)

    # Data collectors (converted to ms for reporting)
    abs_gap_ms = []
    signed_gap_ms = []
    frame_age_ms = []
    state_age_ms = []
    cam_read_latency_ms = []
    state_read_latency_ms = []
    obs_assembly_latency_ms = []

    last_frame_seq = -1
    total_samples = args.warmup_samples + args.measured_samples
    print(f"Collecting {args.warmup_samples} warmup samples + {args.measured_samples} measured samples at {CADENCE_HZ} Hz...")

    loop_start = time.perf_counter()

    for idx in range(total_samples):
        target_tick_time = loop_start + idx * CADENCE_PERIOD_S
        
        t_obs_start = time.monotonic()

        # Step 1: Read Motor State (following canonical run_live._actual_state)
        t_state_read_start = time.monotonic()
        try:
            deg_sample = motor.read_present_positions(normalized=True)
            tick_sample = motor.read_present_positions(normalized=False)
            positions = {name: deg_sample.positions[name] for name in ARM_JOINTS}
            positions["gripper"] = tick_sample.positions["gripper"]
            t_state_sample = max(deg_sample.timestamp, tick_sample.timestamp)
            
            actual_state = state_from_motor_sample(
                positions,
                t_state_sample,
                mapping,
                ik_solver,
            )
        except Exception as e:
            counters["present_position_read_failures"] += 1
            continue
        t_state_read_end = time.monotonic()
        state_read_lat = (t_state_read_end - t_state_read_start) * 1000.0

        # Step 2: Fetch Latest Camera Frame (following canonical run_live._submit_if_due)
        t_cam_read_start = time.monotonic()
        try:
            frame = camera.latest(now=t_obs_start)
            if frame.image_rgb.shape != (256, 256, 3):
                counters["invalid_frame_shape"] += 1
            if frame.sequence == last_frame_seq:
                counters["duplicate_stale_candidates"] += 1
            last_frame_seq = frame.sequence
        except CameraError as ce:
            counters["freshness_failures"] += 1
            counters["camera_read_failures"] += 1
            continue
        except Exception as e:
            counters["camera_read_failures"] += 1
            continue
        t_cam_read_end = time.monotonic()
        cam_read_lat = (t_cam_read_end - t_cam_read_start) * 1000.0

        # Step 3: Assemble Observation
        try:
            snapshot = build_observation(idx, frame, actual_state)
        except Exception as e:
            counters["observation_assembly_exceptions"] += 1
            continue
        t_obs_done = time.monotonic()
        obs_assembly_lat = (t_obs_done - t_obs_start) * 1000.0

        # Timing definitions
        t_frame = frame.timestamp
        t_state = actual_state.timestamp

        # Signed gap: t_state - t_frame (positive if state is newer than camera frame)
        s_gap = (t_state - t_frame) * 1000.0
        a_gap = abs(s_gap)
        f_age = (t_obs_done - t_frame) * 1000.0
        st_age = (t_obs_done - t_state) * 1000.0

        # Record if past warmup
        if idx >= args.warmup_samples:
            abs_gap_ms.append(a_gap)
            signed_gap_ms.append(s_gap)
            frame_age_ms.append(f_age)
            state_age_ms.append(st_age)
            cam_read_latency_ms.append(cam_read_lat)
            state_read_latency_ms.append(state_read_lat)
            obs_assembly_latency_ms.append(obs_assembly_lat)

        # 30 Hz Cadence sleep
        elapsed = time.perf_counter() - target_tick_time
        sleep_dur = CADENCE_PERIOD_S - elapsed
        if sleep_dur > 0.001:
            time.sleep(sleep_dur)

    # Stop camera and motor cleanly
    camera.stop()
    motor.disconnect()

    write_audit = motor.write_audit

    # Compute Statistics
    stats_abs_gap = compute_stats(abs_gap_ms)
    stats_signed_gap = compute_stats(signed_gap_ms)
    stats_frame_age = compute_stats(frame_age_ms)
    stats_state_age = compute_stats(state_age_ms)
    stats_cam_lat = compute_stats(cam_read_latency_ms)
    stats_state_lat = compute_stats(state_read_latency_ms)
    stats_obs_lat = compute_stats(obs_assembly_latency_ms)

    all_writes_zero = (
        write_audit.goal_position_physical_writes == 0
        and write_audit.pid_physical_writes == 0
        and write_audit.torque_physical_writes == 0
        and write_audit.other_register_physical_writes == 0
    )

    # Print Summary Report
    print("\n" + "=" * 70)
    print("BENCHMARK 05 — RESULTS REPORT")
    print("=" * 70)
    print(f"CAMERA CONTRACT:")
    print(f"  Device / Backend        : {cam_cfg['device']} / {cam_cfg['backend']}")
    print(f"  Resolution              : {cam_cfg['width']}x{cam_cfg['height']} -> {cam_cfg['transport_width']}x{cam_cfg['transport_height']}")
    print(f"  Stale Timeout           : {cam_cfg['stale_timeout_s']} s")
    print("-" * 70)
    print(f"IMAGE / STATE SYNCHRONIZATION METRICS (ms):")
    print(f"{'Metric':<28} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'Absolute Image-State Gap':<28} {stats_abs_gap.count:>6} {stats_abs_gap.mean:>8.4f} {stats_abs_gap.p50:>8.4f} {stats_abs_gap.p95:>8.4f} {stats_abs_gap.p99:>8.4f} {stats_abs_gap.max_val:>8.4f}")
    print(f"{'Signed Gap (State - Frame)':<28} {stats_signed_gap.count:>6} {stats_signed_gap.mean:>8.4f} {stats_signed_gap.p50:>8.4f} {stats_signed_gap.p95:>8.4f} {stats_signed_gap.p99:>8.4f} {stats_signed_gap.max_val:>8.4f}")
    print(f"{'Frame Age @ Done':<28} {stats_frame_age.count:>6} {stats_frame_age.mean:>8.4f} {stats_frame_age.p50:>8.4f} {stats_frame_age.p95:>8.4f} {stats_frame_age.p99:>8.4f} {stats_frame_age.max_val:>8.4f}")
    print(f"{'State Age @ Done':<28} {stats_state_age.count:>6} {stats_state_age.mean:>8.4f} {stats_state_age.p50:>8.4f} {stats_state_age.p95:>8.4f} {stats_state_age.p99:>8.4f} {stats_state_age.max_val:>8.4f}")
    print(f"{'Camera Fetch Latency':<28} {stats_cam_lat.count:>6} {stats_cam_lat.mean:>8.4f} {stats_cam_lat.p50:>8.4f} {stats_cam_lat.p95:>8.4f} {stats_cam_lat.p99:>8.4f} {stats_cam_lat.max_val:>8.4f}")
    print(f"{'Motor Read Latency':<28} {stats_state_lat.count:>6} {stats_state_lat.mean:>8.4f} {stats_state_lat.p50:>8.4f} {stats_state_lat.p95:>8.4f} {stats_state_lat.p99:>8.4f} {stats_state_lat.max_val:>8.4f}")
    print(f"{'Observation Assembly Total':<28} {stats_obs_lat.count:>6} {stats_obs_lat.mean:>8.4f} {stats_obs_lat.p50:>8.4f} {stats_obs_lat.p95:>8.4f} {stats_obs_lat.p99:>8.4f} {stats_obs_lat.max_val:>8.4f}")
    print("-" * 70)
    print(f"SIGNED GAP RANGE: Min = {stats_signed_gap.min_val:.4f} ms, Max = {stats_signed_gap.max_val:.4f} ms")
    print("-" * 70)
    print(f"MOTOR WRITE AUDIT:")
    print(f"  Goal_Position physical writes : {write_audit.goal_position_physical_writes}")
    print(f"  PID physical writes           : {write_audit.pid_physical_writes}")
    print(f"  Torque physical writes        : {write_audit.torque_physical_writes}")
    print(f"  Other register writes         : {write_audit.other_register_physical_writes}")
    print("-" * 70)
    print(f"FAILURE COUNTERS:")
    for k, v in counters.items():
        print(f"  {k:<32}: {v}")
    print("=" * 70)
    print("VERDICT: CAMERA_STATE_SYNC_MEASUREMENT = COMPLETE")
    print("CANONICAL THRESHOLD: NOT DEFINED IN CONFIG (Audit reference provided)")
    print("=" * 70)


if __name__ == "__main__":
    main()
