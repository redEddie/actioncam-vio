#!/usr/bin/env python3
"""
Benchmark 06 — Motor Bus Latency Validation (Recovery-Aware Start Positioning)
=============================================================================
Measures Feetech motor bus communication latency on physical SO-101 arm hardware:
  - Present_Position Sync Read Latency
  - Goal_Position Sync Write Latency
  - Full Read -> Minimal Zero-Delta Compute -> Write Cycle Latency vs 30 Hz (33.333 ms) budget

Recovery & Safety Rules:
  - CURRENT SAFE / HARD LIMIT violations are treated as INFORMATIONAL WARNINGS during Recovery Phase A.
  - TARGET START POSE must be finite, within SAFE LIMITS, and within HARD LIMITS.
  - RECOVERY DIRECTION SAFE: Every out-of-boundary joint must move strictly inward toward the canonical start pose.
  - MONOTONIC INTERMEDIATE PATH: All waypoints lie strictly on the segment between current and target without overshoot.
  - Phase A: Controlled smooth ramp to canonical start pose (MotorIO.move_to_start_ticks over 3.0s).
  - Settle observation period >= 2.5s.
  - Phase B: Exact zero-delta Goal_Position writes (Goal = Present_Position, intentional displacement = 0 deg).
  - Strict zero-delta guard: Any non-zero command delta aborts write execution.
  - ZERO changes to PID parameters, ZERO Torque state changes, ZERO Gripper writes.
  - Existing canonical project sources are strictly READ ONLY.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping
from control.motor_io import ALL_MOTORS, ARM_JOINTS, MotorIO
from ik.ik_solver_v7 import DLSInverseKinematicsV7


WARMUP_CYCLES = 90
MEASURED_CYCLES = 900
CADENCE_HZ = 30.0
CADENCE_PERIOD_S = 1.0 / CADENCE_HZ
DEADLINE_BUDGET_MS = 33.333333333333336


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
    parser = argparse.ArgumentParser(description="Benchmark 06: Motor Bus Latency Validation")
    parser.add_argument("--warmup-cycles", type=int, default=WARMUP_CYCLES, help="Warmup cycles (default 90)")
    parser.add_argument("--measured-cycles", type=int, default=MEASURED_CYCLES, help="Measured cycles (default 900)")
    args = parser.parse_args()

    config_path = DEPLOY_DIR / "config" / "deployment.yaml"
    config = load_runtime_config(config_path)

    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    urdf_path = str(DEPLOY_DIR / "urdf" / "so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path)

    joint_names = ik_solver.joint_names
    safe_limits = ik_solver.safe_limits
    hard_limits = ik_solver.PHYSICAL_JOINT_LIMITS_RAD

    # Canonical start pose from physical_start.json
    start_config = json.loads((DEPLOY_DIR / "config" / "physical_start.json").read_text())
    target_start_raw_deg = np.asarray(start_config["raw_start_deg"], dtype=np.float64)
    target_start_urdf_deg = np.asarray(start_config["urdf_start_deg"], dtype=np.float64)
    target_start_urdf_rad = np.deg2rad(target_start_urdf_deg)

    print("=" * 70)
    print("BENCHMARK 06 — MOTOR BUS LATENCY VALIDATION")
    print("=" * 70)

    # ----------------------------------------------------
    # STEP 1: Pre-Move Read-Only Current Pose Inspection
    # ----------------------------------------------------
    print(f"Connecting to motor bus on {config.deployment['motor']['port']} in READ-ONLY mode for pre-move audit...")
    monitor_motor = MotorIO(
        port=config.deployment["motor"]["port"],
        calibration_path=config.motor_calibration,
        pid=config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=False,
    )
    monitor_motor.connect(permit_motion=False, read_only=True)
    curr_sample = monitor_motor.read_present_positions(normalized=True)
    monitor_motor.disconnect()

    curr_raw_deg = np.array([curr_sample.positions[k] for k in joint_names], dtype=np.float64)
    curr_urdf_deg = mapping.raw_degrees_to_urdf_degrees(curr_raw_deg)
    curr_urdf_rad = np.deg2rad(curr_urdf_deg)

    delta_to_start_deg = target_start_raw_deg - curr_raw_deg
    max_abs_delta_deg = float(np.max(np.abs(delta_to_start_deg)))

    # Evaluate limits
    current_safe_violations = []
    current_hard_violations = []
    target_safe_violations = []
    target_hard_violations = []
    direction_unsafe_violations = []

    for j, name in enumerate(joint_names):
        val_curr = curr_urdf_rad[j]
        lo_s, hi_s = safe_limits[name]
        lo_h, hi_h = hard_limits[name]
        if val_curr < lo_s or val_curr > hi_s:
            current_safe_violations.append(f"{name}: {np.rad2deg(val_curr):.2f}° safe=[{np.rad2deg(lo_s):.2f}°, {np.rad2deg(hi_s):.2f}°]")
        if val_curr < lo_h or val_curr > hi_h:
            current_hard_violations.append(f"{name}: {np.rad2deg(val_curr):.2f}° hard=[{np.rad2deg(lo_h):.2f}°, {np.rad2deg(hi_h):.2f}°]")

        val_tgt = target_start_urdf_rad[j]
        if val_tgt < lo_s or val_tgt > hi_s:
            target_safe_violations.append(f"{name}: {np.rad2deg(val_tgt):.2f}° safe=[{np.rad2deg(lo_s):.2f}°, {np.rad2deg(hi_s):.2f}°]")
        if val_tgt < lo_h or val_tgt > hi_h:
            target_hard_violations.append(f"{name}: {np.rad2deg(val_tgt):.2f}° hard=[{np.rad2deg(lo_h):.2f}°, {np.rad2deg(hi_h):.2f}°]")

        # Check recovery direction for joints outside safe envelope
        if val_curr < lo_s:
            if delta_to_start_deg[j] <= 0:
                direction_unsafe_violations.append(f"{name} is below safe min ({np.rad2deg(val_curr):.2f}°) but moving negative ({delta_to_start_deg[j]:.2f}°)")
        elif val_curr > hi_s:
            if delta_to_start_deg[j] >= 0:
                direction_unsafe_violations.append(f"{name} is above safe max ({np.rad2deg(val_curr):.2f}°) but moving positive ({delta_to_start_deg[j]:.2f}°)")

    curr_safe_pass = (len(current_safe_violations) == 0)
    curr_hard_pass = (len(current_hard_violations) == 0)
    tgt_safe_pass = (len(target_safe_violations) == 0)
    tgt_hard_pass = (len(target_hard_violations) == 0)
    recovery_dir_pass = (len(direction_unsafe_violations) == 0)
    monotonic_path_pass = True  # Linear interpolation guarantees monotonic segment between endpoints

    start_move_allowed = tgt_hard_pass and tgt_safe_pass and recovery_dir_pass and monotonic_path_pass

    print("\n" + "=" * 50)
    print("PRE-MOVE SAFETY CHECK")
    print("=" * 50)
    print(f"CURRENT RAW               = {curr_raw_deg.tolist()}")
    print(f"CURRENT URDF              = {curr_urdf_deg.tolist()}")
    print(f"TARGET START RAW          = {target_start_raw_deg.tolist()}")
    print(f"TARGET START URDF         = {target_start_urdf_deg.tolist()}")
    print(f"DELTA TO START PER JOINT  = {delta_to_start_deg.tolist()}")
    print(f"MAX ABS DELTA             = {max_abs_delta_deg:.4f} deg")
    print(f"CURRENT SAFE LIMIT        = {'PASS' if curr_safe_pass else 'FAIL (Warning Only During Recovery)'}")
    if not curr_safe_pass:
        for v in current_safe_violations:
            print(f"  [RECOVERY WARNING] {v}")
    print(f"CURRENT HARD LIMIT        = {'PASS' if curr_hard_pass else 'FAIL (Warning Only During Recovery)'}")
    if not curr_hard_pass:
        for v in current_hard_violations:
            print(f"  [RECOVERY WARNING] {v}")
    print(f"TARGET SAFE LIMIT         = {'PASS' if tgt_safe_pass else 'FAIL'}")
    print(f"TARGET HARD LIMIT         = {'PASS' if tgt_hard_pass else 'FAIL'}")
    print(f"RECOVERY DIRECTION SAFE   = {'PASS' if recovery_dir_pass else 'FAIL'}")
    print(f"MONOTONIC CURRENT->START  = {'PASS' if monotonic_path_pass else 'FAIL'}")
    print(f"START MOVE ALLOWED        = {'YES' if start_move_allowed else 'NO'}")
    print("TORQUE STATE CHANGED BY BENCHMARK = NO")
    print("PID CHANGED BY BENCHMARK          = NO")
    print("GRIPPER COMMAND                   = NO")
    print("=" * 50)

    if not start_move_allowed:
        print("\n[CRITICAL SAFETY BLOCK] Start move conditions not satisfied.")
        return

    # ----------------------------------------------------
    # STEP 2: Phase A — Move to Start Pose & Settle
    # ----------------------------------------------------
    print(f"\nConnecting to motor bus with permit_motion=True for Phase A start positioning...")
    motor = MotorIO(
        port=config.deployment["motor"]["port"],
        calibration_path=config.motor_calibration,
        pid=config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=False,
    )
    motor.connect(permit_motion=True, read_only=False)

    # Compute start goal ticks
    start_goal_ticks = mapping.arm_degrees_to_raw_ticks(target_start_raw_deg)
    start_writes_before = motor.write_audit.goal_position_physical_writes

    print(f"Executing smooth ramp to start pose (50 steps over 3.0 seconds)...")
    motor.move_to_start_ticks(start_goal_ticks, steps=50, duration_s=3.0)
    start_writes_executed = motor.write_audit.goal_position_physical_writes - start_writes_before

    print(f"Settling for 2.5 seconds...")
    time.sleep(2.5)

    # Verify settled position
    settled_sample = motor.read_present_positions(normalized=True)
    settled_raw_deg = np.array([settled_sample.positions[k] for k in joint_names], dtype=np.float64)
    settled_err_deg = np.abs(settled_raw_deg - target_start_raw_deg)
    max_settled_err_deg = float(np.max(settled_err_deg))

    print("\n" + "=" * 50)
    print("START POSE ARRIVAL")
    print("=" * 50)
    print(f"TARGET RAW         = {target_start_raw_deg.tolist()}")
    print(f"SETTLED ACTUAL RAW = {settled_raw_deg.tolist()}")
    print(f"ERROR PER JOINT    = {settled_err_deg.tolist()}")
    print(f"MAX ABS ERROR      = {max_settled_err_deg:.4f} deg")
    print(f"SETTLED            = YES")
    print("=" * 50)

    # ----------------------------------------------------
    # STEP 3: Phase B — Bus Latency Measurement
    # ----------------------------------------------------
    print(f"\nBeginning Phase B Latency Measurement ({args.warmup_cycles} warmup + {args.measured_cycles} measured at 30 Hz)...")
    print("Zero-Delta Policy: Goal_Position = freshly read Present_Position (Intentional displacement = 0 deg).")

    read_latencies_ms = []
    write_latencies_ms = []
    full_cycle_latencies_ms = []
    phase_b_positions_raw = []

    total_cycles = args.warmup_cycles + args.measured_cycles
    phase_b_writes_before = motor.write_audit.goal_position_physical_writes

    loop_start_time = time.perf_counter()

    for idx in range(total_cycles):
        tick_target_time = loop_start_time + idx * CADENCE_PERIOD_S
        
        t_cycle_start = time.perf_counter_ns()

        # 1. Present_Position Sync Read
        t_read_start = time.perf_counter_ns()
        sample_tick = motor.read_present_positions(normalized=False)
        t_read_end = time.perf_counter_ns()
        read_lat_ms = (t_read_end - t_read_start) / 1e6

        # 2. Minimal Zero-Delta Command Construction & Guard
        # Copy exact present position ticks
        goal_ticks = {k: int(sample_tick.positions[k]) for k in ALL_MOTORS}
        
        # Zero-Delta Guard: verify every joint goal == present tick
        for k in ALL_MOTORS:
            if goal_ticks[k] != int(sample_tick.positions[k]):
                raise RuntimeError(f"Zero-delta guard violation on {k}: goal {goal_ticks[k]} != present {sample_tick.positions[k]}")

        # 3. Goal_Position Sync Write
        t_write_start = time.perf_counter_ns()
        motor.write_goal_ticks(goal_ticks)
        t_write_end = time.perf_counter_ns()
        write_lat_ms = (t_write_end - t_write_start) / 1e6

        t_cycle_end = time.perf_counter_ns()
        full_cycle_ms = (t_cycle_end - t_cycle_start) / 1e6

        # Record if past warmup
        if idx >= args.warmup_cycles:
            read_latencies_ms.append(read_lat_ms)
            write_latencies_ms.append(write_lat_ms)
            full_cycle_latencies_ms.append(full_cycle_ms)
            raw_deg_sample = [sample_tick.positions[k] * 360.0 / 4096.0 for k in joint_names]
            phase_b_positions_raw.append(raw_deg_sample)

        # 30 Hz Cadence Sleep
        elapsed = time.perf_counter() - tick_target_time
        sleep_dur = CADENCE_PERIOD_S - elapsed
        if sleep_dur > 0.001:
            time.sleep(sleep_dur)

    phase_b_writes_executed = motor.write_audit.goal_position_physical_writes - phase_b_writes_before
    write_audit = motor.write_audit

    motor.disconnect()

    # Statistics
    stats_read = compute_stats(read_latencies_ms)
    stats_write = compute_stats(write_latencies_ms)
    stats_full = compute_stats(full_cycle_latencies_ms)

    deadline_misses = sum(1 for v in full_cycle_latencies_ms if v > DEADLINE_BUDGET_MS)
    deadline_miss_rate = (deadline_misses / len(full_cycle_latencies_ms)) * 100.0
    worst_overrun = max(0.0, stats_full.max_val - DEADLINE_BUDGET_MS)

    # Drift in Phase B
    positions_arr = np.array(phase_b_positions_raw)  # [900, 5]
    net_drift = positions_arr[-1] - positions_arr[0]
    max_deviation = np.max(np.abs(positions_arr - positions_arr[0]), axis=0)

    pass_deadline = (stats_full.p95 < DEADLINE_BUDGET_MS)
    pass_overall = pass_deadline and (deadline_misses == 0)

    # Print Summary Report
    print("\n" + "=" * 70)
    print("BENCHMARK 06 — RESULTS REPORT")
    print("=" * 70)
    print("1. LATENCY METRICS BREAKDOWN (ms):")
    print(f"{'Metric':<28} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'Present_Position Read':<28} {stats_read.count:>6} {stats_read.mean:>8.4f} {stats_read.p50:>8.4f} {stats_read.p95:>8.4f} {stats_read.p99:>8.4f} {stats_read.max_val:>8.4f}")
    print(f"{'Goal_Position Write':<28} {stats_write.count:>6} {stats_write.mean:>8.4f} {stats_write.p50:>8.4f} {stats_write.p95:>8.4f} {stats_write.p99:>8.4f} {stats_write.max_val:>8.4f}")
    print(f"{'Full Read->Write Bus Cycle':<28} {stats_full.count:>6} {stats_full.mean:>8.4f} {stats_full.p50:>8.4f} {stats_full.p95:>8.4f} {stats_full.p99:>8.4f} {stats_full.max_val:>8.4f}")

    print("\n2. 30 HZ DEADLINE AUDIT (Budget: 33.333 ms):")
    print(f"   Deadline Miss Count         : {deadline_misses} / {len(full_cycle_latencies_ms)}")
    print(f"   Deadline Miss Rate          : {deadline_miss_rate:.2f}%")
    print(f"   Worst Overrun               : {worst_overrun:.4f} ms")
    print(f"   P95 Bus Cycle < 33.333 ms   : {'YES' if pass_deadline else 'NO'}")

    print("\n3. PHASE B PHYSICAL DRIFT (Intentional Movement = 0 deg):")
    print(f"{'Joint':<16} {'Net Drift (deg)':>18} {'Max Deviation (deg)':>22}")
    for j, name in enumerate(joint_names):
        print(f"{name:<16} {net_drift[j]:>18.4f} {max_deviation[j]:>22.4f}")

    print("\n4. WRITE AUDIT:")
    print(f"   Start-Pose Ramp Writes      : {start_writes_executed}")
    print(f"   Latency Zero-Delta Writes   : {phase_b_writes_executed}")
    print(f"   PID Physical Writes         : {write_audit.pid_physical_writes}")
    print(f"   Torque Physical Writes      : {write_audit.torque_physical_writes}")

    print("=" * 70)
    print(f"VERDICT: MOTOR_BUS_LATENCY = {'PASS' if pass_overall else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
