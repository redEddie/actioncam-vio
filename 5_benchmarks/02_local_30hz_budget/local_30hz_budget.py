#!/usr/bin/env python3
"""
Benchmark 02 — Local 30 Hz Computation Budget Validation
=========================================================
Strictly evaluates local computation latency on SO-101 against the 33.333 ms budget.

Pipeline under test:
  Present_Position READ (bus timing excluded from primary timer)
  ↓ [T0]
  RAW → URDF degrees
  ↓
  Scheduler active target selection (ChunkScheduler.sample)
  ↓
  Trajectory interpolation (Linear interpolation on 10 Hz anchors)
  ↓
  Actual FK / yaw preparation (FK on current q_actual)
  ↓
  Inverse Kinematics (DLSInverseKinematicsV7 5D IK)
  ↓
  Gravity compensation (GravityCompensator evaluate)
  ↓
  BP+delta controller (compute_trajectory)
  ↓
  Safety checks (limits & finite verification)
  ↓
  Preview q_cmd generation
  ↓ [T1]

Safety constraints:
  - MOTOR READ ONLY (connect(read_only=True), MotorIO._ReadOnlyBusGuard)
  - Goal_Position write = 0
  - Torque write = 0
  - PID write = 0
  - Calibration write = 0
  - NO SSH / Remote calls
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
from scipy.spatial.transform import Rotation

# Ensure 4_deploy modules can be imported
DEPLOY_DIR = Path(__file__).resolve().parents[2] / "4_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import RuntimeConfig, load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ARM_JOINTS, MotorIO
from control.safety import (
    SafetyError,
    finite_vector,
    require_ik_success,
    require_joint_limits,
)
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.observation import ActualStateSnapshot, state_from_motor_sample
from trajectory.chunk_scheduler import ChunkScheduler
from trajectory.incremental import reconstruct_incremental_states


CONTROL_BUDGET_MS = 33.333333333333336
CONTROL_PERIOD_S = 1.0 / 30.0
WARMUP_TICKS = 90
MEASURED_TICKS = 900


@dataclass
class StageTiming:
    raw_to_urdf_ns: list[int]
    scheduler_ns: list[int]
    interpolation_ns: list[int]
    actual_fk_yaw_ns: list[int]
    ik_ns: list[int]
    gravity_ns: list[int]
    bp_delta_ns: list[int]
    safety_ns: list[int]
    full_local_compute_ns: list[int]
    motor_read_ns: list[int]


def compute_statistics(samples_ns: Sequence[int]) -> dict[str, float]:
    if not samples_ns:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(samples_ns, dtype=np.float64) / 1e6  # convert to ms
    return {
        "count": float(len(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def create_deterministic_validation_chunk(start_state: np.ndarray, num_steps: int = 15, dt: float = 0.1) -> np.ndarray:
    """
    Generate a bounded, smooth deterministic Cartesian variation around start_state.
    Shape: [15, 6] incremental deltas [dX, dY, dZ, dRoll, dPitch, dGripper]
    Total amplitude is kept within +/-1.5 cm and +/-2.0 deg to guarantee IK convergence.
    """
    actions = np.zeros((num_steps, 6), dtype=np.float64)
    # Small sinusoidal velocity profile for X, Y, Z, Roll, Pitch
    for i in range(num_steps):
        t = (i + 1) * dt
        # 0.01 m max displacement over 1.5s -> ~0.0006 m/step delta
        actions[i, 0] = 0.008 * np.sin(2.0 * np.pi * t / 1.5) * (dt / 1.5) * 2.0 * np.pi
        actions[i, 1] = 0.006 * np.cos(2.0 * np.pi * t / 1.5) * (dt / 1.5) * 2.0 * np.pi
        actions[i, 2] = -0.005 * np.sin(np.pi * t / 1.5) * (dt / 1.5) * np.pi
        actions[i, 3] = np.deg2rad(1.5) * np.sin(2.0 * np.pi * t / 1.5) * (dt / 1.5) * 2.0 * np.pi
        actions[i, 4] = np.deg2rad(1.5) * np.cos(2.0 * np.pi * t / 1.5) * (dt / 1.5) * 2.0 * np.pi
        actions[i, 5] = 0.0
    return actions


def main():
    parser = argparse.ArgumentParser(description="Benchmark 02: Local 30 Hz Computation Budget")
    parser.add_argument("--warmup-ticks", type=int, default=WARMUP_TICKS, help="Warmup iterations (default 90)")
    parser.add_argument("--measured-ticks", type=int, default=MEASURED_TICKS, help="Measured iterations (default 900)")
    args = parser.parse_args()

    config_path = DEPLOY_DIR / "config" / "deployment.yaml"
    config = load_runtime_config(config_path)

    # Instantiate canonical components
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    urdf_path = str(DEPLOY_DIR / "ik" / "urdf" / "so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path)
    gravity_compensator = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity_compensator)
    scheduler = ChunkScheduler(
        request_stride_s=config.deployment["scheduler"]["request_stride_s"],
        transition_overlap_s=config.deployment["scheduler"]["transition_overlap_s"],
        action_dt_s=config.action_period_s,
    )
    
    # MotorIO in strictly read-only mode
    motor = MotorIO(
        port=config.deployment["motor"]["port"],
        calibration_path=config.motor_calibration,
        pid=config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=False,
    )

    print("=" * 70)
    print("BENCHMARK 02 — LOCAL 30 HZ COMPUTATION BUDGET VALIDATION")
    print("=" * 70)
    print(f"Connecting to motor bus on {motor.port} in READ-ONLY mode...")
    motor.connect(permit_motion=False, read_only=True)
    if not motor.read_only:
        raise RuntimeError("CRITICAL: MotorIO is NOT in read-only mode!")
    print(f"Motor connected successfully: read_only={motor.read_only}, motion_permitted={motor.motion_permitted}")

    # Correctness counters
    counters = {
        "present_position_read_failures": 0,
        "raw_to_urdf_nonfinite": 0,
        "fk_failures": 0,
        "ik_attempts": 0,
        "ik_success": 0,
        "ik_failures": 0,
        "ik_nonfinite": 0,
        "physical_limit_violations": 0,
        "safe_limit_violations": 0,
        "gravity_nonfinite": 0,
        "controller_nonfinite": 0,
        "safety_rejects": 0,
    }

    def read_actual_state() -> ActualStateSnapshot:
        deg_sample = motor.read_present_positions(normalized=True)
        tick_sample = motor.read_present_positions(normalized=False)
        positions = {name: deg_sample.positions[name] for name in ARM_JOINTS}
        positions["gripper"] = tick_sample.positions["gripper"]
        return state_from_motor_sample(
            positions,
            max(deg_sample.timestamp, tick_sample.timestamp),
            mapping,
            ik_solver,
        )

    # Initial read to establish hold & seed
    initial_actual = read_actual_state()
    q_seed = np.deg2rad(initial_actual.q_urdf_degrees).copy()
    q_cmd_hold = controller.compute_hold(initial_actual.q_urdf_degrees, initial_actual.q_urdf_degrees)
    controller.begin_trajectory(initial_actual.q_urdf_degrees, initial_actual.q_urdf_degrees, q_cmd_hold)

    # Build deterministic trajectory chunk anchored at current actual state
    delta_actions = create_deterministic_validation_chunk(initial_actual.state, num_steps=15, dt=0.1)
    
    # Pre-populate scheduler with active trajectory chunks spanning the test duration
    total_ticks = args.warmup_ticks + args.measured_ticks
    test_duration_s = (total_ticks + 60) * CONTROL_PERIOD_S
    
    now_mono = time.monotonic()
    t_base = now_mono
    states_16 = reconstruct_incremental_states(initial_actual.state, delta_actions)
    times_16 = t_base + np.arange(16) * 0.1
    # Loop trajectory anchors to span entire duration
    full_times = []
    full_states = []
    cur_t = t_base
    while cur_t < t_base + test_duration_s:
        chunk_times = cur_t + np.arange(16) * 0.1
        full_times.extend(chunk_times)
        full_states.extend(states_16)
        cur_t += 1.5
    
    # Register active trajectory into scheduler
    scheduler.active_times = np.asarray(full_times, dtype=np.float64)
    scheduler.active_states = np.asarray(full_states, dtype=np.float64)
    scheduler.state = scheduler.state.VALID

    # Timing containers
    measured_timings = StageTiming(
        raw_to_urdf_ns=[],
        scheduler_ns=[],
        interpolation_ns=[],
        actual_fk_yaw_ns=[],
        ik_ns=[],
        gravity_ns=[],
        bp_delta_ns=[],
        safety_ns=[],
        full_local_compute_ns=[],
        motor_read_ns=[],
    )

    max_delta_q_cmd_deg = np.zeros(5, dtype=np.float64)
    max_delta_q_cmd_norm = 0.0
    prev_q_cmd_deg = q_cmd_hold.copy()

    print(f"\nStarting test: {args.warmup_ticks} warmup ticks + {args.measured_ticks} measured ticks at 30 Hz...")
    
    total_iterations = args.warmup_ticks + args.measured_ticks
    loop_start_time = time.perf_counter()

    for tick in range(total_iterations):
        tick_target_time = loop_start_time + tick * CONTROL_PERIOD_S
        
        # 1. READ MOTOR (Measured separately, outside FULL_LOCAL_COMPUTE)
        t_read_start = time.perf_counter_ns()
        try:
            deg_sample = motor.read_present_positions(normalized=True)
            tick_sample = motor.read_present_positions(normalized=False)
            positions = {name: deg_sample.positions[name] for name in ARM_JOINTS}
            positions["gripper"] = tick_sample.positions["gripper"]
            t_read_sample = max(deg_sample.timestamp, tick_sample.timestamp)
        except Exception as e:
            counters["present_position_read_failures"] += 1
            continue
        t_read_end = time.perf_counter_ns()
        motor_read_ns = t_read_end - t_read_start

        # ----------------------------------------------------
        # PRIMARY LOCAL COMPUTATION TIMING START [T0]
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()

        # Stage A: RAW → URDF
        t_stage_start = time.perf_counter_ns()
        raw_degrees = np.asarray([positions[name] for name in mapping.joint_order], dtype=np.float64)
        q_urdf_degrees = mapping.raw_degrees_to_urdf_degrees(raw_degrees)
        if not np.isfinite(q_urdf_degrees).all():
            counters["raw_to_urdf_nonfinite"] += 1
        t_a = time.perf_counter_ns()

        # Stage B: Scheduler target selection (query active trajectory)
        t_stage_start = t_a
        now_time = time.monotonic()
        sampled = scheduler.sample(now_time)
        t_b = time.perf_counter_ns()

        # Stage C: Interpolation
        t_stage_start = t_b
        target = sampled.get("action")
        if target is None:
            # Fallback if scheduler returned non-valid status
            target = initial_actual.state
        t_c = time.perf_counter_ns()

        # Stage D: Actual FK / Yaw preparation
        t_stage_start = t_c
        try:
            transform = np.asarray(ik_solver.forward_kinematics(np.deg2rad(q_urdf_degrees)), dtype=np.float64)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Invalid FK transform")
            yaw_rad, pitch_rad, roll_rad = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYX", degrees=False)
        except Exception:
            counters["fk_failures"] += 1
            yaw_rad = 0.0
        t_d = time.perf_counter_ns()

        # Stage E: Inverse Kinematics (DLS 5D IK)
        t_stage_start = t_d
        counters["ik_attempts"] += 1
        rotation = Rotation.from_euler("ZYX", [yaw_rad, target[4], target[3]]).as_matrix()
        q_nom_rad = ik_solver.solve(target[:3], rotation, q_seed)
        ik_converged = ik_solver.last_converged
        if not ik_converged:
            counters["ik_failures"] += 1
        else:
            counters["ik_success"] += 1
        if not np.isfinite(q_nom_rad).all():
            counters["ik_nonfinite"] += 1
        t_e = time.perf_counter_ns()

        # Stage F: Gravity compensation & support calculation
        t_stage_start = t_e
        try:
            grav_eval = gravity_compensator.evaluate(np.deg2rad(q_urdf_degrees))
            if not np.isfinite(grav_eval.total_support_bias_deg).all():
                counters["gravity_nonfinite"] += 1
        except Exception:
            counters["gravity_nonfinite"] += 1
        t_f = time.perf_counter_ns()

        # Stage G: BP+delta controller
        t_stage_start = t_f
        q_nom_deg = np.rad2deg(q_nom_rad)
        try:
            q_command_deg = controller.compute_trajectory(q_nom_deg, q_urdf_degrees)
            if not np.isfinite(q_command_deg).all():
                counters["controller_nonfinite"] += 1
        except Exception:
            counters["controller_nonfinite"] += 1
            q_command_deg = q_nom_deg.copy()
        t_g = time.perf_counter_ns()

        # Stage H: Safety gates & preview generation
        t_stage_start = t_g
        try:
            require_ik_success(q_nom_rad, ik_converged)
            require_joint_limits(q_nom_rad, ik_solver.joint_names, ik_solver.PHYSICAL_JOINT_LIMITS_RAD)
            require_joint_limits(q_nom_rad, ik_solver.joint_names, ik_solver.safe_limits)
            require_joint_limits(np.deg2rad(q_command_deg), ik_solver.joint_names, ik_solver.PHYSICAL_JOINT_LIMITS_RAD)
            require_joint_limits(np.deg2rad(q_command_deg), ik_solver.joint_names, ik_solver.safe_limits)
            
            # Goal preview conversion (RAW ticks) without writing to motor
            raw_degrees_cmd = mapping.urdf_degrees_to_raw_degrees(q_command_deg)
            goal_ticks = mapping.arm_degrees_to_raw_ticks(raw_degrees_cmd)
            goal_ticks["gripper"] = mapping.gripper_raw_from_width(float(target[5]))
        except SafetyError as se:
            counters["safety_rejects"] += 1
        t_h = time.perf_counter_ns()

        # ----------------------------------------------------
        # PRIMARY LOCAL COMPUTATION TIMING STOP [T1]
        # ----------------------------------------------------
        t1 = t_h
        full_compute_ns = t1 - t0

        # Update seed for next step
        q_seed = q_nom_rad.copy()

        # Record diagnostics
        delta_q = np.abs(q_command_deg - prev_q_cmd_deg)
        max_delta_q_cmd_deg = np.maximum(max_delta_q_cmd_deg, delta_q)
        max_delta_q_cmd_norm = max(max_delta_q_cmd_norm, float(np.linalg.norm(delta_q)))
        prev_q_cmd_deg = q_command_deg.copy()

        # Store timings if warmup is completed
        if tick >= args.warmup_ticks:
            measured_timings.raw_to_urdf_ns.append(t_a - t0)
            measured_timings.scheduler_ns.append(t_b - t_a)
            measured_timings.interpolation_ns.append(t_c - t_b)
            measured_timings.actual_fk_yaw_ns.append(t_d - t_c)
            measured_timings.ik_ns.append(t_e - t_d)
            measured_timings.gravity_ns.append(t_f - t_e)
            measured_timings.bp_delta_ns.append(t_g - t_f)
            measured_timings.safety_ns.append(t_h - t_g)
            measured_timings.full_local_compute_ns.append(full_compute_ns)
            measured_timings.motor_read_ns.append(motor_read_ns)

        # Precise 30 Hz Cadence pacing (sleep remaining time of the 33.333 ms slot)
        elapsed_loop_s = time.perf_counter() - tick_target_time
        sleep_s = CONTROL_PERIOD_S - elapsed_loop_s
        if sleep_s > 0.0005:
            time.sleep(sleep_s)

    # Disconnect motor cleanly
    try:
        motor.disconnect()
    except Exception:
        pass

    # Audit writes
    write_audit = motor.write_audit

    # Compute Statistics
    stats_full = compute_statistics(measured_timings.full_local_compute_ns)
    stats_raw = compute_statistics(measured_timings.raw_to_urdf_ns)
    stats_sched = compute_statistics(measured_timings.scheduler_ns)
    stats_interp = compute_statistics(measured_timings.interpolation_ns)
    stats_fk = compute_statistics(measured_timings.actual_fk_yaw_ns)
    stats_ik = compute_statistics(measured_timings.ik_ns)
    stats_grav = compute_statistics(measured_timings.gravity_ns)
    stats_bp = compute_statistics(measured_timings.bp_delta_ns)
    stats_safe = compute_statistics(measured_timings.safety_ns)
    stats_read = compute_statistics(measured_timings.motor_read_ns)

    # Deadline evaluation
    full_arr_ms = np.asarray(measured_timings.full_local_compute_ns, dtype=np.float64) / 1e6
    deadline_misses = int(np.sum(full_arr_ms > CONTROL_BUDGET_MS))
    deadline_miss_rate = (deadline_misses / len(full_arr_ms) * 100.0) if len(full_arr_ms) > 0 else 0.0
    worst_overrun = max(0.0, float(np.max(full_arr_ms) - CONTROL_BUDGET_MS)) if len(full_arr_ms) > 0 else 0.0

    p95_pass = stats_full["p95"] < CONTROL_BUDGET_MS
    all_writes_zero = (
        write_audit.goal_position_physical_writes == 0
        and write_audit.pid_physical_writes == 0
        and write_audit.torque_physical_writes == 0
        and write_audit.other_register_physical_writes == 0
    )
    overall_pass = p95_pass and all_writes_zero and (stats_full["count"] == args.measured_ticks)

    # Print Report
    print("\n" + "=" * 70)
    print("BENCHMARK 02 — RESULTS REPORT")
    print("=" * 70)
    print(f"FULL LOCAL COMPUTE (Budget = {CONTROL_BUDGET_MS:.3f} ms):")
    print(f"  Count : {int(stats_full['count'])}")
    print(f"  Mean  : {stats_full['mean']:.4f} ms")
    print(f"  P50   : {stats_full['p50']:.4f} ms")
    print(f"  P95   : {stats_full['p95']:.4f} ms (PASS target < 33.333 ms)")
    print(f"  P99   : {stats_full['p99']:.4f} ms")
    print(f"  Max   : {stats_full['max']:.4f} ms")
    print("-" * 70)
    print(f"STAGE BREAKDOWN (ms):")
    print(f"{'Stage':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'RAW→URDF':<20} {stats_raw['mean']:>8.4f} {stats_raw['p50']:>8.4f} {stats_raw['p95']:>8.4f} {stats_raw['p99']:>8.4f} {stats_raw['max']:>8.4f}")
    print(f"{'Scheduler':<20} {stats_sched['mean']:>8.4f} {stats_sched['p50']:>8.4f} {stats_sched['p95']:>8.4f} {stats_sched['p99']:>8.4f} {stats_sched['max']:>8.4f}")
    print(f"{'Interpolation':<20} {stats_interp['mean']:>8.4f} {stats_interp['p50']:>8.4f} {stats_interp['p95']:>8.4f} {stats_interp['p99']:>8.4f} {stats_interp['max']:>8.4f}")
    print(f"{'Actual FK/Yaw':<20} {stats_fk['mean']:>8.4f} {stats_fk['p50']:>8.4f} {stats_fk['p95']:>8.4f} {stats_fk['p99']:>8.4f} {stats_fk['max']:>8.4f}")
    print(f"{'IK':<20} {stats_ik['mean']:>8.4f} {stats_ik['p50']:>8.4f} {stats_ik['p95']:>8.4f} {stats_ik['p99']:>8.4f} {stats_ik['max']:>8.4f}")
    print(f"{'Gravity':<20} {stats_grav['mean']:>8.4f} {stats_grav['p50']:>8.4f} {stats_grav['p95']:>8.4f} {stats_grav['p99']:>8.4f} {stats_grav['max']:>8.4f}")
    print(f"{'BP+delta':<20} {stats_bp['mean']:>8.4f} {stats_bp['p50']:>8.4f} {stats_bp['p95']:>8.4f} {stats_bp['p99']:>8.4f} {stats_bp['max']:>8.4f}")
    print(f"{'Safety':<20} {stats_safe['mean']:>8.4f} {stats_safe['p50']:>8.4f} {stats_safe['p95']:>8.4f} {stats_safe['p99']:>8.4f} {stats_safe['max']:>8.4f}")
    print("-" * 70)
    print(f"INFORMATIONAL: Present_Position Read Latency (Mean={stats_read['mean']:.4f} ms, P95={stats_read['p95']:.4f} ms)")
    print("-" * 70)
    print(f"DEADLINE EVALUATION:")
    print(f"  Deadline Miss Count : {deadline_misses} / {len(full_arr_ms)}")
    print(f"  Deadline Miss Rate  : {deadline_miss_rate:.2f}%")
    print(f"  Worst Overrun       : {worst_overrun:.4f} ms")
    print("-" * 70)
    print(f"MOTOR WRITE AUDIT:")
    print(f"  Goal_Position physical writes : {write_audit.goal_position_physical_writes}")
    print(f"  PID physical writes           : {write_audit.pid_physical_writes}")
    print(f"  Torque physical writes        : {write_audit.torque_physical_writes}")
    print(f"  Other register writes         : {write_audit.other_register_physical_writes}")
    print("-" * 70)
    print(f"CORRECTNESS COUNTERS:")
    for k, v in counters.items():
        print(f"  {k:<32}: {v}")
    print("=" * 70)
    print(f"VERDICT: LOCAL_30HZ_BUDGET = {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    # Save summary JSON alongside for analysis generation
    results_payload = {
        "stats_full": stats_full,
        "stages": {
            "raw_to_urdf": stats_raw,
            "scheduler": stats_sched,
            "interpolation": stats_interp,
            "actual_fk_yaw": stats_fk,
            "ik": stats_ik,
            "gravity": stats_grav,
            "bp_delta": stats_bp,
            "safety": stats_safe,
            "motor_read": stats_read,
        },
        "deadline": {
            "budget_ms": CONTROL_BUDGET_MS,
            "miss_count": deadline_misses,
            "miss_rate_pct": deadline_miss_rate,
            "worst_overrun_ms": worst_overrun,
        },
        "write_audit": {
            "goal_position": write_audit.goal_position_physical_writes,
            "pid": write_audit.pid_physical_writes,
            "torque": write_audit.torque_physical_writes,
            "other_register": write_audit.other_register_physical_writes,
        },
        "counters": counters,
        "diagnostics": {
            "max_delta_q_cmd_deg": max_delta_q_cmd_deg.tolist(),
            "max_delta_q_cmd_norm": max_delta_q_cmd_norm,
        },
        "verdict": "PASS" if overall_pass else "FAIL",
    }
    
    out_json = Path(__file__).resolve().parent / "benchmark_02_results.json"
    out_json.write_text(json.dumps(results_payload, indent=2))
    print(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
