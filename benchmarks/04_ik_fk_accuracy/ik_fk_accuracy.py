#!/usr/bin/env python3
"""
Benchmark 04 — IK / FK Accuracy Validation
===========================================
Strictly evaluates canonical DLSInverseKinematicsV7 accuracy, precision, limits,
convergence robustness, and joint continuity via FK -> IK -> FK reconstruction.

Tests:
  - Test A: Exact-Seed Reconstruction (500 safe-interior samples, seed = q_ref)
  - Test B: Perturbed-Seed Reconstruction (500 safe-interior samples, seed = q_ref +/- 2 deg)
  - Test C: Smooth Cartesian Trajectory Continuity (300 targets, previous solution seed)

Safety constraints:
  - MOTOR ACCESS = NO
  - PRESENT_POSITION READ = NO
  - MOTOR WRITE = NO
  - /dev/ttyACM0 ACCESS = NO
  - CAMERA ACCESS = NO
  - SSH / REMOTE CALLS = NO
  - Existing canonical sources are strictly READ ONLY
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from ik.ik_solver_v7 import DLSInverseKinematicsV7
from control.safety import require_ik_success, require_joint_limits


RNG_SEED = 42
N_SAMPLES_AB = 500
N_TRAJECTORY_C = 300
PERTURBATION_DEG = 2.0
LARGE_JUMP_THRESHOLD_DEG = 10.0


@dataclass
class AccuracyStats:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max_val: float


def compute_stats(values: Sequence[float]) -> AccuracyStats:
    if len(values) == 0:
        return AccuracyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    return AccuracyStats(
        count=len(arr),
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max_val=float(np.max(arr)),
    )


def generate_safe_interior_samples(
    safe_limits: Mapping[str, tuple[float, float]],
    joint_names: Sequence[str],
    n_samples: int,
    interior_ratio: float = 0.80,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Generate deterministic joint configurations strictly inside the inner 80% safe envelope."""
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, len(joint_names)), dtype=np.float64)
    
    for j, name in enumerate(joint_names):
        lo, hi = safe_limits[name]
        center = (lo + hi) / 2.0
        half_span = (hi - lo) / 2.0
        margin = half_span * interior_ratio
        samples[:, j] = rng.uniform(center - margin, center + margin, size=n_samples)
        
    return samples


def main():
    urdf_path = str(DEPLOY_DIR / "urdf" / "so_arm_with_gopro_final.urdf")
    solver = DLSInverseKinematicsV7(urdf_path)

    joint_names = solver.joint_names
    safe_limits = solver.safe_limits
    hard_limits = solver.PHYSICAL_JOINT_LIMITS_RAD

    print("=" * 70)
    print("BENCHMARK 04 — IK / FK ACCURACY VALIDATION")
    print("=" * 70)
    print("Solver Contract & Configuration:")
    print(f"  Joint Names           : {joint_names}")
    print(f"  Max Iterations        : 100")
    print(f"  Position Tolerance    : 5e-4 m (0.5 mm)")
    print(f"  Orientation Tolerance : 1e-3 rad (0.057 deg)")
    print(f"  Damping Factor        : {solver.damping}")
    print(f"  Safe Limits Margin    : 90% of Physical Limits")
    print("-" * 70)

    # ----------------------------------------------------
    # Generate Test Samples
    # ----------------------------------------------------
    q_refs = generate_safe_interior_samples(safe_limits, joint_names, N_SAMPLES_AB, interior_ratio=0.80, seed=RNG_SEED)

    # Failure counters
    counters = {
        "test_a_ik_attempts": 0,
        "test_a_ik_success": 0,
        "test_a_ik_failures": 0,
        "test_b_ik_attempts": 0,
        "test_b_ik_success": 0,
        "test_b_ik_failures": 0,
        "test_c_ik_attempts": 0,
        "test_c_ik_success": 0,
        "test_c_ik_failures": 0,
        "hard_limit_violations": 0,
        "safe_limit_violations": 0,
        "ik_nonfinite": 0,
        "fk_nonfinite": 0,
        "tolerance_mismatches": 0,
        "large_joint_jumps": 0,
    }

    # ====================================================
    # TEST A: Exact-Seed Reconstruction
    # ====================================================
    pos_errs_a_mm = []
    rot_errs_a_deg = []
    runtimes_a_ms = []

    for i in range(N_SAMPLES_AB):
        q_ref = q_refs[i]
        T_target = solver.forward_kinematics(q_ref)
        if not np.isfinite(T_target).all():
            counters["fk_nonfinite"] += 1
            continue
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]

        counters["test_a_ik_attempts"] += 1
        t_start = time.perf_counter_ns()
        q_sol = solver.solve(p_target, R_target, q_ref)
        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
        runtimes_a_ms.append(t_elapsed_ms)

        if not np.isfinite(q_sol).all():
            counters["ik_nonfinite"] += 1
            counters["test_a_ik_failures"] += 1
            continue

        if solver.last_converged:
            counters["test_a_ik_success"] += 1
        else:
            counters["test_a_ik_failures"] += 1

        # Check limits
        for j, name in enumerate(joint_names):
            if q_sol[j] < hard_limits[name][0] or q_sol[j] > hard_limits[name][1]:
                counters["hard_limit_violations"] += 1
            if q_sol[j] < safe_limits[name][0] or q_sol[j] > safe_limits[name][1]:
                counters["safe_limit_violations"] += 1

        # Evaluate FK reconstruction
        T_rec = solver.forward_kinematics(q_sol)
        p_rec = T_rec[:3, 3]
        R_rec = T_rec[:3, :3]

        pos_err_m = float(np.linalg.norm(p_target - p_rec))
        # 5D task rotation error in robot base frame (Roll & Pitch)
        R_err = R_target @ R_rec.T
        e_rot_5d = Rotation.from_matrix(R_err).as_rotvec()[:2]
        rot_err_rad = float(np.linalg.norm(e_rot_5d))

        pos_errs_a_mm.append(pos_err_m * 1000.0)
        rot_errs_a_deg.append(np.rad2deg(rot_err_rad))

        if solver.last_converged and (pos_err_m > 5e-4 or rot_err_rad > 1e-3):
            counters["tolerance_mismatches"] += 1

    # ====================================================
    # TEST B: Perturbed-Seed Reconstruction (+/- 2 deg)
    # ====================================================
    rng_b = np.random.default_rng(RNG_SEED + 100)
    perturb_rad = np.deg2rad(PERTURBATION_DEG)
    perturbations = rng_b.uniform(-perturb_rad, perturb_rad, size=(N_SAMPLES_AB, len(joint_names)))

    pos_errs_b_mm = []
    rot_errs_b_deg = []
    runtimes_b_ms = []

    for i in range(N_SAMPLES_AB):
        q_ref = q_refs[i]
        q_seed_perturbed = q_ref + perturbations[i]
        # Clamp seed to safe limits just in case
        for j, name in enumerate(joint_names):
            q_seed_perturbed[j] = np.clip(q_seed_perturbed[j], safe_limits[name][0], safe_limits[name][1])

        T_target = solver.forward_kinematics(q_ref)
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]

        counters["test_b_ik_attempts"] += 1
        t_start = time.perf_counter_ns()
        q_sol = solver.solve(p_target, R_target, q_seed_perturbed)
        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
        runtimes_b_ms.append(t_elapsed_ms)

        if not np.isfinite(q_sol).all():
            counters["ik_nonfinite"] += 1
            counters["test_b_ik_failures"] += 1
            continue

        if solver.last_converged:
            counters["test_b_ik_success"] += 1
        else:
            counters["test_b_ik_failures"] += 1

        for j, name in enumerate(joint_names):
            if q_sol[j] < hard_limits[name][0] or q_sol[j] > hard_limits[name][1]:
                counters["hard_limit_violations"] += 1
            if q_sol[j] < safe_limits[name][0] or q_sol[j] > safe_limits[name][1]:
                counters["safe_limit_violations"] += 1

        T_rec = solver.forward_kinematics(q_sol)
        p_rec = T_rec[:3, 3]
        R_rec = T_rec[:3, :3]

        pos_err_m = float(np.linalg.norm(p_target - p_rec))
        R_err = R_target @ R_rec.T
        e_rot_5d = Rotation.from_matrix(R_err).as_rotvec()[:2]
        rot_err_rad = float(np.linalg.norm(e_rot_5d))

        pos_errs_b_mm.append(pos_err_m * 1000.0)
        rot_errs_b_deg.append(np.rad2deg(rot_err_rad))

        if solver.last_converged and (pos_err_m > 5e-4 or rot_err_rad > 1e-3):
            counters["tolerance_mismatches"] += 1

    # ====================================================
    # TEST C: Smooth Trajectory Continuity (300 steps)
    # ====================================================
    # Safe envelope center configuration as specified in Section 25
    q0_safe = np.zeros(len(joint_names), dtype=np.float64)
    for j, name in enumerate(joint_names):
        lo, hi = safe_limits[name]
        q0_safe[j] = (lo + hi) / 2.0
    T0 = solver.forward_kinematics(q0_safe)
    p0 = T0[:3, 3]
    yaw0, pitch0, roll0 = Rotation.from_matrix(T0[:3, :3]).as_euler("ZYX", degrees=False)

    pos_errs_c_mm = []
    rot_errs_c_deg = []
    runtimes_c_ms = []
    joint_trajectory_deg = []

    current_seed = q0_safe.copy()

    for step in range(N_TRAJECTORY_C):
        t = step * 0.03333333333333333  # 30 Hz step
        # Smooth bounded sinusoidal trajectory: +/- 8 mm XYZ, +/- 1.5 deg Roll/Pitch
        delta_p = np.array([
            0.008 * np.sin(2.0 * np.pi * t / 4.0),
            0.006 * np.cos(2.0 * np.pi * t / 4.0),
            0.005 * np.sin(np.pi * t / 4.0),
        ])
        target_pos = p0 + delta_p
        target_roll = roll0 + np.deg2rad(1.5) * np.sin(2.0 * np.pi * t / 4.0)
        target_pitch = pitch0 + np.deg2rad(1.5) * np.cos(2.0 * np.pi * t / 4.0)
        target_rot = Rotation.from_euler("ZYX", [yaw0, target_pitch, target_roll]).as_matrix()

        counters["test_c_ik_attempts"] += 1
        t_start = time.perf_counter_ns()
        q_sol = solver.solve(target_pos, target_rot, current_seed)
        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6
        runtimes_c_ms.append(t_elapsed_ms)

        if not np.isfinite(q_sol).all():
            counters["ik_nonfinite"] += 1
            counters["test_c_ik_failures"] += 1
            continue

        if solver.last_converged:
            counters["test_c_ik_success"] += 1
        else:
            counters["test_c_ik_failures"] += 1

        for j, name in enumerate(joint_names):
            if q_sol[j] < hard_limits[name][0] or q_sol[j] > hard_limits[name][1]:
                counters["hard_limit_violations"] += 1
            if q_sol[j] < safe_limits[name][0] or q_sol[j] > safe_limits[name][1]:
                counters["safe_limit_violations"] += 1

        T_rec = solver.forward_kinematics(q_sol)
        p_rec = T_rec[:3, 3]
        R_rec = T_rec[:3, :3]

        pos_err_m = float(np.linalg.norm(target_pos - p_rec))
        R_err = target_rot @ R_rec.T
        e_rot_5d = Rotation.from_matrix(R_err).as_rotvec()[:2]
        rot_err_rad = float(np.linalg.norm(e_rot_5d))

        pos_errs_c_mm.append(pos_err_m * 1000.0)
        rot_errs_c_deg.append(np.rad2deg(rot_err_rad))
        joint_trajectory_deg.append(np.rad2deg(q_sol))

        # Update seed for next step
        current_seed = q_sol.copy()

    # Calculate continuity metrics on Test C
    joint_traj_arr = np.array(joint_trajectory_deg)  # [300, 5]
    joint_diffs_deg = np.abs(joint_traj_arr[1:] - joint_traj_arr[:-1])  # [299, 5]
    vector_diffs_deg = np.linalg.norm(joint_diffs_deg, axis=1)  # [299]

    large_jumps = int(np.sum(joint_diffs_deg > LARGE_JUMP_THRESHOLD_DEG))
    counters["large_joint_jumps"] = large_jumps

    # Statistics
    stats_pos_a = compute_stats(pos_errs_a_mm)
    stats_pos_b = compute_stats(pos_errs_b_mm)
    stats_pos_c = compute_stats(pos_errs_c_mm)

    stats_rot_a = compute_stats(rot_errs_a_deg)
    stats_rot_b = compute_stats(rot_errs_b_deg)
    stats_rot_c = compute_stats(rot_errs_c_deg)

    stats_time_a = compute_stats(runtimes_a_ms)
    stats_time_b = compute_stats(runtimes_b_ms)
    stats_time_c = compute_stats(runtimes_c_ms)

    per_joint_continuity = [compute_stats(joint_diffs_deg[:, j]) for j in range(len(joint_names))]
    vec_continuity = compute_stats(vector_diffs_deg)

    # Pass/Fail conditions
    pass_a = (counters["test_a_ik_success"] == N_SAMPLES_AB) and (counters["test_a_ik_failures"] == 0)
    pass_b = (counters["test_b_ik_success"] == N_SAMPLES_AB) and (counters["test_b_ik_failures"] == 0)
    pass_c = (counters["test_c_ik_success"] == N_TRAJECTORY_C) and (counters["test_c_ik_failures"] == 0) and (large_jumps == 0)
    safety_pass = (counters["hard_limit_violations"] == 0) and (counters["safe_limit_violations"] == 0) and (counters["ik_nonfinite"] == 0)

    overall_pass = pass_a and pass_b and pass_c and safety_pass

    # Print Report
    print("\n" + "=" * 70)
    print("ACCURACY & RUNTIME METRICS SUMMARY")
    print("=" * 70)
    print("1. POSITION ERROR (mm):")
    print(f"{'Test':<8} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'A (Exact)':<8} {stats_pos_a.count:>6} {stats_pos_a.mean:>8.4f} {stats_pos_a.p50:>8.4f} {stats_pos_a.p95:>8.4f} {stats_pos_a.p99:>8.4f} {stats_pos_a.max_val:>8.4f}")
    print(f"{'B (Perturb)':<8} {stats_pos_b.count:>6} {stats_pos_b.mean:>8.4f} {stats_pos_b.p50:>8.4f} {stats_pos_b.p95:>8.4f} {stats_pos_b.p99:>8.4f} {stats_pos_b.max_val:>8.4f}")
    print(f"{'C (Traj)':<8} {stats_pos_c.count:>6} {stats_pos_c.mean:>8.4f} {stats_pos_c.p50:>8.4f} {stats_pos_c.p95:>8.4f} {stats_pos_c.p99:>8.4f} {stats_pos_c.max_val:>8.4f}")

    print("\n2. ORIENTATION ERROR (deg):")
    print(f"{'Test':<8} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'A (Exact)':<8} {stats_rot_a.count:>6} {stats_rot_a.mean:>8.4f} {stats_rot_a.p50:>8.4f} {stats_rot_a.p95:>8.4f} {stats_rot_a.p99:>8.4f} {stats_rot_a.max_val:>8.4f}")
    print(f"{'B (Perturb)':<8} {stats_rot_b.count:>6} {stats_rot_b.mean:>8.4f} {stats_rot_b.p50:>8.4f} {stats_rot_b.p95:>8.4f} {stats_rot_b.p99:>8.4f} {stats_rot_b.max_val:>8.4f}")
    print(f"{'C (Traj)':<8} {stats_rot_c.count:>6} {stats_rot_c.mean:>8.4f} {stats_rot_c.p50:>8.4f} {stats_rot_c.p95:>8.4f} {stats_rot_c.p99:>8.4f} {stats_rot_c.max_val:>8.4f}")

    print("\n3. IK RUNTIME (ms):")
    print(f"{'Test':<8} {'Count':>6} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"{'A (Exact)':<8} {stats_time_a.count:>6} {stats_time_a.mean:>8.4f} {stats_time_a.p50:>8.4f} {stats_time_a.p95:>8.4f} {stats_time_a.p99:>8.4f} {stats_time_a.max_val:>8.4f}")
    print(f"{'B (Perturb)':<8} {stats_time_b.count:>6} {stats_time_b.mean:>8.4f} {stats_time_b.p50:>8.4f} {stats_time_b.p95:>8.4f} {stats_time_b.p99:>8.4f} {stats_time_b.max_val:>8.4f}")
    print(f"{'C (Traj)':<8} {stats_time_c.count:>6} {stats_time_c.mean:>8.4f} {stats_time_c.p50:>8.4f} {stats_time_c.p95:>8.4f} {stats_time_c.p99:>8.4f} {stats_time_c.max_val:>8.4f}")

    print("\n4. JOINT CONTINUITY (Test C):")
    print(f"{'Joint':<16} {'Mean Δdeg':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
    for j, name in enumerate(joint_names):
        st = per_joint_continuity[j]
        print(f"{name:<16} {st.mean:>10.4f} {st.p95:>10.4f} {st.p99:>10.4f} {st.max_val:>10.4f}")
    print(f"{'VECTOR NORM':<16} {vec_continuity.mean:>10.4f} {vec_continuity.p95:>10.4f} {vec_continuity.p99:>10.4f} {vec_continuity.max_val:>10.4f}")

    print("\n5. FAILURE AUDIT:")
    for k, v in counters.items():
        print(f"  {k:<32}: {v}")

    print("=" * 70)
    print(f"VERDICT: IK_FK_ACCURACY = {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
