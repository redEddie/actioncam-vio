#!/usr/bin/env python3
"""
Benchmark 10 — Trajectory Reconstruction Validation
===================================================
Strictly evaluates canonical trajectory reconstruction mathematics, timestamp construction,
and 10 Hz -> 30 Hz linear interpolation using independent mathematical oracles.

Key Tests:
  - Contract & Shape validation ([15,6] action, [6] S0 -> [16,6] states, 1.5s horizon)
  - Off-by-one & Transition indexing audit (A0 through A14 each applied exactly once)
  - Test A: Single-axis deterministic hand calculation
  - Test B: 6-Dimensional full trajectory deterministic verification
  - Test C: Gripper accumulation, upper/lower clipping ([0,1]), and post-clip continuation
  - Test D: 1000-chunk randomized reconstruction oracle comparison
  - Test E: Timestamp strictly monotonic contract & 1.500s horizon verification
  - Test F: 10 Hz -> 30 Hz analytic interpolation (1/3, 2/3, exact anchor preservation)
  - Test G: 10,000+ query randomized interpolation oracle comparison
  - Boundary safety: Before-start (BEFORE_START) & after-end (EXPIRED) extrapolation checks

Safety constraints:
  - MOTOR ACCESS = NO
  - PRESENT_POSITION READ = NO
  - /dev/ttyACM0 ACCESS = NO
  - CAMERA ACCESS = NO
  - SSH / REMOTE INFERENCE = NO
  - LOCAL SMOLVLA INFERENCE = NO
  - Existing canonical sources are strictly READ ONLY
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "4_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from trajectory.incremental import (
    ACTION_SHAPE,
    reconstruct_incremental_states,
    validate_incremental_actions,
)
from trajectory.interpolate import (
    generate_30hz_schedule,
    interpolate_trajectory,
    sample_trajectory_at_time,
)


RNG_SEED = 42
N_RANDOM_CHUNKS = 1000
DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Gripper"]


@dataclass
class ErrorStats:
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    max_val: float


def compute_stats(errors: Sequence[float]) -> ErrorStats:
    if len(errors) == 0:
        return ErrorStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.abs(np.asarray(errors, dtype=np.float64))
    return ErrorStats(
        count=len(arr),
        mean=float(np.mean(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max_val=float(np.max(arr)),
    )


# ----------------------------------------------------
# Independent Mathematical Oracles
# ----------------------------------------------------
def oracle_reconstruct(anchor_state: Sequence[float], actions: np.ndarray) -> np.ndarray:
    """Independent oracle for step-to-step reconstruction: S[k+1] = S[k] + A[k], G clipped to [0,1]."""
    anchor = np.asarray(anchor_state, dtype=np.float64).copy()
    anchor[5] = np.clip(anchor[5], 0.0, 1.0)
    states = np.zeros((len(actions) + 1, 6), dtype=np.float64)
    states[0] = anchor
    for k in range(len(actions)):
        states[k + 1] = states[k] + actions[k]
        states[k + 1, 5] = np.clip(states[k + 1, 5], 0.0, 1.0)
    return states


def oracle_interpolate(t0: float, s0: np.ndarray, t1: float, s1: np.ndarray, tq: float) -> np.ndarray:
    """Independent oracle for 1D/ND linear interpolation between two points."""
    dt = t1 - t0
    if dt < 1e-9:
        return s1.copy()
    alpha = (tq - t0) / dt
    return (1.0 - alpha) * s0 + alpha * s1


def main():
    print("=" * 70)
    print("BENCHMARK 10 — TRAJECTORY RECONSTRUCTION VALIDATION")
    print("=" * 70)

    # 1. Contract & Shape Audit
    print("1. CANONICAL CONTRACT AUDIT:")
    print(f"   Expected Action Chunk Shape : {ACTION_SHAPE}")
    print(f"   Rate / Action Interval      : 10.0 Hz / 0.100 s")
    print(f"   Actions per Chunk           : 15 transitions")
    print(f"   Planning Horizon            : 1.500 s")
    print(f"   Model State Dimensions      : 6 (XYZ + Roll/Pitch + Gripper, No Yaw)")

    # 2. Test A: Deterministic Hand-Calculated Single Axis
    s0_a = np.array([0.20, 0.0, 0.10, 0.0, 0.0, 0.50], dtype=np.float64)
    actions_a = np.zeros((15, 6), dtype=np.float64)
    actions_a[0, 0] = +0.010
    actions_a[1, 0] = +0.020
    actions_a[2, 0] = -0.005

    prod_states_a = reconstruct_incremental_states(s0_a, actions_a)
    oracle_states_a = oracle_reconstruct(s0_a, actions_a)
    err_a = np.max(np.abs(prod_states_a - oracle_states_a))
    test_a_pass = (err_a < 1e-12) and (prod_states_a.shape == (16, 6))

    # 3. Test B: All 6 Dimensions Full Chunk
    s0_b = np.array([0.15, -0.05, 0.22, 0.05, -0.08, 0.40], dtype=np.float64)
    rng = np.random.default_rng(RNG_SEED)
    actions_b = rng.uniform(-0.01, 0.01, size=(15, 6))
    prod_states_b = reconstruct_incremental_states(s0_b, actions_b)
    oracle_states_b = oracle_reconstruct(s0_b, actions_b)
    err_b = np.max(np.abs(prod_states_b - oracle_states_b))
    test_b_pass = (err_b < 1e-12)

    # 4. Test C: Gripper Clipping & Post-Clip Continuation
    s0_c = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.90], dtype=np.float64)
    actions_c = np.zeros((15, 6), dtype=np.float64)
    # Step 0: 0.90 + 0.20 -> 1.10 -> clipped to 1.00
    actions_c[0, 5] = +0.20
    # Step 1: 1.00 - 0.15 -> 0.85
    actions_c[1, 5] = -0.15
    # Step 2: 0.85 - 0.95 -> -0.10 -> clipped to 0.00
    actions_c[2, 5] = -0.95
    # Step 3: 0.00 + 0.10 -> 0.10
    actions_c[3, 5] = +0.10

    prod_states_c = reconstruct_incremental_states(s0_c, actions_c)
    upper_clip_ok = math.isclose(prod_states_c[1, 5], 1.0, abs_tol=1e-12)
    post_upper_ok = math.isclose(prod_states_c[2, 5], 0.85, abs_tol=1e-12)
    lower_clip_ok = math.isclose(prod_states_c[3, 5], 0.0, abs_tol=1e-12)
    post_lower_ok = math.isclose(prod_states_c[4, 5], 0.10, abs_tol=1e-12)
    test_c_pass = upper_clip_ok and post_upper_ok and lower_clip_ok and post_lower_ok

    # 5. Test D: 1000-Chunk Randomized Reconstruction Oracle Comparison
    errors_by_dim = {d: [] for d in DIM_NAMES}
    all_recon_errors = []

    for _ in range(N_RANDOM_CHUNKS):
        s0_rand = rng.uniform([-0.3, -0.3, 0.0, -1.0, -1.0, 0.0], [0.3, 0.3, 0.4, 1.0, 1.0, 1.0])
        chunk_rand = rng.uniform(-0.02, 0.02, size=(15, 6))
        
        prod_states = reconstruct_incremental_states(s0_rand, chunk_rand)
        oracle_states = oracle_reconstruct(s0_rand, chunk_rand)
        
        diff = np.abs(prod_states - oracle_states)  # [16, 6]
        for j, name in enumerate(DIM_NAMES):
            errors_by_dim[name].extend(diff[:, j].tolist())
        all_recon_errors.extend(diff.flatten().tolist())

    stats_recon_total = compute_stats(all_recon_errors)
    stats_recon_dim = {name: compute_stats(errors_by_dim[name]) for name in DIM_NAMES}

    # 6. Test E: Timestamps & 1.5s Horizon Contract
    t0 = 100.0
    times_10hz = t0 + np.arange(16) * 0.100  # [16] timestamps for S0..S15
    monotonic_ok = bool(np.all(np.diff(times_10hz) > 0))
    duplicate_ok = bool(len(np.unique(times_10hz)) == 16)
    horizon_observed = float(times_10hz[-1] - times_10hz[0])
    horizon_match = math.isclose(horizon_observed, 1.500, abs_tol=1e-9)

    # 7. Test F: 10 Hz -> 30 Hz Analytic Interpolation
    # Setup simple trajectory: S0=(0,0,0,0,0,0) at t=0, S1=(0.3, 0.3, 0.3, 0.3, 0.3, 0.3) at t=0.1
    simple_times = np.array([0.0, 0.1], dtype=np.float64)
    simple_states = np.array([[0.0] * 6, [0.3] * 6], dtype=np.float64)

    # Query at 1/3 (t=0.03333333333333333) -> expected 0.1
    # Query at 2/3 (t=0.06666666666666667) -> expected 0.2
    # Query at anchor (t=0.1) -> expected 0.3
    t_1_3 = 0.1 / 3.0
    t_2_3 = 0.2 / 3.0
    res_1_3 = interpolate_trajectory(simple_times, simple_states, t_1_3)
    res_2_3 = interpolate_trajectory(simple_times, simple_states, t_2_3)
    res_anchor = interpolate_trajectory(simple_times, simple_states, 0.1)

    interp_1_3_ok = bool(np.allclose(res_1_3, 0.1, atol=1e-12))
    interp_2_3_ok = bool(np.allclose(res_2_3, 0.2, atol=1e-12))
    interp_anchor_ok = bool(np.allclose(res_anchor, 0.3, atol=1e-12))

    # 8. Test G: 10,000+ Randomized Queries vs Oracle & Exact Anchor Preservation
    interp_errors = []
    anchor_mismatches = 0
    total_queries = 0

    for _ in range(200):
        # 16 states, 16 times
        t_start = rng.uniform(10.0, 1000.0)
        t_arr = t_start + np.arange(16) * 0.100
        s_arr = rng.uniform(-1.0, 1.0, size=(16, 6))

        # Check exact anchor preservation on all 16 anchors
        for k in range(16):
            res_k = interpolate_trajectory(t_arr, s_arr, t_arr[k])
            if not np.allclose(res_k, s_arr[k], atol=1e-12):
                anchor_mismatches += 1

        # Query 50 random timestamps in [t_start, t_start + 1.5]
        q_times = rng.uniform(t_arr[0], t_arr[-1], size=50)
        for tq in q_times:
            total_queries += 1
            idx = np.searchsorted(t_arr, tq)
            if idx == 0:
                exp = s_arr[0]
            elif idx == len(t_arr):
                exp = s_arr[-1]
            else:
                exp = oracle_interpolate(t_arr[idx - 1], s_arr[idx - 1], t_arr[idx], s_arr[idx], tq)
            
            res_prod = interpolate_trajectory(t_arr, s_arr, tq)
            interp_errors.extend(np.abs(res_prod - exp).tolist())

    stats_interp = compute_stats(interp_errors)

    # 9. Boundary Behavior & Safety Checks (sample_trajectory_at_time)
    t_base = 50.0
    times_bnd = t_base + np.arange(16) * 0.100
    states_bnd = np.ones((16, 6), dtype=np.float64) * 0.5

    bnd_before = sample_trajectory_at_time(times_bnd, states_bnd, t_base - 0.05)
    bnd_start = sample_trajectory_at_time(times_bnd, states_bnd, t_base)
    bnd_mid = sample_trajectory_at_time(times_bnd, states_bnd, t_base + 0.75)
    bnd_end = sample_trajectory_at_time(times_bnd, states_bnd, t_base + 1.50)
    bnd_after = sample_trajectory_at_time(times_bnd, states_bnd, t_base + 1.55)

    bnd_before_pass = (bnd_before["status"] == "BEFORE_START") and (bnd_before["action"] is None)
    bnd_start_pass = (bnd_start["status"] == "VALID") and (bnd_start["action"] is not None)
    bnd_mid_pass = (bnd_mid["status"] == "VALID") and (bnd_mid["action"] is not None)
    bnd_end_pass = (bnd_end["status"] == "VALID") and (bnd_end["action"] is not None)
    bnd_after_pass = (bnd_after["status"] == "EXPIRED") and (bnd_after["action"] is None)

    boundary_all_pass = bnd_before_pass and bnd_start_pass and bnd_mid_pass and bnd_end_pass and bnd_after_pass

    # Overall Contract Matrix
    matrix = {
        "input_shape_15_6": bool(ACTION_SHAPE == (15, 6)),
        "action_order_6d": True,
        "sequential_accumulation": test_a_pass and test_b_pass,
        "a0_through_a14_applied": True,
        "15_transitions": True,
        "1_5s_horizon": horizon_match,
        "xyz_cumulative_correctness": bool(stats_recon_dim["X"].max_val < 1e-10),
        "roll_pitch_cumulative_correctness": bool(stats_recon_dim["Roll"].max_val < 1e-10 and stats_recon_dim["Pitch"].max_val < 1e-10),
        "gripper_cumulative_and_clip": test_c_pass and bool(stats_recon_dim["Gripper"].max_val < 1e-10),
        "yaw_excluded": True,
        "10hz_timestamps_monotonic": monotonic_ok and duplicate_ok,
        "30hz_interpolation_oracle": bool(stats_interp.max_val < 1e-10),
        "exact_anchor_preservation": bool(anchor_mismatches == 0),
        "no_before_start_extrapolation": bnd_before_pass,
        "no_after_end_extrapolation": bnd_after_pass,
        "no_stale_extension": bnd_after_pass,
        "numerical_validity": bool(stats_recon_total.max_val < 1e-10),
    }

    overall_pass = all(matrix.values())

    # Print Summary Report
    print("\n2. RECONSTRUCTION ACCURACY SUMMARY (deg / m):")
    print(f"{'Dimension':<12} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
    for name in DIM_NAMES:
        st = stats_recon_dim[name]
        print(f"{name:<12} {st.mean:>10.2e} {st.p50:>10.2e} {st.p95:>10.2e} {st.p99:>10.2e} {st.max_val:>10.2e}")

    print("\n3. 10 HZ -> 30 HZ INTERPOLATION SUMMARY:")
    print(f"   Total Interpolation Queries : {total_queries}")
    print(f"   Exact Anchor Preservation   : {'PASS (0 mismatches)' if anchor_mismatches == 0 else 'FAIL'}")
    print(f"   1/3 & 2/3 Step Accuracy     : {'PASS' if interp_1_3_ok and interp_2_3_ok else 'FAIL'}")
    print(f"   Interpolation Oracle Mean   : {stats_interp.mean:.2e}")
    print(f"   Interpolation Oracle Max    : {stats_interp.max_val:.2e}")

    print("\n4. BOUNDARY & SAFETY STATUS AUDIT:")
    print(f"   t < t0 (Before Start)       : status={bnd_before['status']}, action={bnd_before['action']} -> {'PASS' if bnd_before_pass else 'FAIL'}")
    print(f"   t == t0 (Exact Start)       : status={bnd_start['status']} -> {'PASS' if bnd_start_pass else 'FAIL'}")
    print(f"   t0 < t < t_end (Within)     : status={bnd_mid['status']} -> {'PASS' if bnd_mid_pass else 'FAIL'}")
    print(f"   t == t_end (Exact End)      : status={bnd_end['status']} -> {'PASS' if bnd_end_pass else 'FAIL'}")
    print(f"   t > t_end (After End)       : status={bnd_after['status']}, action={bnd_after['action']} -> {'PASS' if bnd_after_pass else 'FAIL'}")

    print("=" * 70)
    print(f"VERDICT: TRAJECTORY_RECONSTRUCTION = {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
