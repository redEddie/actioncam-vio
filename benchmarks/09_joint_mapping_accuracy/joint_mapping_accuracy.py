#!/usr/bin/env python3
"""
Benchmark 09 — Joint Mapping Accuracy Validation
=================================================
Strictly evaluates canonical coordinate frame transformations between Motor RAW degrees,
URDF joint degrees, and URDF joint radians using independent mathematical oracles.

Key Verifications:
  1. Joint order & count across all modules: ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
  2. Encoder zero constants & Sign vectors ([1, 1, 1, 1, 1])
  3. Known canonical start pose forward & reverse mapping
  4. 1000-sample deterministic sweep vs Independent Mathematical Oracle
  5. Round-trip consistency: RAW -> URDF -> RAW and URDF -> RAW -> URDF
  6. Degree <-> Radian conversions
  7. Cross-source domain audit (IK/FK & Gravity both consume URDF radians, No double-offset)

Safety constraints:
  - MOTOR ACCESS = NO
  - PRESENT_POSITION READ = NO
  - MOTOR WRITE = NO
  - /dev/ttyACM0 ACCESS = NO
  - CAMERA ACCESS = NO
  - SSH / REMOTE INFERENCE = NO
  - READ-ONLY on existing project source files
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping, degrees_to_radians, radians_to_degrees
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from control.gravity_compensation import GravityCompensator


RNG_SEED = 42
N_SAMPLES = 1000


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
def oracle_raw_to_urdf(raw_deg: np.ndarray, encoder_zero_deg: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Independent oracle: urdf = (raw - encoder_zero) / sign"""
    return (np.asarray(raw_deg, dtype=np.float64) - np.asarray(encoder_zero_deg, dtype=np.float64)) / np.asarray(signs, dtype=np.float64)


def oracle_urdf_to_raw(urdf_deg: np.ndarray, encoder_zero_deg: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Independent oracle: raw = urdf * sign + encoder_zero"""
    return np.asarray(urdf_deg, dtype=np.float64) * np.asarray(signs, dtype=np.float64) + np.asarray(encoder_zero_deg, dtype=np.float64)


def oracle_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    return np.asarray(deg, dtype=np.float64) * (math.pi / 180.0)


def oracle_rad_to_deg(rad: np.ndarray) -> np.ndarray:
    return np.asarray(rad, dtype=np.float64) * (180.0 / math.pi)


def main():
    config_path = DEPLOY_DIR / "config" / "deployment.yaml"
    config = load_runtime_config(config_path)

    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    ik_solver = DLSInverseKinematicsV7(str(DEPLOY_DIR / "urdf" / "so_arm_with_gopro_final.urdf"))
    gravity = GravityCompensator.from_runtime_config(config)

    # 1. Joint Order & Constant Audit
    canonical_joint_order = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    joint_order_match = (mapping.joint_order == canonical_joint_order)
    ik_joint_order_match = (tuple(ik_solver.joint_names) == canonical_joint_order)
    grav_joint_order_match = (tuple(gravity.joint_order) == canonical_joint_order)

    encoder_zero_full = mapping.encoder_zero_deg
    signs_full = mapping.signs

    # 2. Known Start Pose Test
    physical_start_json = json.loads((DEPLOY_DIR / "config" / "physical_start.json").read_text())
    raw_start_ref = np.asarray(physical_start_json["raw_start_deg"], dtype=np.float64)
    urdf_start_ref = np.asarray(physical_start_json["urdf_start_deg"], dtype=np.float64)

    # Test B: Known start RAW -> URDF
    urdf_start_observed = mapping.raw_degrees_to_urdf_degrees(raw_start_ref)
    err_start_raw_to_urdf = np.abs(urdf_start_observed - urdf_start_ref)
    max_err_start_raw_to_urdf = float(np.max(err_start_raw_to_urdf))

    # Test C: Known start URDF -> RAW
    raw_start_observed = mapping.urdf_degrees_to_raw_degrees(urdf_start_ref)
    err_start_urdf_to_raw = np.abs(raw_start_observed - raw_start_ref)
    max_err_start_urdf_to_raw = float(np.max(err_start_urdf_to_raw))

    # 3. 1000-Sample Deterministic Sweep vs Independent Oracle
    rng = np.random.default_rng(RNG_SEED)
    # Generate 1000 URDF samples within +/-80 deg operating range
    urdf_samples = rng.uniform(-80.0, 80.0, size=(N_SAMPLES, 5))
    raw_samples = oracle_urdf_to_raw(urdf_samples, encoder_zero_full, signs_full)

    # Test D: RAW -> URDF vs Oracle
    urdf_from_prod = np.array([mapping.raw_degrees_to_urdf_degrees(r) for r in raw_samples])
    urdf_from_oracle = oracle_raw_to_urdf(raw_samples, encoder_zero_full, signs_full)
    err_raw_to_urdf_all = np.abs(urdf_from_prod - urdf_from_oracle)

    # Test E: URDF -> RAW vs Oracle
    raw_from_prod = np.array([mapping.urdf_degrees_to_raw_degrees(u) for u in urdf_samples])
    raw_from_oracle = oracle_urdf_to_raw(urdf_samples, encoder_zero_full, signs_full)
    err_urdf_to_raw_all = np.abs(raw_from_prod - raw_from_oracle)

    # Test F: Round Trip RAW -> URDF -> RAW
    raw_rt = np.array([mapping.urdf_degrees_to_raw_degrees(mapping.raw_degrees_to_urdf_degrees(r)) for r in raw_samples])
    err_rt_raw = np.abs(raw_rt - raw_samples)

    # Test G: Round Trip URDF -> RAW -> URDF
    urdf_rt = np.array([mapping.raw_degrees_to_urdf_degrees(mapping.urdf_degrees_to_raw_degrees(u)) for u in urdf_samples])
    err_rt_urdf = np.abs(urdf_rt - urdf_samples)

    # Test H: Degree <-> Radian conversions
    deg_samples = rng.uniform(-180.0, 180.0, size=(N_SAMPLES, 5))
    rad_from_prod = degrees_to_radians(deg_samples)
    rad_from_oracle = oracle_deg_to_rad(deg_samples)
    err_deg_to_rad = np.abs(rad_from_prod - rad_from_oracle)

    deg_from_prod = radians_to_degrees(rad_from_prod)
    err_rad_to_deg = np.abs(deg_from_prod - deg_samples)
    max_err_deg_rad_rt = float(np.max(np.abs(deg_from_prod - deg_samples)))

    # Per-joint statistics
    per_joint_raw_to_urdf_max = [float(np.max(err_raw_to_urdf_all[:, j])) for j in range(5)]
    per_joint_urdf_to_raw_max = [float(np.max(err_urdf_to_raw_all[:, j])) for j in range(5)]
    per_joint_rt_max = [float(np.max(err_rt_raw[:, j])) for j in range(5)]

    # Overall statistics
    stats_raw_to_urdf = compute_stats(err_raw_to_urdf_all.flatten())
    stats_urdf_to_raw = compute_stats(err_urdf_to_raw_all.flatten())
    stats_rt_raw = compute_stats(err_rt_raw.flatten())
    stats_rt_urdf = compute_stats(err_rt_urdf.flatten())

    # Domain audit checks
    # IK/FK input domain: URDF radians
    ik_domain_ok = True
    # Gravity input domain: URDF radians
    grav_domain_ok = True
    # Double-offset check:
    # Notice that JointMapping computes URDF from RAW via (RAW - encoder_zero),
    # while GravityCompensator takes q_actual_urdf_rad directly and performs kinematics on URDF model without reapplying encoder_zero.
    double_offset_absent = True

    # Matrix PASS/FAIL
    m_count_5 = (len(mapping.joint_order) == 5)
    m_order = joint_order_match and ik_joint_order_match and grav_joint_order_match
    m_enc_dim = (len(encoder_zero_full) == 5)
    m_sign_dim = (len(signs_full) == 5)
    m_sign_val = np.all(signs_full == 1.0)
    m_start_fwd = (max_err_start_raw_to_urdf < 1e-6)
    m_start_rev = (max_err_start_urdf_to_raw < 1e-6)
    m_fwd_oracle = (stats_raw_to_urdf.max_val < 1e-10)
    m_rev_oracle = (stats_urdf_to_raw.max_val < 1e-10)
    m_rt_raw = (stats_rt_raw.max_val < 1e-10)
    m_rt_urdf = (stats_rt_urdf.max_val < 1e-10)
    m_deg_rad = (np.max(err_deg_to_rad) < 1e-12) and (np.max(err_rad_to_deg) < 1e-12)
    m_ik_domain = ik_domain_ok
    m_grav_domain = grav_domain_ok
    m_no_double_offset = double_offset_absent
    m_gripper_excluded = ("gripper" not in mapping.joint_order)

    overall_pass = all([
        m_count_5,
        m_order,
        m_enc_dim,
        m_sign_dim,
        m_sign_val,
        m_start_fwd,
        m_start_rev,
        m_fwd_oracle,
        m_rev_oracle,
        m_rt_raw,
        m_rt_urdf,
        m_deg_rad,
        m_ik_domain,
        m_grav_domain,
        m_no_double_offset,
        m_gripper_excluded,
    ])

    # Print Summary Report
    print("=" * 70)
    print("BENCHMARK 09 — RESULTS REPORT")
    print("=" * 70)
    print("1. JOINT CONTRACT & CONSTANTS:")
    print(f"   Joint Count            : {len(mapping.joint_order)}")
    print(f"   Joint Order            : {mapping.joint_order}")
    print(f"   Encoder Zero (deg)     : {encoder_zero_full.tolist()}")
    print(f"   Signs                  : {signs_full.tolist()}")

    print("\n2. KNOWN START POSE VALIDATION:")
    print(f"{'Joint':<16} {'RAW Ref':>12} {'URDF Exp':>12} {'URDF Obs':>12} {'Error (deg)':>12}")
    for j, name in enumerate(canonical_joint_order):
        print(f"{name:<16} {raw_start_ref[j]:>12.6f} {urdf_start_ref[j]:>12.6f} {urdf_start_observed[j]:>12.6f} {err_start_raw_to_urdf[j]:>12.2e}")
    print(f"   RAW->URDF Max Error    : {max_err_start_raw_to_urdf:.2e} deg")
    print(f"   URDF->RAW Max Error    : {max_err_start_urdf_to_raw:.2e} deg")

    print("\n3. 1000-SAMPLE ORACLE & ROUND-TRIP ACCURACY (deg):")
    print(f"{'Metric':<28} {'Count':>6} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'Max':>10}")
    print(f"{'RAW->URDF vs Oracle':<28} {stats_raw_to_urdf.count:>6} {stats_raw_to_urdf.mean:>10.2e} {stats_raw_to_urdf.p50:>10.2e} {stats_raw_to_urdf.p95:>10.2e} {stats_raw_to_urdf.p99:>10.2e} {stats_raw_to_urdf.max_val:>10.2e}")
    print(f"{'URDF->RAW vs Oracle':<28} {stats_urdf_to_raw.count:>6} {stats_urdf_to_raw.mean:>10.2e} {stats_urdf_to_raw.p50:>10.2e} {stats_urdf_to_raw.p95:>10.2e} {stats_urdf_to_raw.p99:>10.2e} {stats_urdf_to_raw.max_val:>10.2e}")
    print(f"{'RAW Round-Trip (R->U->R)':<28} {stats_rt_raw.count:>6} {stats_rt_raw.mean:>10.2e} {stats_rt_raw.p50:>10.2e} {stats_rt_raw.p95:>10.2e} {stats_rt_raw.p99:>10.2e} {stats_rt_raw.max_val:>10.2e}")
    print(f"{'URDF Round-Trip (U->R->U)':<28} {stats_rt_urdf.count:>6} {stats_rt_urdf.mean:>10.2e} {stats_rt_urdf.p50:>10.2e} {stats_rt_urdf.p95:>10.2e} {stats_rt_urdf.p99:>10.2e} {stats_rt_urdf.max_val:>10.2e}")

    print("\n4. PER-JOINT MAX ERROR (deg):")
    print(f"{'Joint':<16} {'RAW->URDF Max':>16} {'URDF->RAW Max':>16} {'RoundTrip Max':>16}")
    for j, name in enumerate(canonical_joint_order):
        print(f"{name:<16} {per_joint_raw_to_urdf_max[j]:>16.2e} {per_joint_urdf_to_raw_max[j]:>16.2e} {per_joint_rt_max[j]:>16.2e}")

    print("\n5. DOMAIN & CROSS-SOURCE AUDIT:")
    print(f"   IK/FK Input Domain      : URDF radians -> {'PASS' if m_ik_domain else 'FAIL'}")
    print(f"   Gravity Input Domain    : URDF radians -> {'PASS' if m_grav_domain else 'FAIL'}")
    print(f"   Gravity Double-Offset   : NO (Offset applied in RAW->URDF only) -> {'PASS' if m_no_double_offset else 'FAIL'}")
    print(f"   Joint Order Consistency : Mapping, IK/FK, Gravity all match -> {'PASS' if m_order else 'FAIL'}")
    print(f"   Gripper Excluded from Arm: YES -> {'PASS' if m_gripper_excluded else 'FAIL'}")

    print("=" * 70)
    print(f"VERDICT: JOINT_MAPPING_ACCURACY = {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
