#!/usr/bin/env python3
"""STEP8.26: one-session +5 mm Cartesian K_ext sweep (0.5, 0.8, 1.0).

This is deliberately a narrow continuation of STEP8.21: fixed start procedure,
IK, yaw-free target, +5 mm trajectory, frozen per-run support bias, timing and
return.  K_ext is the only controller parameter that changes.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
INFERENCE_START_ENCODER_DEG = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
K_EXT_VALUES = (0.5, 0.8, 1.0)
CORRECTION_CLAMP_DEG = 1.0
DELTA_Z_METERS = 0.005
START_FK_CONFOUND_MM = 3.0
START_JOINT_CONF0UND_DEG = 1.0
EXECUTION_TOKEN = "STEP8_26_FAST_CARTESIAN_GAIN_SWEEP"


@dataclass(frozen=True)
class Metrics:
    steady_delta_z_mm: float
    tracking_ratio_percent: float
    peak_delta_z_mm: float
    cartesian_norm_p95_mm: float
    return_norm_mm: float
    max_abs_q_error_deg: np.ndarray
    max_abs_q_corr_deg: np.ndarray
    clamp_fraction_per_joint: np.ndarray
    hunting_or_oscillation: bool


def external_correction(q_nom_rad: np.ndarray, q_actual_rad: np.ndarray, k_ext: float) -> tuple[np.ndarray, np.ndarray]:
    """q_error=q_nom-q_actual; q_corr=clip(K_ext*q_error, +/-1 degree)."""
    q_error_rad = q_nom_rad - q_actual_rad
    q_corr_deg = np.clip(np.rad2deg(q_error_rad) * k_ext, -CORRECTION_CLAMP_DEG, CORRECTION_CLAMP_DEG)
    return q_error_rad, np.deg2rad(q_corr_deg)


def calculate_metrics(tick_logs: Mapping[str, np.ndarray], p0: np.ndarray) -> Metrics:
    t = tick_logs["t_since_motion_start"]
    p_actual = np.column_stack((tick_logs["p_actual_x"], tick_logs["p_actual_y"], tick_logs["p_actual_z"]))
    p_target = np.column_stack((tick_logs["p_target_x"], tick_logs["p_target_y"], tick_logs["p_target_z"]))
    delta_mm = (p_actual - p0) * 1000.0
    hold = (t >= 3.50) & (t <= 4.00)  # fixed STEP8.21 late +5 mm hold
    returned = (t >= 9.50) & (t <= 10.00)  # fixed STEP8.21 final return settle
    if not np.any(hold) or not np.any(returned):
        raise RuntimeError("incomplete fixed trajectory log")
    steady_z = float(np.mean(delta_mm[hold, 2]))
    cartesian_norm = np.linalg.norm((p_target - p_actual) * 1000.0, axis=1)
    q_error_deg = tick_logs["joint_tracking_error"]
    q_corr_deg = tick_logs["q_corr_applied_deg"]
    clamp = np.isclose(np.abs(q_corr_deg), CORRECTION_CLAMP_DEG, atol=1e-9)
    # A sustained sign flip is four or more non-zero crossings in the final hold.
    hold_errors = q_error_deg[hold]
    sign_flips = 0
    for joint in range(hold_errors.shape[1]):
        signs = np.sign(hold_errors[:, joint]); signs = signs[signs != 0]
        sign_flips = max(sign_flips, int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0)
    return Metrics(
        steady_delta_z_mm=steady_z,
        tracking_ratio_percent=steady_z / 5.0 * 100.0,
        peak_delta_z_mm=float(np.max(delta_mm[:, 2])),
        cartesian_norm_p95_mm=float(np.percentile(cartesian_norm[hold], 95)),
        return_norm_mm=float(np.linalg.norm(np.mean(delta_mm[returned], axis=0))),
        max_abs_q_error_deg=np.max(np.abs(q_error_deg), axis=0),
        max_abs_q_corr_deg=np.max(np.abs(q_corr_deg), axis=0),
        clamp_fraction_per_joint=np.mean(clamp[hold], axis=0),
        hunting_or_oscillation=sign_flips >= 4,
    )


def select_next_step(metrics: Metrics) -> str:
    if metrics.tracking_ratio_percent >= 85.0:
        return "Increase correction clamp from +/-1.0 to +/-1.5 degrees once, only if meaningful steady error remains."
    if metrics.tracking_ratio_percent >= 75.0:
        return "Keep the best K_ext and test the +/-1.5 degree correction clamp once."
    return "Use load/small-command compensation or a servo-level support strategy; stop K_ext tuning."


def is_hard_stop(metrics: Metrics) -> bool:
    # Continuous clamp through most of the steady hold is a stop condition.
    return metrics.hunting_or_oscillation or bool(np.any(metrics.clamp_fraction_per_joint >= 0.70))


def make_hard_stop_observer() -> Callable[[float, np.ndarray, np.ndarray], bool]:
    """Detect persistent hold clamp or repeated hold sign flips while motion is live."""
    hold_ticks = 0
    clamp_ticks = np.zeros(5, dtype=int)
    sign_flips = np.zeros(5, dtype=int)
    previous_sign = np.zeros(5, dtype=int)

    def observe(current_t: float, q_error_deg: np.ndarray, q_corr_deg: np.ndarray) -> bool:
        nonlocal hold_ticks
        if not 2.0 <= current_t <= 4.0:  # fixed +5 mm hold only, excluding the ramp
            return False
        hold_ticks += 1
        clamp_ticks[:] += np.isclose(np.abs(q_corr_deg), CORRECTION_CLAMP_DEG, atol=1e-9)
        signs = np.sign(q_error_deg).astype(int)
        switched = (signs != 0) & (previous_sign != 0) & (signs != previous_sign)
        sign_flips[:] += switched
        previous_sign[:] = np.where(signs != 0, signs, previous_sign)
        return bool(np.any(sign_flips >= 4) or (hold_ticks >= 15 and np.any(clamp_ticks / hold_ticks >= 0.70)))

    return observe


def _runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    project_root = Path(__file__).resolve().parents[1]
    deploy_dir = project_root / "4_deploy"
    lerobot_src = deploy_dir / "Teleop" / "lerobot" / "src"
    for path in (str(lerobot_src), str(deploy_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import deploy_smolvla_yawfree as deploy
    from ik_solver_v7 import DLSInverseKinematicsV7
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SO100Follower
    return deploy, DLSInverseKinematicsV7, SOFollowerRobotConfig, SO100Follower


def run_session(*, port: str, log_dir: Path) -> list[tuple[float, Mapping[str, Any], Metrics | None]]:
    """Hardware-only path. Uses STEP8.21's fixed start, IK, yaw-free, timing and return."""
    deploy, IK, Config, Follower = _runtime_dependencies()
    import debug_cartesian_step8_21 as lineage
    ik_solver = IK(urdf_path=str(Path(__file__).resolve().parent / "URDF" / "so_arm_with_gopro_final.urdf"))
    follower = Follower(Config(port=port, use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    results: list[tuple[float, Mapping[str, Any], Metrics | None]] = []
    try:
        follower.bus.connect(handshake=True)
        # Fixed P64/I0/D32 is verified, never changed in this sweep.
        for register, expected in (("P_Coefficient", 64), ("I_Coefficient", 0), ("D_Coefficient", 32)):
            values = follower.bus.sync_read(register)
            if any(int(values[name]) != expected for name in ARM_NAMES):
                raise RuntimeError(f"fixed condition violated: {register} is not {expected}")
        previous_start: Mapping[str, Any] | None = None
        frozen_b_q_rad: np.ndarray | None = None
        frozen_b_q_deg: np.ndarray | None = None
        for k_ext in K_EXT_VALUES:
            present_deg = follower.bus.sync_read("Present_Position")
            start_q = np.array([float(present_deg[n]) for n in ARM_NAMES])
            target_q = np.array(INFERENCE_START_ENCODER_DEG)
            steps = 50
            for i in range(1, steps + 1):
                interp_q = start_q + (target_q - start_q) * (i / steps)
                follower.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(ARM_NAMES, interp_q)})
                time.sleep(0.02)
            time.sleep(1.0)
            actual_q = np.array([float(follower.bus.sync_read("Present_Position")[n]) for n in ARM_NAMES])
            q_err = np.max(np.abs(actual_q - target_q))
            print(f"RESET target: {np.round(target_q, 3)}")
            print(f"RESET actual: {np.round(actual_q, 3)}")
            print(f"RESET error: {q_err:.3f} deg")

            start, status = lineage.acquire_start_state(follower, list(ARM_NAMES), INFERENCE_START_ENCODER_DEG, ik_solver)
            if status != "PASS":
                results.append((k_ext, {"status": "FAIL_START"}, None)); break
            if frozen_b_q_rad is None:
                frozen_b_q_rad = np.array(start["b_q_rad"], copy=True)
                frozen_b_q_deg = np.array(start["b_q_deg"], copy=True)
            # b_q is fixed for every gain condition; only K_ext below varies.
            start["b_q_rad"] = frozen_b_q_rad
            start["b_q_deg"] = frozen_b_q_deg
            start["q_hold_goal"] = start["q_start_actual"] + frozen_b_q_deg
            if previous_start is not None:
                fk_delta = float(np.linalg.norm(start["p0"] - previous_start["p0"]) * 1000.0)
                q_delta = float(np.max(np.abs(start["q_start_actual_raw"] - previous_start["q_start_actual_raw"])))
                if fk_delta > START_FK_CONFOUND_MM or q_delta > START_JOINT_CONF0UND_DEG:
                    print(f"CONFOUND K={k_ext:.1f}: start FK={fk_delta:.3f} mm, q={q_delta:.3f} deg; retaining raw result")
            result = lineage.run_trial(
                follower, ik_solver, list(ARM_NAMES), k_ext, start, log_dir, f"step8_26_kext_{k_ext:.1f}.npz",
                safety_observer=make_hard_stop_observer(),
            )
            metrics = calculate_metrics(result["tick_logs"], start["p0"]) if result["status"] == "PASS" else None
            results.append((k_ext, result, metrics))
            previous_start = start
            if metrics is None or is_hard_stop(metrics):
                print(f"HARD STOP at K_ext={k_ext:.1f}")
                break
        return results
    finally:
        # Preserve the existing physical state; do not torque-cycle or tune anything.
        try:
            present = follower.bus.sync_read("Present_Position")
            follower.bus.sync_write("Goal_Position", {name: float(present[name]) for name in ARM_NAMES})
        finally:
            follower.disconnect()


def print_report(results: list[tuple[float, Mapping[str, Any], Metrics | None]]) -> None:
    print("K_ext | steady_dZ_mm | ratio_pct | peak_dZ_mm | cart_P95_mm | return_norm_mm | hunting")
    valid = []
    for k_ext, result, metrics in results:
        if metrics is None:
            print(f"{k_ext:.1f} | status={result['status']}")
            continue
        valid.append((k_ext, metrics))
        print(f"{k_ext:.1f} | {metrics.steady_delta_z_mm:.3f} | {metrics.tracking_ratio_percent:.1f} | {metrics.peak_delta_z_mm:.3f} | {metrics.cartesian_norm_p95_mm:.3f} | {metrics.return_norm_mm:.3f} | {'YES' if metrics.hunting_or_oscillation else 'NO'}")
        print(f"  max_abs_q_error_deg={metrics.max_abs_q_error_deg}; max_abs_q_corr_deg={metrics.max_abs_q_corr_deg}; clamp_fraction_per_joint={metrics.clamp_fraction_per_joint}")
    if valid:
        best_k, best = max(valid, key=lambda item: item[1].tracking_ratio_percent)
        print(f"SELECTED_BEST_K_EXT={best_k:.1f}")
        print(f"NEXT_STEP={select_next_step(best)}")
    else:
        print("SELECTED_BEST_K_EXT=none")
        print("NEXT_STEP=Use load/small-command compensation or a servo-level support strategy; stop K_ext tuning.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true", help="required: permit the physical three-condition session")
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    if not args.execute or args.token != EXECUTION_TOKEN:
        print("DRY RUN ONLY: pass --execute and the exact --token to enable hardware.")
        return 0
    if input("Type EXECUTE_STEP8_26 to continue: ").strip() != "EXECUTE_STEP8_26":
        raise RuntimeError("local confirmation did not match")
    log_dir = Path(__file__).resolve().parent / "logs" / "step8_26_kext_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    print_report(run_session(port=args.port, log_dir=log_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
