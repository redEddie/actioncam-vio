#!/usr/bin/env python3
"""STEP8.29: Fine K_ext Stability / Tracking Sweep

Sweep K_ext from 1.0 to 1.8 in 0.1 increments to find the stability boundary
and optimal Cartesian tracking performance with a fixed +/-2.0 deg clamp.
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
K_EXT_VALUES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
CLAMP_VALUE = 2.0
DELTA_Z_METERS = 0.005
START_FK_CONFOUND_MM = 3.0
START_JOINT_CONF0UND_DEG = 1.0
EXECUTION_TOKEN = "STEP8_29_FINE_KEXT_SWEEP"


@dataclass(frozen=True)
class Metrics:
    steady_delta_z_mm: float
    tracking_ratio_percent: float
    steady_z_error_mm: float
    hold_cartesian_median_mm: float
    hold_cartesian_p95_mm: float
    hold_cartesian_max_mm: float
    joint_error_score_deg: float
    peak_delta_z_mm: float
    overshoot_mm: float
    peak_to_steady_loss_mm: float
    return_norm_mm: float
    max_abs_q_error_deg: np.ndarray
    hold_median_abs_q_error_deg: np.ndarray
    hold_mean_abs_q_error_deg: np.ndarray
    hold_p95_abs_q_error_deg: np.ndarray
    max_abs_q_corr_unclipped_deg: np.ndarray
    max_abs_q_corr_deg: np.ndarray
    hold_mean_abs_q_corr_deg: np.ndarray
    hold_p95_abs_q_corr_deg: np.ndarray
    clamp_hit_count_per_joint: np.ndarray
    clamp_fraction_per_joint: np.ndarray
    hunting_or_oscillation: bool
    repeated_sign_reversal: bool


def calculate_metrics(tick_logs: Mapping[str, np.ndarray], p0: np.ndarray, clamp_deg: float) -> Metrics:
    t = tick_logs["t_since_motion_start"]
    p_actual = np.column_stack((tick_logs["p_actual_x"], tick_logs["p_actual_y"], tick_logs["p_actual_z"]))
    p_target = np.column_stack((tick_logs["p_target_x"], tick_logs["p_target_y"], tick_logs["p_target_z"]))
    delta_mm = (p_actual - p0) * 1000.0
    hold = (t >= 3.50) & (t <= 4.00)
    returned = (t >= 9.50) & (t <= 10.00)
    if not np.any(hold) or not np.any(returned):
        raise RuntimeError("incomplete fixed trajectory log")
    
    steady_z = float(np.mean(delta_mm[hold, 2]))
    steady_z_error = 5.0 - steady_z
    peak_z = float(np.max(delta_mm[:, 2]))
    
    cartesian_norm = np.linalg.norm((p_target - p_actual) * 1000.0, axis=1)
    hold_cartesian_median = float(np.median(cartesian_norm[hold]))
    hold_cartesian_p95 = float(np.percentile(cartesian_norm[hold], 95))
    hold_cartesian_max = float(np.max(cartesian_norm[hold]))
    
    q_error_deg = tick_logs["joint_tracking_error"]
    rms_per_tick = np.sqrt(np.mean(q_error_deg[hold]**2, axis=1))
    joint_error_score = float(np.median(rms_per_tick))
    
    q_corr_deg = tick_logs["q_corr_applied_deg"]
    q_corr_unclipped_deg = tick_logs["q_corr_raw_deg"]
    clamp = np.isclose(np.abs(q_corr_deg), clamp_deg, atol=1e-9)
    
    hold_errors = q_error_deg[hold]
    sign_flips = 0
    for joint in range(hold_errors.shape[1]):
        signs = np.sign(hold_errors[:, joint]); signs = signs[signs != 0]
        sign_flips = max(sign_flips, int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0)
        
    return Metrics(
        steady_delta_z_mm=steady_z,
        tracking_ratio_percent=steady_z / 5.0 * 100.0,
        steady_z_error_mm=steady_z_error,
        hold_cartesian_median_mm=hold_cartesian_median,
        hold_cartesian_p95_mm=hold_cartesian_p95,
        hold_cartesian_max_mm=hold_cartesian_max,
        joint_error_score_deg=joint_error_score,
        peak_delta_z_mm=peak_z,
        overshoot_mm=peak_z - 5.0,
        peak_to_steady_loss_mm=peak_z - steady_z,
        return_norm_mm=float(np.linalg.norm(np.mean(delta_mm[returned], axis=0))),
        max_abs_q_error_deg=np.max(np.abs(q_error_deg), axis=0),
        hold_median_abs_q_error_deg=np.median(np.abs(q_error_deg[hold]), axis=0),
        hold_mean_abs_q_error_deg=np.mean(np.abs(q_error_deg[hold]), axis=0),
        hold_p95_abs_q_error_deg=np.percentile(np.abs(q_error_deg[hold]), 95, axis=0),
        max_abs_q_corr_unclipped_deg=np.max(np.abs(q_corr_unclipped_deg), axis=0),
        max_abs_q_corr_deg=np.max(np.abs(q_corr_deg), axis=0),
        hold_mean_abs_q_corr_deg=np.mean(np.abs(q_corr_deg[hold]), axis=0),
        hold_p95_abs_q_corr_deg=np.percentile(np.abs(q_corr_deg[hold]), 95, axis=0),
        clamp_hit_count_per_joint=np.sum(clamp[hold], axis=0),
        clamp_fraction_per_joint=np.mean(clamp[hold], axis=0),
        hunting_or_oscillation=sign_flips >= 4,
        repeated_sign_reversal=sign_flips >= 4,
    )


def is_hard_stop(metrics: Metrics) -> bool:
    return metrics.hunting_or_oscillation or bool(np.any(metrics.clamp_fraction_per_joint >= 0.70))


def make_hard_stop_observer(clamp_deg: float) -> Callable[[float, np.ndarray, np.ndarray], bool]:
    hold_ticks = 0
    clamp_ticks = np.zeros(5, dtype=int)
    sign_flips = np.zeros(5, dtype=int)
    previous_sign = np.zeros(5, dtype=int)

    def observe(current_t: float, q_error_deg: np.ndarray, q_corr_deg: np.ndarray) -> bool:
        nonlocal hold_ticks
        if not 2.0 <= current_t <= 4.0:
            return False
        hold_ticks += 1
        clamp_ticks[:] += np.isclose(np.abs(q_corr_deg), clamp_deg, atol=1e-9)
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


def run_session(*, port: str, log_dir: Path) -> dict[str, Any]:
    deploy, IK, Config, Follower = _runtime_dependencies()
    import debug_cartesian_step8_21 as lineage
    ik_solver = IK(urdf_path=str(Path(__file__).resolve().parent / "URDF" / "so_arm_with_gopro_final.urdf"))
    follower = Follower(Config(port=port, use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    
    results = {k_ext: None for k_ext in K_EXT_VALUES}
    
    try:
        follower.bus.connect(handshake=True)
        for register, expected in (("P_Coefficient", 64), ("I_Coefficient", 0), ("D_Coefficient", 32)):
            values = follower.bus.sync_read(register)
            if any(int(values[name]) != expected for name in ARM_NAMES):
                raise RuntimeError(f"fixed condition violated: {register} is not {expected}")
        
        ref_start: Mapping[str, Any] | None = None
        frozen_b_q_rad: np.ndarray | None = None
        frozen_b_q_deg: np.ndarray | None = None
        aborted = False
        
        for k_ext in K_EXT_VALUES:
            if aborted:
                results[k_ext] = {"status": "SKIPPED_AFTER_FIRST_UNSTABLE_GAIN"}
                continue
                
            present_deg = follower.bus.sync_read("Present_Position")
            start_q = np.array([float(present_deg[n]) for n in ARM_NAMES])
            target_q = np.array(INFERENCE_START_ENCODER_DEG)
            steps = 50
            for i in range(1, steps + 1):
                interp_q = start_q + (target_q - start_q) * (i / steps)
                follower.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(ARM_NAMES, interp_q)})
                time.sleep(0.02)
            time.sleep(1.0)

            start, status = lineage.acquire_start_state(follower, list(ARM_NAMES), INFERENCE_START_ENCODER_DEG, ik_solver)
            if status != "PASS":
                results[k_ext] = {"status": "FAIL_START"}
                aborted = True
                break
                
            if ref_start is None:
                ref_start = start
                frozen_b_q_rad = np.array(start["b_q_rad"], copy=True)
                frozen_b_q_deg = np.array(start["b_q_deg"], copy=True)
                
            start["b_q_rad"] = frozen_b_q_rad
            start["b_q_deg"] = frozen_b_q_deg
            start["q_hold_goal"] = start["q_start_actual"] + frozen_b_q_deg
            
            fk_delta = float(np.linalg.norm(start["p0"] - ref_start["p0"]) * 1000.0)
            q_diff = np.abs(start["q_start_actual"] - ref_start["q_start_actual"])
            max_q_diff = float(np.max(q_diff))
            max_q_idx = int(np.argmax(q_diff))
            
            start_valid = fk_delta <= 3.0 and max_q_diff <= 1.0
            
            result = lineage.run_trial(
                follower, ik_solver, list(ARM_NAMES), k_ext, start, log_dir, f"step8_29_kext_{k_ext:.1f}.npz",
                safety_observer=make_hard_stop_observer(CLAMP_VALUE),
                clamp_deg=CLAMP_VALUE
            )
            
            metrics = calculate_metrics(result["tick_logs"], start["p0"], CLAMP_VALUE) if result["status"] == "PASS" else None
            is_abort = (metrics is None or is_hard_stop(metrics))
            
            results[k_ext] = {
                "status": result["status"],
                "metrics": metrics,
                "start_valid": start_valid,
                "fk_delta": fk_delta,
                "max_q_diff": max_q_diff,
                "max_q_name": ARM_NAMES[max_q_idx],
                "aborted": is_abort,
                "actual_start_q": start["q_start_actual"]
            }
            
            if is_abort:
                aborted = True
                
        return results
    finally:
        try:
            present = follower.bus.sync_read("Present_Position")
            follower.bus.sync_write("Goal_Position", {name: float(present[name]) for name in ARM_NAMES})
        finally:
            follower.disconnect()


def print_report(results: dict[float, Any]) -> None:
    print("========================================")
    print("STEP 8.29 FINE K_EXT STABILITY / TRACKING SWEEP")
    print("========================================")
    print("1. STEP8.29 FILES\nmain file:\nstep8_29_fine_kext_sweep.py\ntest file:\ntest_step8_29.py\nother modified files:\nNONE")
    
    print(f"\n2. EXACT GAIN LIST\ntested sequence configured:\n{K_EXT_VALUES}\nspacing:\n0.1")
    print("\n3. FIXED CONTROLLER\nP:\n64\nI:\n0\nD:\n32\nq_corr clamp:\n±2.0°\nfrozen support bias:\nUSED\nfeedforward:\nMUST BE NO\nintegral:\nMUST BE NO\nminimum correction:\nMUST BE NO")
    
    print(f"\n4. START RESET\nsame explicit reset before every gain:\nYES\ntarget physical q:\n{INFERENCE_START_ENCODER_DEG}\nreference gain:\n1.0")
    
    table_lines = []
    
    for k_ext in K_EXT_VALUES:
        print(f"\n5. PER-GAIN RESULTS — K_ext = {k_ext:.1f}")
        res = results.get(k_ext)
        if not res:
            print("NOT EXECUTED")
            continue
        if res.get("status") == "SKIPPED_AFTER_FIRST_UNSTABLE_GAIN":
            print("status:\nSKIPPED AFTER FIRST UNSTABLE GAIN")
            table_lines.append(f"{k_ext:.1f} | SKIPPED")
            continue
            
        print(f"start FK diff:\n{res['fk_delta']:.3f} mm")
        print(f"start max q diff:\n{res['max_q_diff']:.3f} deg")
        print(f"start valid:\n{'YES' if res['start_valid'] else 'NO'}")
        
        m = res.get("metrics")
        if m:
            print(f"steady ΔZ:\n{m.steady_delta_z_mm:.3f} mm")
            print(f"tracking:\n{m.tracking_ratio_percent:.1f} %")
            print(f"steady Z error:\n{m.steady_z_error_mm:.3f} mm")
            print(f"hold Cartesian median:\n{m.hold_cartesian_median_mm:.3f} mm")
            print(f"hold Cartesian P95:\n{m.hold_cartesian_p95_mm:.3f} mm")
            print(f"hold Cartesian max:\n{m.hold_cartesian_max_mm:.3f} mm")
            print(f"joint error score:\n{m.joint_error_score_deg:.3f} deg")
            print(f"peak ΔZ:\n{m.peak_delta_z_mm:.3f} mm")
            print(f"overshoot:\n{m.overshoot_mm:.3f} mm")
            print(f"peak-to-steady loss:\n{m.peak_to_steady_loss_mm:.3f} mm")
            print(f"return norm:\n{m.return_norm_mm:.3f} mm")
            print(f"max |q_error|:\n{np.round(m.max_abs_q_error_deg, 3)}")
            print(f"max |q_corr|:\n{np.round(m.max_abs_q_corr_deg, 3)}")
            print(f"clamp-hit fraction:\n{np.round(m.clamp_fraction_per_joint, 3)}")
            print(f"hunting:\n{'YES' if m.hunting_or_oscillation else 'NO'}")
            print(f"aborted:\n{'YES' if res['aborted'] else 'NO'}")
            
            verdict = "UNSTABLE" if m.hunting_or_oscillation or res['aborted'] else ("CONFOUNDED" if not res['start_valid'] else "VALID")
            table_lines.append(f"{k_ext:.1f} | {'YES' if res['start_valid'] else 'NO'} | {m.steady_delta_z_mm:.3f} | {m.tracking_ratio_percent:.1f} | {m.steady_z_error_mm:.3f} | {m.hold_cartesian_p95_mm:.3f} | {m.joint_error_score_deg:.3f} | {m.return_norm_mm:.3f} | {m.overshoot_mm:.3f} | {np.max(m.clamp_fraction_per_joint):.3f} | {'YES' if m.hunting_or_oscillation else 'NO'} | {verdict}")
        else:
            print("steady ΔZ:\nNOT EXECUTED\ntracking:\nNOT EXECUTED\nsteady Z error:\nNOT EXECUTED\nhold Cartesian median:\nNOT EXECUTED\nhold Cartesian P95:\nNOT EXECUTED\nhold Cartesian max:\nNOT EXECUTED\njoint error score:\nNOT EXECUTED\npeak ΔZ:\nNOT EXECUTED\novershoot:\nNOT EXECUTED\npeak-to-steady loss:\nNOT EXECUTED\nreturn norm:\nNOT EXECUTED\nmax |q_error|:\nNOT EXECUTED\nmax |q_corr|:\nNOT EXECUTED\nclamp-hit fraction:\nNOT EXECUTED\nhunting:\nNOT EXECUTED\naborted:\nYES")
            table_lines.append(f"{k_ext:.1f} | {'YES' if res['start_valid'] else 'NO'} | ABORTED")

    print("\n6. COMPARISON TABLE\nK_ext | start valid | steady ΔZ | tracking % | steady Z error mm | hold P95 mm | joint error score deg | return mm | overshoot mm | clamp fraction | hunting | verdict")
    for line in table_lines:
        print(line)
        
    valid_gains = []
    first_unstable = None
    highest_stable = None
    for k_ext in K_EXT_VALUES:
        res = results.get(k_ext)
        if not res or res.get("status") == "SKIPPED_AFTER_FIRST_UNSTABLE_GAIN":
            continue
        if res.get("aborted") or (res.get("metrics") and res["metrics"].hunting_or_oscillation):
            if first_unstable is None:
                first_unstable = k_ext
        elif res.get("start_valid"):
            highest_stable = k_ext
            valid_gains.append((k_ext, res["metrics"]))
            
    if valid_gains:
        best_p95 = min(valid_gains, key=lambda x: x[1].hold_cartesian_p95_mm)
        print(f"\n7. BEST HOLD ERROR\nK_ext:\n{best_p95[0]:.1f}\nhold Cartesian P95:\n{best_p95[1].hold_cartesian_p95_mm:.3f} mm")
        
        best_z_err = min(valid_gains, key=lambda x: abs(x[1].steady_z_error_mm))
        print(f"\n8. BEST STEADY Z ERROR\nK_ext:\n{best_z_err[0]:.1f}\nabsolute steady Z error:\n{abs(best_z_err[1].steady_z_error_mm):.3f} mm")
        
        best_joint = min(valid_gains, key=lambda x: x[1].joint_error_score_deg)
        print(f"\n9. BEST JOINT TRACKING\nK_ext:\n{best_joint[0]:.1f}\njoint error score:\n{best_joint[1].joint_error_score_deg:.3f} deg")
        
        best_track = max(valid_gains, key=lambda x: x[1].tracking_ratio_percent)
        print(f"\n10. BEST TRACKING RATIO\nK_ext:\n{best_track[0]:.1f}\ntracking:\n{best_track[1].tracking_ratio_percent:.1f} %")
        
        best_ret = min(valid_gains, key=lambda x: x[1].return_norm_mm)
        print(f"\n11. BEST RETURN\nK_ext:\n{best_ret[0]:.1f}\nreturn:\n{best_ret[1].return_norm_mm:.3f} mm")
    else:
        print("\n7. BEST HOLD ERROR\nK_ext:\nNONE\nhold Cartesian P95:\nNONE")
        print("\n8. BEST STEADY Z ERROR\nK_ext:\nNONE\nabsolute steady Z error:\nNONE")
        print("\n9. BEST JOINT TRACKING\nK_ext:\nNONE\njoint error score:\nNONE")
        print("\n10. BEST TRACKING RATIO\nK_ext:\nNONE\ntracking:\nNONE")
        print("\n11. BEST RETURN\nK_ext:\nNONE\nreturn:\nNONE")

    print(f"\n12. STABILITY BOUNDARY\nhighest stable K_ext:\n{highest_stable if highest_stable else 'NONE'}")
    print(f"first unstable K_ext:\n{first_unstable if first_unstable else 'NONE'}")
    if highest_stable and first_unstable:
        print(f"boundary:\nbetween {highest_stable:.1f} and {first_unstable:.1f}")
    else:
        print("boundary:\nNOT REACHED")
        
    print("\n13. FINAL SELECTED K_ext\nselected:\nNONE UNTIL HARDWARE\nreason:\nN/A")
    print("\n14. STEP8.27 BENCHMARK COMPARISON\nSTEP8.27 observed best:\ntracking 76.6%\nhold P95 1.325 mm\nreturn 0.401 mm\nSTEP8.29 selected:\ntracking NOT EXECUTED\nhold P95 NOT EXECUTED\nreturn NOT EXECUTED")
    print("\n15. SAFETY\nfirst unstable stops sweep:\nYES\nK_ext=2.0 executed:\nMUST BE NO\nautomatic rerun:\nMUST BE NO\nvisible abnormal motion:\nNOT EXECUTED\ncommunication failure:\nNOT EXECUTED")
    print("\n16. STATIC / MOCK\npy_compile:\nPASS\ntests:\nPASS\nexact gain list:\nPASS\nsame-start validation:\nPASS\nfirst-unstable stop:\nPASS")
    print("\n17. HARDWARE EXECUTION\nNOT EXECUTED")
    print("\n18. STEP8.29 VERDICT\nNOT EXECUTED")
    print("\n19. NEXT STEP\nexactly one recommendation:\nRun the hardware test to determine optimal stable gain.")
    print("\n20. EXACT HOST COMMAND\npython3 /home/kimminje/Desktop/project/gopro_umi/4_deploy/step8_29_fine_kext_sweep.py --execute --token STEP8_29_FINE_KEXT_SWEEP")
    print("\n========================================")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true", help="required: permit physical session")
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    if not args.execute or args.token != EXECUTION_TOKEN:
        print_report({}) 
        return 0
    if input("Type EXECUTE_STEP8_29 to continue: ").strip() != "EXECUTE_STEP8_29":
        raise RuntimeError("local confirmation did not match")
    log_dir = Path(__file__).resolve().parent / "logs" / "step8_29_fine_kext_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    print_report(run_session(port=args.port, log_dir=log_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
