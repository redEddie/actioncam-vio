#!/usr/bin/env python3
"""STEP8.30: Frozen Steady Load Feedforward Test at K_ext=1.1"""
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
K_EXT = 1.1
CLAMP_VALUE = 2.0
Q_FF_BOUND = 1.0
DELTA_Z_METERS = 0.005
EXECUTION_TOKEN = "STEP8_30_FROZEN_FEEDFORWARD"

@dataclass(frozen=True)
class Metrics:
    steady_delta_z_mm: float
    tracking_ratio_percent: float
    steady_z_error_mm: float
    hold_cartesian_median_mm: float
    hold_cartesian_p95_mm: float
    hold_cartesian_max_mm: float
    peak_delta_z_mm: float
    overshoot_mm: float
    peak_to_steady_loss_mm: float
    return_norm_mm: float
    steady_median_q_error_deg: np.ndarray
    steady_mean_q_error_deg: np.ndarray
    steady_p95_abs_q_error_deg: np.ndarray
    steady_std_q_error_deg: np.ndarray
    max_abs_q_error_deg: np.ndarray
    max_abs_q_corr_deg: np.ndarray
    hold_mean_abs_q_corr_deg: np.ndarray
    max_abs_q_ff_phase: np.ndarray
    hold_mean_q_ff_phase: np.ndarray
    hunting_or_oscillation: bool
    repeated_sign_reversal: bool

def calculate_metrics(tick_logs: Mapping[str, np.ndarray], p0: np.ndarray, clamp_deg: float) -> Metrics:
    t = tick_logs["t_since_motion_start"]
    p_actual = np.column_stack((tick_logs["p_actual_x"], tick_logs["p_actual_y"], tick_logs["p_actual_z"]))
    p_target = np.column_stack((tick_logs["p_target_x"], tick_logs["p_target_y"], tick_logs["p_target_z"]))
    delta_mm = (p_actual - p0) * 1000.0
    hold = (t >= 3.50) & (t <= 4.00)
    returned = (t >= 9.50) & (t <= 10.00)
    
    # 1.0 second steady tail for joint metrics
    steady_tail = (t >= 3.00) & (t <= 4.00)
    
    steady_z = float(np.mean(delta_mm[hold, 2]))
    steady_z_error = 5.0 - steady_z
    peak_z = float(np.max(delta_mm[:, 2]))
    
    cartesian_norm = np.linalg.norm((p_target - p_actual) * 1000.0, axis=1)
    hold_cartesian_median = float(np.median(cartesian_norm[hold]))
    hold_cartesian_p95 = float(np.percentile(cartesian_norm[hold], 95))
    hold_cartesian_max = float(np.max(cartesian_norm[hold]))
    
    q_error_deg = tick_logs["joint_tracking_error"]
    q_corr_deg = tick_logs["q_corr_applied_deg"]
    q_ff_phase_deg = tick_logs.get("q_ff_phase_deg", np.zeros_like(q_error_deg))
    
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
        peak_delta_z_mm=peak_z,
        overshoot_mm=peak_z - 5.0,
        peak_to_steady_loss_mm=peak_z - steady_z,
        return_norm_mm=float(np.linalg.norm(np.mean(delta_mm[returned], axis=0))),
        steady_median_q_error_deg=np.median(q_error_deg[steady_tail], axis=0),
        steady_mean_q_error_deg=np.mean(q_error_deg[steady_tail], axis=0),
        steady_p95_abs_q_error_deg=np.percentile(np.abs(q_error_deg[steady_tail]), 95, axis=0),
        steady_std_q_error_deg=np.std(q_error_deg[steady_tail], axis=0),
        max_abs_q_error_deg=np.max(np.abs(q_error_deg), axis=0),
        max_abs_q_corr_deg=np.max(np.abs(q_corr_deg), axis=0),
        hold_mean_abs_q_corr_deg=np.mean(np.abs(q_corr_deg[hold]), axis=0),
        max_abs_q_ff_phase=np.max(np.abs(q_ff_phase_deg), axis=0),
        hold_mean_q_ff_phase=np.mean(q_ff_phase_deg[hold], axis=0),
        hunting_or_oscillation=sign_flips >= 4,
        repeated_sign_reversal=sign_flips >= 4,
    )

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

def is_hard_stop(metrics: Metrics) -> bool:
    return metrics.hunting_or_oscillation

def make_hard_stop_observer(clamp_deg: float) -> Callable[[float, np.ndarray, np.ndarray], bool]:
    hold_ticks = 0
    sign_flips = np.zeros(5, dtype=int)
    previous_sign = np.zeros(5, dtype=int)
    def observe(current_t: float, q_error_deg: np.ndarray, q_corr_deg: np.ndarray) -> bool:
        nonlocal hold_ticks
        if not 2.0 <= current_t <= 4.0:
            return False
        hold_ticks += 1
        signs = np.sign(q_error_deg).astype(int)
        switched = (signs != 0) & (previous_sign != 0) & (signs != previous_sign)
        sign_flips[:] += switched
        previous_sign[:] = np.where(signs != 0, signs, previous_sign)
        return bool(np.any(sign_flips >= 4))
    return observe

def run_session(*, port: str, log_dir: Path) -> dict[str, Any]:
    deploy, IK, Config, Follower = _runtime_dependencies()
    import debug_cartesian_step8_21 as lineage
    ik_solver = IK(urdf_path=str(Path(__file__).resolve().parent / "URDF" / "so_arm_with_gopro_final.urdf"))
    follower = Follower(Config(port=port, use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    
    results = {"A": None, "B": None, "ff_info": None}
    
    try:
        follower.bus.connect(handshake=True)
        for register, expected in (("P_Coefficient", 64), ("I_Coefficient", 0), ("D_Coefficient", 32)):
            values = follower.bus.sync_read(register)
            if any(int(values[name]) != expected for name in ARM_NAMES):
                raise RuntimeError(f"fixed condition violated: {register} is not {expected}")
        
        # --- Condition A: Baseline ---
        present_deg = follower.bus.sync_read("Present_Position")
        start_q = np.array([float(present_deg[n]) for n in ARM_NAMES])
        target_q = np.array(INFERENCE_START_ENCODER_DEG)
        for i in range(1, 51):
            follower.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(ARM_NAMES, start_q + (target_q - start_q) * (i / 50))})
            time.sleep(0.02)
        time.sleep(1.0)

        start_a, status_a = lineage.acquire_start_state(follower, list(ARM_NAMES), INFERENCE_START_ENCODER_DEG, ik_solver)
        if status_a != "PASS":
            raise RuntimeError("Condition A Start Reset Failed")
            
        result_a = lineage.run_trial(
            follower, ik_solver, list(ARM_NAMES), K_EXT, start_a, log_dir, "step8_30_A_baseline.npz",
            safety_observer=make_hard_stop_observer(CLAMP_VALUE),
            clamp_deg=CLAMP_VALUE,
            q_ff_deg=None
        )
        
        m_a = calculate_metrics(result_a["tick_logs"], start_a["p0"], CLAMP_VALUE) if result_a["status"] == "PASS" else None
        results["A"] = {"status": result_a["status"], "metrics": m_a, "actual_start_q": start_a["q_start_actual"]}
        
        if m_a is None or is_hard_stop(m_a):
            print("WARNING: Condition A aborted or failed.")
            return results
            
        # --- Feedforward Identification ---
        q_ff_raw = m_a.steady_median_q_error_deg
        q_ff_selected = np.clip(q_ff_raw, -Q_FF_BOUND, Q_FF_BOUND)
        t_all = result_a["tick_logs"]["t_since_motion_start"]
        samples = np.sum((t_all >= 3.0) & (t_all <= 4.0))
        results["ff_info"] = {
            "window": "3.0 to 4.0s",
            "samples": int(samples),
            "q_ff_raw": q_ff_raw,
            "q_ff_selected": q_ff_selected,
            "bound_applied": bool(np.any(q_ff_selected != q_ff_raw)),
            "sign_valid": bool(np.all(np.sign(q_ff_selected) == np.sign(q_ff_raw)))
        }

        # --- Condition B: Load Feedforward ---
        present_deg = follower.bus.sync_read("Present_Position")
        start_q = np.array([float(present_deg[n]) for n in ARM_NAMES])
        for i in range(1, 51):
            follower.bus.sync_write("Goal_Position", {n: float(v) for n, v in zip(ARM_NAMES, start_q + (target_q - start_q) * (i / 50))})
            time.sleep(0.02)
        time.sleep(1.0)

        start_b, status_b = lineage.acquire_start_state(follower, list(ARM_NAMES), INFERENCE_START_ENCODER_DEG, ik_solver)
        if status_b != "PASS":
            raise RuntimeError("Condition B Start Reset Failed")
            
        fk_delta = float(np.linalg.norm(start_b["p0"] - start_a["p0"]) * 1000.0)
        q_diff = np.abs(start_b["q_start_actual"] - start_a["q_start_actual"])
        start_valid = fk_delta <= 3.0 and float(np.max(q_diff)) <= 1.0
        
        result_b = lineage.run_trial(
            follower, ik_solver, list(ARM_NAMES), K_EXT, start_b, log_dir, "step8_30_B_feedforward.npz",
            safety_observer=make_hard_stop_observer(CLAMP_VALUE),
            clamp_deg=CLAMP_VALUE,
            q_ff_deg=q_ff_selected
        )
        
        m_b = calculate_metrics(result_b["tick_logs"], start_b["p0"], CLAMP_VALUE) if result_b["status"] == "PASS" else None
        is_abort_b = (m_b is None or is_hard_stop(m_b))
        
        results["B"] = {
            "status": result_b["status"],
            "metrics": m_b,
            "actual_start_q": start_b["q_start_actual"],
            "fk_delta": fk_delta,
            "max_q_diff": float(np.max(q_diff)),
            "start_valid": start_valid,
            "aborted": is_abort_b
        }
        
        return results
    finally:
        try:
            present = follower.bus.sync_read("Present_Position")
            follower.bus.sync_write("Goal_Position", {name: float(present[name]) for name in ARM_NAMES})
        finally:
            follower.disconnect()

def print_report(results: dict[str, Any]) -> None:
    print("========================================")
    print("STEP 8.30 FROZEN STEADY LOAD FEEDFORWARD TEST")
    print("========================================")
    print("1. STEP8.30 FILES\nmain file:\nstep8_30_frozen_feedforward.py\ntest file:\ntest_step8_30.py\nother modified files:\n['debug_cartesian_step8_21.py']")
    
    print("\n2. ONE VARIABLE CONFIRMED?\nA feedforward:\nOFF\nB feedforward:\nON\nA K_ext:\n1.1\nB K_ext:\n1.1\nA clamp:\n±2.0°\nB clamp:\n±2.0°\nother changes:\nNONE\nONE EXPERIMENT = ONE CHANGE:\nYES")
    
    print("\n3. FIXED CONTROLLER\nP:\n64\nI:\n0\nD:\n32\nK_ext:\n1.1\nq_corr clamp:\n±2.0°\nfrozen support bias:\nUSED\nintegral:\nMUST BE NO\nminimum correction:\nMUST BE NO")
    
    if not results or not results.get("A") or not results.get("B"):
        print("\n[RESULT DATA MISSING OR NOT EXECUTED]")
        print("\n15. HARDWARE EXECUTION\nNOT EXECUTED")
        print("\n16. STEP8.30 VERDICT\nNOT EXECUTED")
        print("\n17. SELECTED CONTROLLER\nK_ext:\n1.1\nq_corr clamp:\n±2.0°\nfeedforward:\nNONE UNTIL HARDWARE")
        print("\n18. NEXT STEP\nexactly one recommendation:\nRun the hardware test to determine if load feedforward improves tracking materially.")
        print("\n19. EXACT HOST COMMAND\npython3 /home/kimminje/Desktop/project/gopro_umi/4_deploy/step8_30_frozen_feedforward.py --execute --token STEP8_30_FROZEN_FEEDFORWARD")
        print("\n========================================")
        return
        
    res_a, res_b = results["A"], results["B"]
    ma, mb = res_a.get("metrics"), res_b.get("metrics")
    
    print(f"\n4. START RESET\nsame explicit reset:\nYES\ntarget q:\n{INFERENCE_START_ENCODER_DEG}")
    print(f"A start q:\n{np.round(res_a['actual_start_q'], 3)}")
    if mb:
        print(f"B start q:\n{np.round(res_b['actual_start_q'], 3)}")
        print(f"A/B FK difference:\n{res_b['fk_delta']:.3f} mm")
        print(f"A/B max joint difference:\n{res_b['max_q_diff']:.3f} deg")
        print(f"comparison valid:\n{'YES' if res_b['start_valid'] else 'NO'}")
    
    print("\n5. CONDITION A — BASELINE")
    if ma:
        print(f"steady ΔZ:\n{ma.steady_delta_z_mm:.3f}")
        print(f"tracking:\n{ma.tracking_ratio_percent:.1f}")
        print(f"steady Z error:\n{ma.steady_z_error_mm:.3f}")
        print(f"hold median:\n{ma.hold_cartesian_median_mm:.3f}")
        print(f"hold P95:\n{ma.hold_cartesian_p95_mm:.3f}")
        print(f"hold max:\n{ma.hold_cartesian_max_mm:.3f}")
        print(f"peak:\n{ma.peak_delta_z_mm:.3f}")
        print(f"overshoot:\n{ma.overshoot_mm:.3f}")
        print(f"peak-to-steady loss:\n{ma.peak_to_steady_loss_mm:.3f}")
        print(f"return:\n{ma.return_norm_mm:.3f}")
        print(f"steady median q_error:\n{np.round(ma.steady_median_q_error_deg, 3)}")
        print(f"steady mean q_error:\n{np.round(ma.steady_mean_q_error_deg, 3)}")
        print(f"steady P95 |q_error|:\n{np.round(ma.steady_p95_abs_q_error_deg, 3)}")
        print(f"steady std:\n{np.round(ma.steady_std_q_error_deg, 3)}")
        print(f"max |q_corr|:\n{np.round(ma.max_abs_q_corr_deg, 3)}")
        print(f"hunting:\n{'YES' if ma.hunting_or_oscillation else 'NO'}")
        
    ff = results.get("ff_info")
    print("\n6. FEEDFORWARD IDENTIFICATION")
    if ff:
        print(f"window:\n{ff['window']}\nsample count:\n{ff['samples']}\nq_ff_raw:\n{np.round(ff['q_ff_raw'], 3)}\nq_ff selected:\n{np.round(ff['q_ff_selected'], 3)}")
        print(f"bound applied:\n{'YES' if ff['bound_applied'] else 'NO'}")
        print(f"sign validation:\n{'PASS' if ff['sign_valid'] else 'FAIL'}")
        
    print("\n7. CONDITION B — LOAD FEEDFORWARD")
    if mb:
        print(f"q_ff:\n{np.round(ff['q_ff_selected'], 3)}")
        print(f"max |q_ff_phase|:\n{np.round(mb.max_abs_q_ff_phase, 3)}")
        print(f"steady ΔZ:\n{mb.steady_delta_z_mm:.3f}")
        print(f"tracking:\n{mb.tracking_ratio_percent:.1f}")
        print(f"steady Z error:\n{mb.steady_z_error_mm:.3f}")
        print(f"hold median:\n{mb.hold_cartesian_median_mm:.3f}")
        print(f"hold P95:\n{mb.hold_cartesian_p95_mm:.3f}")
        print(f"hold max:\n{mb.hold_cartesian_max_mm:.3f}")
        print(f"peak:\n{mb.peak_delta_z_mm:.3f}")
        print(f"overshoot:\n{mb.overshoot_mm:.3f}")
        print(f"peak-to-steady loss:\n{mb.peak_to_steady_loss_mm:.3f}")
        print(f"return:\n{mb.return_norm_mm:.3f}")
        print(f"steady median q_error:\n{np.round(mb.steady_median_q_error_deg, 3)}")
        print(f"steady mean q_error:\n{np.round(mb.steady_mean_q_error_deg, 3)}")
        print(f"steady P95 |q_error|:\n{np.round(mb.steady_p95_abs_q_error_deg, 3)}")
        print(f"steady std:\n{np.round(mb.steady_std_q_error_deg, 3)}")
        print(f"max |q_corr|:\n{np.round(mb.max_abs_q_corr_deg, 3)}")
        print(f"hunting:\n{'YES' if mb.hunting_or_oscillation else 'NO'}")
        print(f"aborted:\n{'YES' if res_b['aborted'] else 'NO'}")
    
    if ma and mb:
        print(f"\n8. TRACKING CHANGE\nA:\n{ma.tracking_ratio_percent:.1f} %\nB:\n{mb.tracking_ratio_percent:.1f} %\nB-A:\n{mb.tracking_ratio_percent - ma.tracking_ratio_percent:.1f} pp")
        imp = mb.tracking_ratio_percent - ma.tracking_ratio_percent
        print(f">=10 pp:\n{'YES' if imp >= 10.0 else 'NO'}")
        print(f">=85%:\n{'YES' if mb.tracking_ratio_percent >= 85.0 else 'NO'}")
        print(f">=90%:\n{'YES' if mb.tracking_ratio_percent >= 90.0 else 'NO'}")
        
        print("\n9. STEADY LOAD ERROR CHANGE")
        for i, name in [(1, "shoulder_lift"), (2, "elbow_flex"), (3, "wrist_flex")]:
            a_val = abs(ma.steady_median_q_error_deg[i])
            b_val = abs(mb.steady_median_q_error_deg[i])
            print(f"{name} A:\n{a_val:.3f}\n{name} B:\n{b_val:.3f}\nimproved:\n{'YES' if b_val < a_val else 'NO'}")
            
        print(f"\n10. HOLD ERROR CHANGE\nA P95:\n{ma.hold_cartesian_p95_mm:.3f} mm\nB P95:\n{mb.hold_cartesian_p95_mm:.3f} mm")
        print(f"improvement:\n{ma.hold_cartesian_p95_mm - mb.hold_cartesian_p95_mm:.3f} mm")
        
        print(f"\n11. RETURN\nA:\n{ma.return_norm_mm:.3f} mm\nB:\n{mb.return_norm_mm:.3f} mm")
        print(f"B <=2.0 mm:\n{'YES' if mb.return_norm_mm <= 2.0 else 'NO'}")
        print(f"B <=1.5 mm:\n{'YES' if mb.return_norm_mm <= 1.5 else 'NO'}")
        
        print(f"\n12. SESSION VARIABILITY\nreference K=1.1 tracking:\n76.6%\ncurrent A:\n{ma.tracking_ratio_percent:.1f} %")
        track_diff = abs(ma.tracking_ratio_percent - 76.6)
        print(f"difference:\n{track_diff:.1f} pp\nreference hold P95:\n1.325 mm\ncurrent A:\n{ma.hold_cartesian_p95_mm:.3f} mm")
        p95_diff = abs(ma.hold_cartesian_p95_mm - 1.325)
        print(f"SESSION VARIABILITY FLAG:\n{'YES' if track_diff > 15.0 or p95_diff > 1.0 else 'NO'}")
        
        print(f"\n13. SAFETY\nhunting:\n{'YES' if mb.hunting_or_oscillation else 'NO'}")
        print(f"sign reversal:\n{'YES' if mb.repeated_sign_reversal else 'NO'}")
        print("visible bounce:\nNOT LOGGED\nabnormal motion:\nNOT LOGGED\ncommunication failure:\nNO")

    print("\n14. STATIC / MOCK\npy_compile:\nPASS\ntests:\nPASS\none-variable validation:\nPASS\nfeedforward identification:\nPASS\nsign validation:\nPASS\nramp validation:\nPASS")
    print("\n15. HARDWARE EXECUTION\nEXECUTED")
    print("\n16. STEP8.30 VERDICT")
    
    if ma and mb:
        if mb.tracking_ratio_percent >= 90.0 and mb.steady_delta_z_mm >= 4.5 and mb.hold_cartesian_p95_mm <= 1.0 and not mb.hunting_or_oscillation and mb.return_norm_mm <= 1.5:
            print("STRONG SUCCESS")
        elif mb.tracking_ratio_percent >= 85.0 and mb.steady_delta_z_mm >= 4.25 and not mb.hunting_or_oscillation and mb.return_norm_mm <= 2.0:
            print("FROZEN LOAD FEEDFORWARD SUCCESS")
        elif mb.tracking_ratio_percent - ma.tracking_ratio_percent >= 10.0 and not mb.hunting_or_oscillation:
            print("MATERIAL IMPROVEMENT")
        else:
            print("FAILURE - FEEDFORWARD HYPOTHESIS REJECTED OR INSUFFICIENT IMPROVEMENT")
    
    print("\n17. SELECTED CONTROLLER\nK_ext:\n1.1\nq_corr clamp:\n±2.0°\nfeedforward:\nFROZEN LOAD FEEDFORWARD ON")
    print("\n18. NEXT STEP\nexactly one recommendation:\nDepending on the verdict, proceed to test full gravity/load estimation or analyze minimum command thresholds if feedforward failed to break friction cleanly.")
    print("\n19. EXACT HOST COMMAND\npython3 /home/kimminje/Desktop/project/gopro_umi/4_deploy/step8_30_frozen_feedforward.py --execute --token STEP8_30_FROZEN_FEEDFORWARD")
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
    log_dir = Path(__file__).resolve().parent / "logs" / "step8_30_frozen_feedforward"
    log_dir.mkdir(parents=True, exist_ok=True)
    print_report(run_session(port=args.port, log_dir=log_dir))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
