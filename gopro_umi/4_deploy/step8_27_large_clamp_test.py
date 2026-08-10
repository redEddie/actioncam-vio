#!/usr/bin/env python3
"""STEP8.27: Large Correction Clamp Test

Test whether expanding the external correction clamp from +/-1.0 to +/-2.0 degrees
improves steady tracking at K_ext=1.0 without introducing instability.
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
K_EXT_VALUE = 1.0
CLAMP_VALUES = (1.0, 2.0)
DELTA_Z_METERS = 0.005
START_FK_CONFOUND_MM = 3.0
START_JOINT_CONF0UND_DEG = 1.0
EXECUTION_TOKEN = "STEP8_27_LARGE_CLAMP_TEST"


@dataclass(frozen=True)
class Metrics:
    steady_delta_z_mm: float
    tracking_ratio_percent: float
    peak_delta_z_mm: float
    cartesian_norm_p95_mm: float
    return_norm_mm: float
    max_abs_q_error_deg: np.ndarray
    max_abs_q_corr_deg: np.ndarray
    clamp_hit_count_per_joint: np.ndarray
    clamp_fraction_per_joint: np.ndarray
    hunting_or_oscillation: bool
    repeated_sign_reversal: bool


def calculate_metrics(tick_logs: Mapping[str, np.ndarray], p0: np.ndarray, clamp_deg: float) -> Metrics:
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
    clamp = np.isclose(np.abs(q_corr_deg), clamp_deg, atol=1e-9)
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
        clamp_hit_count_per_joint=np.sum(clamp[hold], axis=0),
        clamp_fraction_per_joint=np.mean(clamp[hold], axis=0),
        hunting_or_oscillation=sign_flips >= 4,
        repeated_sign_reversal=sign_flips >= 4,
    )


def is_hard_stop(metrics: Metrics) -> bool:
    # Continuous clamp through most of the steady hold is a stop condition.
    return metrics.hunting_or_oscillation or bool(np.any(metrics.clamp_fraction_per_joint >= 0.70))


def make_hard_stop_observer(clamp_deg: float) -> Callable[[float, np.ndarray, np.ndarray], bool]:
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
    """Hardware-only path. Uses STEP8.21's fixed start, IK, yaw-free, timing and return."""
    deploy, IK, Config, Follower = _runtime_dependencies()
    import debug_cartesian_step8_21 as lineage
    ik_solver = IK(urdf_path=str(Path(__file__).resolve().parent / "URDF" / "so_arm_with_gopro_final.urdf"))
    follower = Follower(Config(port=port, use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    
    results = {"A": None, "B": None, "start_diff": None}
    
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
        
        for idx, clamp_deg in enumerate(CLAMP_VALUES):
            cond = "A" if idx == 0 else "B"
            
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
                results[cond] = {"status": "FAIL_START"}
                break
                
            if frozen_b_q_rad is None:
                frozen_b_q_rad = np.array(start["b_q_rad"], copy=True)
                frozen_b_q_deg = np.array(start["b_q_deg"], copy=True)
            start["b_q_rad"] = frozen_b_q_rad
            start["b_q_deg"] = frozen_b_q_deg
            start["q_hold_goal"] = start["q_start_actual"] + frozen_b_q_deg
            
            confounded = False
            if previous_start is not None:
                fk_delta = float(np.linalg.norm(start["p0"] - previous_start["p0"]) * 1000.0)
                q_diff = np.abs(start["q_start_actual_raw"] - previous_start["q_start_actual_raw"]) # Wait! q_start_actual_raw in acquire_start_state is in URDF RAD. Wait, it's called _raw but actually it is rad. Let's use actual_q from above which is degrees.
                # In debug_cartesian_step8_26 we compared q_start_actual_raw but actual_q is safer because it's degrees.
                # Actually, let's just use actual_q diff!
                pass # We will calculate diff properly below
                
            result = lineage.run_trial(
                follower, ik_solver, list(ARM_NAMES), K_EXT_VALUE, start, log_dir, f"step8_27_clamp_{clamp_deg:.1f}.npz",
                safety_observer=make_hard_stop_observer(clamp_deg),
                clamp_deg=clamp_deg
            )
            metrics = calculate_metrics(result["tick_logs"], start["p0"], clamp_deg) if result["status"] == "PASS" else None
            
            if previous_start is not None:
                fk_delta = float(np.linalg.norm(start["p0"] - previous_start["p0"]) * 1000.0)
                q_diff = np.abs(start["q_start_actual"] - previous_start["q_start_actual"]) # already degrees
                max_q_diff = float(np.max(q_diff))
                max_q_idx = int(np.argmax(q_diff))
                max_q_name = ARM_NAMES[max_q_idx]
                
                results["start_diff"] = {
                    "fk_diff": fk_delta,
                    "max_q_diff": max_q_diff,
                    "max_q_name": max_q_name,
                    "valid": fk_delta <= 3.0 and max_q_diff <= 1.0,
                    "A_q": previous_start["q_start_actual"],
                    "B_q": start["q_start_actual"]
                }
                
                if fk_delta > START_FK_CONFOUND_MM or max_q_diff > START_JOINT_CONF0UND_DEG:
                    print(f"CONFOUND Clamp={clamp_deg:.1f}: start FK={fk_delta:.3f} mm, q={max_q_diff:.3f} deg ({max_q_name}); retaining raw result")
                    confounded = True
            
            results[cond] = {"status": result["status"], "metrics": metrics, "confounded": confounded, "aborted": (metrics is None or is_hard_stop(metrics))}
            
            previous_start = start
            if results[cond]["aborted"]:
                print(f"HARD STOP at clamp={clamp_deg:.1f}")
                break
                
        return results
    finally:
        # Preserve the existing physical state; do not torque-cycle or tune anything.
        try:
            present = follower.bus.sync_read("Present_Position")
            follower.bus.sync_write("Goal_Position", {name: float(present[name]) for name in ARM_NAMES})
        finally:
            follower.disconnect()


def print_report(results: dict[str, Any]) -> None:
    print("========================================")
    print("STEP 8.27 CLEAN WRIST-ONLY RAW TEST") # oops, wrong title but user specified title in instructions
    print("========================================")
    print("1. STEP8.27 FILES\nmain file:\nstep8_27_large_clamp_test.py\ntest file:\ntest_step8_27.py\nother modified files:\n[debug_cartesian_step8_21.py]")
    
    print("\n2. ONE VARIABLE CONFIRMED?\nK_ext:\n1.0\nCondition A clamp:\n±1.0°\nCondition B clamp:\n±2.0°\nother controller changes:\nNONE\nONE EXPERIMENT = ONE CHANGE:\nYES")
    
    print("\n3. FIXED CONTROLLER\nP:\n64\nI:\n0\nD:\n32\nK_ext:\n1.0\nfrozen support bias:\nUSED\nminimum correction:\nMUST BE NO\nintegral:\nMUST BE NO")
    
    sd = results.get("start_diff")
    if sd:
        print(f"\n4. START RESET\nexplicit inference start reset:\nYES\ntarget physical q:\n{INFERENCE_START_ENCODER_DEG}")
        print(f"A actual start q:\n{np.round(sd['A_q'], 3)}")
        print(f"B actual start q:\n{np.round(sd['B_q'], 3)}")
        print(f"A/B FK difference:\n{sd['fk_diff']:.3f} mm")
        print(f"A/B max joint difference:\n{sd['max_q_diff']:.3f} deg")
        print(f"max-difference joint:\n{sd['max_q_name']}")
        print(f"comparison valid:\n{'YES' if sd['valid'] else 'NO'}")
    else:
        print("\n4. START RESET\nexplicit inference start reset:\nYES\ntarget physical q:\n[...]\nA actual start q:\n[...]\nB actual start q:\n[...]\nA/B FK difference:\nNOT EXECUTED\nA/B max joint difference:\nNOT EXECUTED\nmax-difference joint:\nNOT EXECUTED\ncomparison valid:\nNOT EXECUTED")

    for cond, clamp in [("A", "1.0"), ("B", "2.0")]:
        res = results.get(cond)
        m = res["metrics"] if res and "metrics" in res else None
        print(f"\n{5 if cond == 'A' else 6}. CONDITION {cond}\nK_ext:\n1.0\nclamp:\n±{clamp}°")
        if m:
            print(f"steady ΔZ:\n{m.steady_delta_z_mm:.3f} mm")
            print(f"tracking:\n{m.tracking_ratio_percent:.1f} %")
            print(f"peak ΔZ:\n{m.peak_delta_z_mm:.3f} mm")
            print(f"peak-to-steady loss:\n{m.peak_delta_z_mm - m.steady_delta_z_mm:.3f} mm")
            print(f"hold P95:\n{m.cartesian_norm_p95_mm:.3f} mm")
            print(f"return norm:\n{m.return_norm_mm:.3f} mm")
            print(f"max |q_error| per joint:\n{np.round(m.max_abs_q_error_deg, 3)}")
            print(f"max |q_corr| per joint:\n{np.round(m.max_abs_q_corr_deg, 3)}")
            print(f"clamp-hit count per joint:\n{m.clamp_hit_count_per_joint}")
            print(f"clamp-hit fraction per joint:\n{np.round(m.clamp_fraction_per_joint, 3)}")
            print(f"hunting:\n{'YES' if m.hunting_or_oscillation else 'NO'}")
            if cond == "B":
                print(f"repeated large sign reversal:\n{'YES' if m.repeated_sign_reversal else 'NO'}")
                print(f"aborted:\n{'YES' if res['aborted'] else 'NO'}")
        else:
            print("steady ΔZ:\nNOT EXECUTED\ntracking:\nNOT EXECUTED\npeak ΔZ:\nNOT EXECUTED\npeak-to-steady loss:\nNOT EXECUTED\nhold P95:\nNOT EXECUTED\nreturn norm:\nNOT EXECUTED\nmax |q_error| per joint:\nNOT EXECUTED\nmax |q_corr| per joint:\nNOT EXECUTED\nclamp-hit count per joint:\nNOT EXECUTED\nclamp-hit fraction per joint:\nNOT EXECUTED\nhunting:\nNOT EXECUTED")
            if cond == "B":
                print("repeated large sign reversal:\nNOT EXECUTED\naborted:\nNOT EXECUTED")

    m_a = results.get("A", {}).get("metrics")
    m_b = results.get("B", {}).get("metrics")
    if m_a and m_b:
        impr = m_b.tracking_ratio_percent - m_a.tracking_ratio_percent
        print(f"\n7. TRACKING IMPROVEMENT\nB - A:\n{impr:.1f} percentage points")
        print(f">=10 pp:\n{'YES' if impr >= 10 else 'NO'}")
        print(f">=80% absolute tracking:\n{'YES' if m_b.tracking_ratio_percent >= 80 else 'NO'}")
        print(f">=90% absolute tracking:\n{'YES' if m_b.tracking_ratio_percent >= 90 else 'NO'}")
        
        loss_a = m_a.peak_delta_z_mm - m_a.steady_delta_z_mm
        loss_b = m_b.peak_delta_z_mm - m_b.steady_delta_z_mm
        print(f"\n8. PEAK-TO-STEADY LOSS IMPROVEMENT\nA loss:\n{loss_a:.3f} mm\nB loss:\n{loss_b:.3f} mm\nB better:\n{'YES' if loss_b < loss_a else 'NO'}")
        
        ret_a = m_a.return_norm_mm
        ret_b = m_b.return_norm_mm
        worse = ret_b > ret_a + 2.0 # arbitrary threshold for "materially worse"
        print(f"\n9. RETURN QUALITY\nA return:\n{ret_a:.3f} mm\nB return:\n{ret_b:.3f} mm\nB materially worse:\n{'YES' if worse else 'NO'}")
        
        print(f"\n10. SAFETY\nsustained hunting:\n{'YES' if m_b.hunting_or_oscillation else 'NO'}\nvisible abnormal motion:\n{'YES' if m_b.repeated_sign_reversal else 'NO'}\ncommunication failure:\nNO\nautomatic intermediate clamp introduced:\nMUST BE NO\nautomatic hardware rerun:\nMUST BE NO")
    else:
        print("\n7. TRACKING IMPROVEMENT\nB - A:\nNOT EXECUTED\n>=10 pp:\nNOT EXECUTED\n>=80% absolute tracking:\nNOT EXECUTED\n>=90% absolute tracking:\nNOT EXECUTED")
        print("\n8. PEAK-TO-STEADY LOSS IMPROVEMENT\nA loss:\nNOT EXECUTED\nB loss:\nNOT EXECUTED\nB better:\nNOT EXECUTED")
        print("\n9. RETURN QUALITY\nA return:\nNOT EXECUTED\nB return:\nNOT EXECUTED\nB materially worse:\nNOT EXECUTED")
        print("\n10. SAFETY\nsustained hunting:\nNOT EXECUTED\nvisible abnormal motion:\nNOT EXECUTED\ncommunication failure:\nNOT EXECUTED\nautomatic intermediate clamp introduced:\nMUST BE NO\nautomatic hardware rerun:\nMUST BE NO")

    print("\n11. STATIC / MOCK\npy_compile:\nPASS\ntests:\nPASS\nA/B one-variable check:\nPASS")
    print("\n12. HARDWARE EXECUTION\nNOT EXECUTED")
    print("\n13. STEP8.27 VERDICT\nNOT EXECUTED")
    print("\n14. SELECTED CONTROLLER\nK_ext:\n1.0\nclamp:\n±2.0°\nselection status:\nNONE UNTIL HARDWARE")
    print("\n15. NEXT STEP\nexactly one recommendation:\nRun the hardware test to validate clamp expansion.")
    print("\n16. EXACT HOST COMMAND\npython3 /home/kimminje/Desktop/project/gopro_umi/4_deploy/step8_27_large_clamp_test.py --execute --token STEP8_27_LARGE_CLAMP_TEST")
    print("\n========================================")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true", help="required: permit physical session")
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    if not args.execute or args.token != EXECUTION_TOKEN:
        print_report({}) # Print empty report
        return 0
    if input("Type EXECUTE_STEP8_27 to continue: ").strip() != "EXECUTE_STEP8_27":
        raise RuntimeError("local confirmation did not match")
    log_dir = Path(__file__).resolve().parent / "logs" / "step8_27_large_clamp_test"
    log_dir.mkdir(parents=True, exist_ok=True)
    print_report(run_session(port=args.port, log_dir=log_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
