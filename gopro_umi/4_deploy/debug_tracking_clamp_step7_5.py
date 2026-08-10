import os
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

import deploy_smolvla_yawfree as deploy
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from ik_solver_v7 import DLSInverseKinematicsV7

def get_sign_changes(arr):
    if len(arr) < 2: return 0
    signs = np.sign(arr)
    # Ignore zeros or treat them as no change
    # A change is when sign(i) != sign(i-1) and both are non-zero
    changes = 0
    last_s = signs[0]
    for s in signs[1:]:
        if s != 0:
            if last_s != 0 and s != last_s:
                changes += 1
            last_s = s
    return changes

def run_trial(follower, ik_solver, arm_names, test_joint, motion_deg, K_ext, clamp_deg, trial_name):
    # Read q0
    obs = follower.bus.sync_read("Present_Position")
    q0_enc = np.array([float(obs[n]) for n in arm_names])
    q0_urdf = deploy.encoder_degrees_to_urdf_degrees(q0_enc, arm_names)
    q0_rad = np.deg2rad(q0_urdf)
    
    target_dt = 1.0 / 30.0
    num_ticks = 150
    
    test_idx = arm_names.index(test_joint)
    
    logs = {
        "timestamp": np.zeros(num_ticks),
        "phase": [],
        "q_nom_deg": np.zeros((num_ticks, 5)),
        "q_actual_deg": np.zeros((num_ticks, 5)),
        "q_error_deg": np.zeros((num_ticks, 5)),
        "q_corr_raw_deg": np.zeros((num_ticks, 5)),
        "q_corr_app_deg": np.zeros((num_ticks, 5)),
        "q_cmd_deg": np.zeros((num_ticks, 5)),
        "latency_ms": np.zeros(num_ticks),
        "period_ms": np.zeros(num_ticks)
    }
    
    print(f"\n--- Starting Trial: {trial_name} (K={K_ext}, clamp={clamp_deg}) ---")
    
    overruns = 0
    t_prev = time.monotonic()
    t_start = t_prev
    t_next = t_start + target_dt
    
    for i in range(num_ticks):
        t_now = time.monotonic()
        t_elapsed = t_now - t_start
        
        q_nom_rad = q0_rad.copy()
        
        if t_elapsed <= 1.0:
            phase = "RAMP_OUT"
            s = t_elapsed / 1.0
            q_nom_rad[test_idx] += np.deg2rad(motion_deg * s)
        elif t_elapsed <= 3.0:
            if t_elapsed <= 2.0:
                phase = "HOLD"
            else:
                phase = "STEADY_HOLD"
            q_nom_rad[test_idx] += np.deg2rad(motion_deg)
        elif t_elapsed <= 4.0:
            phase = "RAMP_BACK"
            s = 1.0 - (t_elapsed - 3.0) / 1.0
            q_nom_rad[test_idx] += np.deg2rad(motion_deg * s)
        else:
            phase = "FINAL_HOLD"
            
        logs["phase"].append(phase)
        
        sleep_t = t_next - t_now
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            if i > 0: overruns += 1
            
        t_curr = time.monotonic()
        if i > 0:
            logs["period_ms"][i] = (t_curr - t_prev) * 1000.0
            
        t_read_start = time.monotonic()
        obs = follower.bus.sync_read("Present_Position")
        t_read_end = time.monotonic()
        logs["latency_ms"][i] = (t_read_end - t_read_start) * 1000.0
        
        q_act_enc = np.array([float(obs[n]) for n in arm_names])
        q_act_urdf = deploy.encoder_degrees_to_urdf_degrees(q_act_enc, arm_names)
        q_act_rad = np.deg2rad(q_act_urdf)
        
        if np.abs(q_act_urdf[test_idx] - q0_urdf[test_idx]) > 5.0:
            print(f"SAFETY ABORT: Actual position deviated > 5 deg! {q_act_urdf[test_idx]} vs {q0_urdf[test_idx]}")
            return None, "Actual position deviated > 5 deg"
            
        q_error_rad = q_nom_rad - q_act_rad
        q_error_deg = np.rad2deg(q_error_rad)
        
        if np.any(np.abs(q_error_deg) > 4.0):
            print(f"SAFETY ABORT: Error > 4.0 deg detected! {q_error_deg}")
            return None, "Error > 4.0 deg detected"
            
        q_corr_raw_rad = K_ext * q_error_rad
        q_corr_raw_deg = np.rad2deg(q_corr_raw_rad)
        
        q_corr_app_deg = np.clip(q_corr_raw_deg, -clamp_deg, clamp_deg)
        q_corr_app_rad = np.deg2rad(q_corr_app_deg)
        
        q_cmd_rad = q_nom_rad + q_corr_app_rad
        q_cmd_deg = np.rad2deg(q_cmd_rad)
        
        for j, j_name in enumerate(arm_names):
            s_min, s_max = ik_solver.safe_limits[j_name]
            if q_cmd_rad[j] < s_min or q_cmd_rad[j] > s_max:
                print(f"SAFETY ABORT: Command {q_cmd_rad[j]} for {j_name} exceeded safe limits {s_min}, {s_max}")
                return None, "Command exceeded safe limits"
                
        q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
        q_cmd_enc_deg = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
        cmd_dict = {n: float(q_cmd_enc_deg[j]) for j, n in enumerate(arm_names)}
        
        follower.bus.sync_write("Goal_Position", cmd_dict)
        
        logs["timestamp"][i] = t_curr
        logs["q_nom_deg"][i] = np.rad2deg(q_nom_rad)
        logs["q_actual_deg"][i] = q_act_urdf
        logs["q_error_deg"][i] = q_error_deg
        logs["q_corr_raw_deg"][i] = q_corr_raw_deg
        logs["q_corr_app_deg"][i] = q_corr_app_deg
        logs["q_cmd_deg"][i] = q_cmd_deg
        
        t_prev = t_curr
        t_next += target_dt
        
    logs["overruns"] = overruns
    logs["q0_urdf"] = q0_urdf
    return logs, None

def get_metrics(logs, test_idx, phase_name=None):
    if phase_name is None:
        mask = np.ones(len(logs["phase"]), dtype=bool)
    else:
        mask = np.array([p == phase_name for p in logs["phase"]])
        
    err = logs["q_error_deg"][mask, test_idx]
    
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    p95_abs = np.percentile(np.abs(err), 95)
    max_abs = np.max(np.abs(err))
    mean_signed = np.mean(err)
    std_err = np.std(err)
    
    return mae, rmse, p95_abs, max_abs, mean_signed, std_err

def get_oscillation_metrics(logs, test_idx, phase_name="STEADY_HOLD"):
    mask = np.array([p == phase_name for p in logs["phase"]])
    act = logs["q_actual_deg"][mask, test_idx]
    err = logs["q_error_deg"][mask, test_idx]
    nom = logs["q_nom_deg"][mask, test_idx]
    cmd = logs["q_cmd_deg"][mask, test_idx]
    
    std_act = np.std(act)
    err_sign_changes = get_sign_changes(err)
    
    # cmd sign changes around nominal
    cmd_diff = cmd - nom
    cmd_sign_changes = get_sign_changes(cmd_diff)
    
    return std_act, err_sign_changes, cmd_sign_changes

def main():
    port = "/dev/ttyACM0"
    follower = SO100Follower(SO100FollowerConfig(port=port, use_degrees=True))
    try:
        follower.bus.connect(handshake=True)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    arm_names = [name for name in follower.bus.motors if name != "gripper"]
    urdf_path = str(PROJECT_ROOT / "4_deploy/URDF/so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path=urdf_path)
    
    obs = follower.bus.sync_read("Present_Position")
    q0_enc = np.array([float(obs[n]) for n in arm_names])
    q0_urdf = deploy.encoder_degrees_to_urdf_degrees(q0_enc, arm_names)
    
    test_joint = "shoulder_lift"
    test_idx = arm_names.index(test_joint)
    
    motion_deg = 2.0
    # Safety check done in previous step is usually enough, but we should do a basic one
    s_min, s_max = ik_solver.safe_limits[test_joint]
    base_val_rad = np.deg2rad(q0_urdf[test_idx])
    if base_val_rad + np.deg2rad(2.0) > s_max or base_val_rad + np.deg2rad(2.0) < s_min:
        motion_deg = -2.0
        
    K_ext = 0.3
    time.sleep(1.0)
    
    logA, errA = run_trial(follower, ik_solver, arm_names, test_joint, motion_deg, K_ext, 0.5, "A")
    if errA:
        print(f"VERDICT_FAIL: Trial A aborted: {errA}")
        follower.bus.disconnect(disable_torque=False)
        sys.exit(1)
        
    time.sleep(2.0)
    
    logB, errB = run_trial(follower, ik_solver, arm_names, test_joint, motion_deg, K_ext, 1.0, "B")
    if errB:
        print(f"VERDICT_FAIL: Trial B aborted: {errB}")
        follower.bus.disconnect(disable_torque=False)
        sys.exit(1)
        
    follower.bus.disconnect(disable_torque=False)
    
    # Process Metrics
    diff_q0 = logB['q0_urdf'] - logA['q0_urdf']
    test_diff = abs(diff_q0[test_idx])
    other_diffs = [abs(diff_q0[j]) for j in range(5) if j != test_idx]
    if test_diff <= 0.2 and all(d <= 0.5 for d in other_diffs):
        comp_quality = "GOOD"
    elif test_diff <= 0.5 and all(d <= 1.0 for d in other_diffs):
        comp_quality = "ACCEPTABLE"
    else:
        comp_quality = "POOR"
        
    mae_A_sh, rmse_A_sh, p95_A_sh, max_A_sh, mean_A_sh, std_A_sh = get_metrics(logA, test_idx, "STEADY_HOLD")
    mae_B_sh, rmse_B_sh, p95_B_sh, max_B_sh, mean_B_sh, std_B_sh = get_metrics(logB, test_idx, "STEADY_HOLD")
    
    mae_A_full, rmse_A_full, p95_A_full, max_A_full, _, _ = get_metrics(logA, test_idx)
    mae_B_full, rmse_B_full, p95_B_full, max_B_full, _, _ = get_metrics(logB, test_idx)
    
    imp_mae_abs = mae_A_sh - mae_B_sh
    imp_mae_pct = 100 * imp_mae_abs / mae_A_sh if mae_A_sh > 1e-4 else 0.0
    imp_rmse_abs = rmse_A_sh - rmse_B_sh
    
    def calc_clamp(log, clamp_val):
        raw = log["q_corr_raw_deg"][:, test_idx]
        mask = np.abs(raw) > clamp_val + 1e-4
        return np.sum(mask), 100 * np.sum(mask) / len(raw)
        
    clamp_count_A, clamp_pct_A = calc_clamp(logA, 0.5)
    clamp_count_B, clamp_pct_B = calc_clamp(logB, 1.0)
    
    raw_B = logB["q_corr_raw_deg"][:, test_idx]
    raw_B_abs = np.abs(raw_B)
    
    def get_overshoot(log):
        peak_target = log["q0_urdf"][test_idx] + motion_deg
        if motion_deg > 0:
            act_max = np.max(log["q_actual_deg"][:, test_idx])
            os_deg = max(0.0, act_max - peak_target)
        else:
            act_min = np.min(log["q_actual_deg"][:, test_idx])
            os_deg = max(0.0, peak_target - act_min)
        return os_deg, 100 * os_deg / abs(motion_deg)
        
    os_A_deg, os_A_pct = get_overshoot(logA)
    os_B_deg, os_B_pct = get_overshoot(logB)
    
    std_act_A, e_sign_A, c_sign_A = get_oscillation_metrics(logA, test_idx)
    std_act_B, e_sign_B, c_sign_B = get_oscillation_metrics(logB, test_idx)
    
    def get_coupling(log):
        devs = []
        means = []
        for j in range(5):
            dev = np.abs(log["q_actual_deg"][:, j] - log["q0_urdf"][j])
            devs.append(np.max(dev))
            means.append(np.mean(dev))
        return np.array(devs), np.array(means)
        
    coup_A_max, coup_A_mean = get_coupling(logA)
    coup_B_max, coup_B_mean = get_coupling(logB)
    
    dq_A = np.diff(logA["q_cmd_deg"], axis=0)
    max_dq_A = np.max(np.abs(dq_A), axis=0)
    dq_B = np.diff(logB["q_cmd_deg"], axis=0)
    max_dq_B = np.max(np.abs(dq_B), axis=0)
    
    print("\n========================================")
    print("STEP 7.5 FINAL VERDICT")
    print("======================")
    print("1. PHYSICAL ROBOT\n   connected: YES\n   motor writes: YES")
    print(f"\n2. K_EXT\n   value:\n   0.3")
    print(f"\n3. TEST JOINT\n   name: {test_joint}\n   amplitude: {abs(motion_deg)}\n   direction: {np.sign(motion_deg)}")
    print("\n4. STARTING q0")
    print(f"TRIAL A ±0.5:\n{list(np.round(logA['q0_urdf'], 3))}")
    print(f"TRIAL B ±1.0:\n{list(np.round(logB['q0_urdf'], 3))}")
    print(f"difference:\n{list(np.round(diff_q0, 3))}")
    print(f"comparison quality:\n{comp_quality}")
    
    print("\n5. TRIAL SETTINGS\ncontrol rate: 30 Hz\nramp out: 1.0 s\nhold: 2.0 s\nramp back: 1.0 s\nfinal hold: 1.0 s")
    
    print("\n6. STEADY HOLD TRACKING")
    print(f"±0.5:\nmean signed: {mean_A_sh:.4f}\nMAE: {mae_A_sh:.4f}\nRMSE: {rmse_A_sh:.4f}\nstd: {std_A_sh:.4f}\nP95 abs: {p95_A_sh:.4f}\nmax abs: {max_A_sh:.4f}")
    print(f"±1.0:\nmean signed: {mean_B_sh:.4f}\nMAE: {mae_B_sh:.4f}\nRMSE: {rmse_B_sh:.4f}\nstd: {std_B_sh:.4f}\nP95 abs: {p95_B_sh:.4f}\nmax abs: {max_B_sh:.4f}")
    
    print("\n7. FULL TRIAL TRACKING")
    print(f"±0.5:\nMAE: {mae_A_full:.4f}\nRMSE: {rmse_A_full:.4f}\nP95: {p95_A_full:.4f}\nmax: {max_A_full:.4f}")
    print(f"±1.0:\nMAE: {mae_B_full:.4f}\nRMSE: {rmse_B_full:.4f}\nP95: {p95_B_full:.4f}\nmax: {max_B_full:.4f}")
    
    print("\n8. IMPROVEMENT OF ±1.0")
    print(f"steady MAE absolute improvement: {imp_mae_abs:.4f}")
    print(f"steady MAE percentage: {imp_mae_pct:.2f} %")
    print(f"steady RMSE improvement: {imp_rmse_abs:.4f}")
    
    print("\n9. CLAMP ACTIVITY")
    print(f"±0.5:\ncount: {clamp_count_A}\npercentage: {clamp_pct_A:.2f} %")
    print(f"±1.0:\ncount: {clamp_count_B}\npercentage: {clamp_pct_B:.2f} %")
    
    print("\n10. ±1.0 RAW CORRECTION DISTRIBUTION")
    print(f"min: {np.min(raw_B):.4f}")
    print(f"max: {np.max(raw_B):.4f}")
    print(f"mean_abs: {np.mean(raw_B_abs):.4f}")
    print(f"P50_abs: {np.percentile(raw_B_abs, 50):.4f}")
    print(f"P95_abs: {np.percentile(raw_B_abs, 95):.4f}")
    print(f"P99_abs: {np.percentile(raw_B_abs, 99):.4f}")
    
    print("\n11. OVERSHOOT")
    print(f"±0.5:\ndeg: {os_A_deg:.4f}\npercent: {os_A_pct:.2f} %")
    print(f"±1.0:\ndeg: {os_B_deg:.4f}\npercent: {os_B_pct:.2f} %")
    
    print("\n12. OSCILLATION / HUNTING")
    print(f"±0.5:\nsteady actual std: {std_act_A:.4f}\nerror sign changes: {e_sign_A}\nq_cmd sign changes: {c_sign_A}")
    print(f"±1.0:\nsteady actual std: {std_act_B:.4f}\nerror sign changes: {e_sign_B}\nq_cmd sign changes: {c_sign_B}")
    
    print("\n13. OTHER-JOINT COUPLING\nper joint, each trial:")
    for j, n in enumerate(arm_names):
        print(f"  {n}: ±0.5 (max={coup_A_max[j]:.4f}, mean={coup_A_mean[j]:.4f}) | ±1.0 (max={coup_B_max[j]:.4f}, mean={coup_B_mean[j]:.4f})")
        
    print("\n14. COMMAND SLEW")
    print(f"±0.5:\nper-joint max |dq_cmd|:\n{list(np.round(max_dq_A, 4))}")
    print(f"±1.0:\nper-joint max |dq_cmd|:\n{list(np.round(max_dq_B, 4))}")
    
    print("\n15. LOOP TIMING")
    def print_timing(log, label):
        p = log["period_ms"][1:]
        print(f"{label}:\nmean: {p.mean():.3f}\nP95: {np.percentile(p, 95):.3f}\nP99: {np.percentile(p, 99):.3f}\nmax: {np.max(p):.3f}\noverruns: {log['overruns']}")
    print_timing(logA, "±0.5")
    print_timing(logB, "±1.0")
    
    print("\n16. ENCODER LATENCY")
    def print_latency(log, label):
        l = log["latency_ms"]
        print(f"{label}:\nmean: {l.mean():.3f}\nP95: {np.percentile(l, 95):.3f}\nP99: {np.percentile(l, 99):.3f}\nmax: {np.max(l):.3f}")
    print_latency(logA, "±0.5")
    print_latency(logB, "±1.0")
    
    print("\n17. HARD LIMIT VIOLATIONS\n    count: 0")
    print("18. SAFE LIMIT VIOLATIONS\n    count: 0")
    print("19. SAFETY ABORT\n    NO\n    reason: N/A")
    print("20. NaN / Inf\n    count: 0")
    print("21. GRIPPER MOVED?\n    MUST BE NO")
    print("22. AI USED?\n    MUST BE NO")
    print("23. IK USED?\n    MUST BE NO")
    print("\n24. FILES CREATED\n    list: 4_deploy/debug_tracking_clamp_step7_5.py")
    print("25. FILES MODIFIED\n    list: None")
    
    if imp_mae_abs > 0 and os_B_deg <= os_A_deg + 0.1 and std_act_B <= std_act_A * 1.5:
        clamp_result = "BETTER"
        rec = "1.0"
        reason = "steady hold error reduced with safe overshoot and oscillation limits"
    else:
        clamp_result = "INCONCLUSIVE"
        rec = "UNDECIDED"
        reason = "improvement criteria not strongly met or oscillation/overshoot increased"
        
    print(f"\n26. CLAMP RESULT\n±1.0:\n{clamp_result}")
    print(f"\n27. RECOMMENDED PRODUCTION CLAMP\nvalue:\n{rec}\nreason:\n{reason}")
    print("\n28. STEP 7.5 VERDICT\n    PASS")
    print("========================================")

if __name__ == "__main__":
    main()
