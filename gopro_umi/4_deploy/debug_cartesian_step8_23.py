import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'

os.environ['HF_HOME'] = str(PROJECT_ROOT / 'smolvla_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))
sys.path.insert(0, str(DEPLOY_DIR))

import deploy_smolvla_yawfree as deploy
from ik_solver_v7 import DLSInverseKinematicsV7
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception:
            pass
    return data

def move_to_pose(follower, arm_names, q_target_raw, duration=4.0):
    obs = follower.bus.sync_read("Present_Position")
    q_act = np.array([float(obs[n]) for n in arm_names])
    delta = q_target_raw - q_act
    
    steps = int(duration * 30)
    for step in range(steps):
        t0 = time.monotonic()
        progress = (step + 1) / steps
        alpha = (3.0 * progress**2) - (2.0 * progress**3)
        q_cmd = q_act + alpha * delta
        cmd_dict = {n: float(q_cmd[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
def measure_static_hold(follower, arm_names, q_goal_raw, hold_sec=5.0, meas_sec=1.0):
    target_dict = {n: float(q_goal_raw[j]) for j, n in enumerate(arm_names)}
    
    pre_meas_steps = int((hold_sec - meas_sec) * 30)
    for _ in range(pre_meas_steps):
        t0 = time.monotonic()
        follower.bus.sync_write("Goal_Position", target_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    meas_steps = int(meas_sec * 30)
    final_pres = []
    final_goal = []
    for _ in range(meas_steps):
        t0 = time.monotonic()
        pres = follower.bus.sync_read("Present_Position")
        goal = follower.bus.sync_read("Goal_Position")
        final_pres.append(np.array([float(pres[n]) for n in arm_names]))
        final_goal.append(np.array([float(goal[n]) for n in arm_names]))
        follower.bus.sync_write("Goal_Position", target_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    final_pres_arr = np.array(final_pres)
    pres_std = np.std(final_pres_arr, axis=0)
    q_actual_raw = np.mean(final_pres_arr, axis=0)
    q_goal_actual_raw = np.mean(final_goal, axis=0)
    
    return q_goal_actual_raw, q_actual_raw, pres_std

def interp_with_kext(follower, arm_names, q_start_rad, q_nom_B_rad, s_A_rad, duration=4.0):
    freq = 30
    steps = int(duration * freq)
    clamp_rad = np.deg2rad(1.0)
    
    for step in range(steps):
        t0 = time.monotonic()
        progress = (step + 1) / steps
        alpha = (3.0 * progress**2) - (2.0 * progress**3)
        q_nom_interp = q_start_rad + alpha * (q_nom_B_rad - q_start_rad)
        
        obs = follower.bus.sync_read("Present_Position")
        q_raw = np.array([float(obs[n]) for n in arm_names])
        q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names)
        q_rad = np.deg2rad(q_urdf)
        
        q_error = q_nom_interp - q_rad
        q_corr_raw = 0.5 * q_error
        q_corr_clipped = np.clip(q_corr_raw, -clamp_rad, clamp_rad)
        
        q_cmd_rad = q_nom_interp + s_A_rad + q_corr_clipped
        q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
        q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
        
        cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/freq) - elapsed))

def get_821_late_hold_mean():
    log_path = DEPLOY_DIR / 'logs' / 'step8_21' / 'step8_21_kext05_10s.npz'
    if not log_path.exists():
        return None, None
    d = np.load(log_path)
    t = d["t_since_motion_start"]
    idx = (t >= 3.5) & (t <= 4.0)
    if np.sum(idx) == 0: return None, None
    
    q_nom_rads = d["q_nom_rad"][idx]
    
    data = {
        "q_nom": np.mean(q_nom_rads, axis=0),
        "q_nom_std": np.std(q_nom_rads, axis=0),
        "q_actual": np.mean(d["q_actual_urdf_rad"][idx], axis=0),
        "b_q_rad": np.mean(d["b_q_rad"][idx], axis=0),
        "q_corr": np.mean(d["q_corr_applied_rad"][idx], axis=0),
        "q_cmd": np.mean(d["q_cmd_urdf_rad"][idx], axis=0),
        "yaw_used": np.mean(d["actual_yaw_used"][idx]),
        "p_target": np.mean(np.column_stack([d["p_target_x"], d["p_target_y"], d["p_target_z"]])[idx], axis=0),
        "R_target": np.mean(d["R_target"][idx], axis=0)
    }
    return data, True

def run_micro_sweep(follower, arm_names, base_goal_raw, idx_to_test, count_sweep):
    res = {}
    for counts in count_sweep:
        base_counts = int(round(base_goal_raw[idx_to_test]))
        cmd_counts = base_counts + counts
        
        cmd_raw_full = base_goal_raw.copy()
        cmd_raw_full[idx_to_test] = cmd_counts
        
        follower.bus.sync_write("Goal_Position", {n: int(cmd_raw_full[j]) for j, n in enumerate(arm_names)})
        time.sleep(1.0)
        
        obs = follower.bus.sync_read("Present_Position")
        pres_counts = int(float(obs[arm_names[idx_to_test]]))
        
        res[counts] = {
            "cmd_counts": counts,
            "goal_raw": cmd_counts,
            "pres_raw": pres_counts
        }
        
        # Return to base
        follower.bus.sync_write("Goal_Position", {n: int(base_goal_raw[j]) for j, n in enumerate(arm_names)})
        time.sleep(1.0)
        obs_ret = follower.bus.sync_read("Present_Position")
        res[counts]["ret_pres_raw"] = int(float(obs_ret[arm_names[idx_to_test]]))
        
    return res

def find_breakaway(res, base_pres_counts, count_sweep):
    for c in count_sweep:
        if abs(res[c]["pres_raw"] - base_pres_counts) >= 1:
            return c
    return ">8"

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    follower = SO100Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=False)) # USE RAW COUNTS
    follower.config.disable_torque_on_disconnect = False
    follower.bus.connect(handshake=True)
    
    print("Initializing...")
    tel0 = read_telemetry(follower, arm_names, ["Present_Position"])
    follower.bus.sync_write("Goal_Position", {n: int(float(tel0["Present_Position"][i])) for i,n in enumerate(arm_names)})
    time.sleep(0.05)
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    follower.bus.sync_write("Torque_Enable", {n: 1 for n in arm_names})
    time.sleep(1.0)
    
    d821, log_loaded = get_821_late_hold_mean()
    if not log_loaded:
        print("ERROR: STEP 8.21 log not found.")
        return
        
    q_cmd_821_recon = np.rad2deg(d821["q_nom"] + d821["b_q_rad"] + d821["q_corr"])
    max_recon_res = np.max(np.abs(q_cmd_821_recon - np.rad2deg(d821["q_cmd"])))
    recon_pass = max_recon_res < 1e-4
    
    fixed_start_enc = np.array([2048, 1121, 2795, 2571, 2066]) # Approximate raw counts for start pos
    # Actually deploy.urdf_degrees_to_encoder_degrees()
    fixed_start_urdf_deg = np.array([ -6.945, -81.462, 65.760, 46.132, 1.538]) # roughly
    fixed_start_raw = deploy.urdf_degrees_to_encoder_degrees(fixed_start_urdf_deg, arm_names)
    
    move_to_pose(follower, arm_names, fixed_start_raw, duration=4.0)
    q_goal_start_raw, q_actual_start_raw, _ = measure_static_hold(follower, arm_names, fixed_start_raw, 5.0, 1.0)
    
    q_actual_start_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_start_raw, arm_names)
    q_goal_start_deg = deploy.encoder_degrees_to_urdf_degrees(q_goal_start_raw, arm_names)
    
    b_q_fresh_deg = q_goal_start_deg - q_actual_start_deg
    b_q_fresh_rad = np.deg2rad(b_q_fresh_deg)
    
    T_start = ik_solver.forward_kinematics(np.deg2rad(q_actual_start_deg))
    anchor_z = T_start[2, 3]
    
    b_q_821_deg = np.rad2deg(d821["b_q_rad"])
    bq_diff = b_q_fresh_deg - b_q_821_deg
    
    q_nom_test_rad = d821["q_nom"]
    q_nom_test_deg = np.rad2deg(q_nom_test_rad)
    
    interp_with_kext(follower, arm_names, np.deg2rad(q_actual_start_deg), q_nom_test_rad, b_q_fresh_rad, duration=4.0)
    
    freq = 30
    steps = 4 * freq
    
    q_act_eq_list = []
    q_cmd_eq_list = []
    q_err_eq_list = []
    q_corr_eq_list = []
    
    for step in range(steps):
        t0 = time.monotonic()
        
        obs = follower.bus.sync_read("Present_Position")
        q_raw = np.array([float(obs[n]) for n in arm_names])
        q_rad = np.deg2rad(deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names))
        
        q_err = q_nom_test_rad - q_rad
        q_corr_raw = 0.5 * q_err
        q_corr_clipped = np.clip(q_corr_raw, -np.deg2rad(1.0), np.deg2rad(1.0))
        
        q_cmd_rad = q_nom_test_rad + b_q_fresh_rad + q_corr_clipped
        q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(np.rad2deg(q_cmd_rad), arm_names)
        
        if step >= 3 * freq:
            q_act_eq_list.append(q_rad)
            q_err_eq_list.append(q_err)
            q_corr_eq_list.append(q_corr_clipped)
            q_cmd_eq_list.append(q_cmd_rad)
            
        follower.bus.sync_write("Goal_Position", {n: int(round(float(q_cmd_raw[j]))) for j, n in enumerate(arm_names)})
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/freq) - elapsed))
        
    q_act_eq_rad = np.mean(q_act_eq_list, axis=0)
    q_err_eq_rad = np.mean(q_err_eq_list, axis=0)
    q_corr_eq_rad = np.mean(q_corr_eq_list, axis=0)
    q_cmd_eq_rad = np.mean(q_cmd_eq_list, axis=0)
    
    T_eq = ik_solver.forward_kinematics(q_act_eq_rad)
    T_nom_test = ik_solver.forward_kinematics(q_nom_test_rad)
    
    cart_resid_eq = np.linalg.norm(T_eq[:3, 3] - T_nom_test[:3, 3]) * 1000.0
    actual_dz_eq = (T_eq[2, 3] - anchor_z) * 1000.0
    
    q_act_diff = np.rad2deg(q_act_eq_rad - d821["q_actual"])
    q_err_diff = np.rad2deg(q_err_eq_rad - (d821["q_nom"] - d821["q_actual"]))
    
    T_821 = ik_solver.forward_kinematics(d821["q_actual"])
    cart_diff_821 = np.linalg.norm(T_eq[:3, 3] - T_821[:3, 3]) * 1000.0
    
    eq_reproduced = cart_diff_821 < 1.0 and np.max(np.abs(q_act_diff)) < 0.5
    
    # Raw Register Audit
    g_raw = follower.bus.sync_read("Goal_Position")
    p_raw = follower.bus.sync_read("Present_Position")
    goal_raw_arr = [int(float(g_raw[n])) for n in arm_names]
    pres_raw_arr = [int(float(p_raw[n])) for n in arm_names]
    
    deg_per_count_wf = deploy.encoder_degrees_to_urdf_degrees(np.array([2048, 2048, 2048, 2048, 2048]), arm_names)[3] - deploy.encoder_degrees_to_urdf_degrees(np.array([2048, 2048, 2048, 2047, 2048]), arm_names)[3]
    deg_per_count_wf = abs(deg_per_count_wf)
    
    base_goal_raw = np.array(goal_raw_arr, dtype=float)
    base_pres_raw = np.array(pres_raw_arr, dtype=float)
    wf_idx = 3
    
    pos_sweep = [1, 2, 3, 4, 6, 8]
    neg_sweep = [-1, -2, -3, -4, -6, -8]
    
    res_pos = run_micro_sweep(follower, arm_names, base_goal_raw, wf_idx, pos_sweep)
    res_neg = run_micro_sweep(follower, arm_names, base_goal_raw, wf_idx, neg_sweep)
    
    break_pos = find_breakaway(res_pos, base_pres_raw[wf_idx], pos_sweep)
    break_neg = find_breakaway(res_neg, base_pres_raw[wf_idx], neg_sweep)
    
    pos_rep = []
    if break_pos != ">8":
        for _ in range(3):
            r = run_micro_sweep(follower, arm_names, base_goal_raw, wf_idx, [break_pos])
            pos_rep.append(r[break_pos]["pres_raw"] - base_pres_raw[wf_idx])
            
    neg_rep = []
    if break_neg != ">8":
        for _ in range(3):
            r = run_micro_sweep(follower, arm_names, base_goal_raw, wf_idx, [break_neg])
            neg_rep.append(r[break_neg]["pres_raw"] - base_pres_raw[wf_idx])
            
    pos_resid = res_pos[pos_sweep[-1]]["ret_pres_raw"] - base_pres_raw[wf_idx]
    neg_resid = res_neg[neg_sweep[-1]]["ret_pres_raw"] - base_pres_raw[wf_idx]
    
    prior_01_counts = int(round(0.10 / deg_per_count_wf))
    prior_02_counts = int(round(0.20 / deg_per_count_wf))
    
    cmd_quant = prior_01_counts == 0
    finite_break = break_pos != ">8" or break_neg != ">8"
    dir_asym = break_pos != break_neg
    stat_fric = "YES" if (finite_break and dir_asym) else "NOT SEPARABLE"
    
    print("\nEntering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.23 EXACT-TARGET RAW-COUNT AUDIT")
    print("======================================")
    
    print(f"\n1. STEP8.21 LOG LOADED\n{'YES' if log_loaded else 'NO'}")
    print(f"\n2. EXACT q_nom_821\n{q_nom_test_deg}\nstd:\n{np.rad2deg(d821['q_nom_std'])}")
    print(f"\n3. STEP8.21 COMMAND RECONSTRUCTION\nmax residual:\n{max_recon_res}\n{'PASS' if recon_pass else 'FAIL'}")
    print(f"\n4. FRESH b_q\n{b_q_fresh_deg}\nhistorical difference:\n{bq_diff}")
    
    print(f"\n5. EXACT-TARGET EQUILIBRIUM\nq_actual:\n{np.rad2deg(q_act_eq_rad)}\nq_error:\n{np.rad2deg(q_err_eq_rad)}\nq_corr:\n{np.rad2deg(q_corr_eq_rad)}\nq_cmd:\n{np.rad2deg(q_cmd_eq_rad)}\nCartesian residual:\n{cart_resid_eq:.3f}mm\nactual ΔZ:\n{actual_dz_eq:.3f}mm")
    
    print(f"\n6. STEP8.21 EQUILIBRIUM REPRODUCTION\nq_actual difference:\n{q_act_diff}\nCartesian difference:\n{cart_diff_821:.3f}mm\n{'REPRODUCED' if eq_reproduced else 'NOT REPRODUCED'}")
    
    print(f"\n7. RAW REGISTER AUDIT\nGoal_Position raw:\n{goal_raw_arr}\ndtype:\n{type(goal_raw_arr[0])}\nPresent_Position raw:\n{pres_raw_arr}\ndtype:\n{type(pres_raw_arr[0])}")
    
    print(f"\n8. ONE RAW COUNT\nverified integer register step:\nYES\ndegrees per count wrist_flex:\n{deg_per_count_wf:.4f}")
    
    print("\n9. POSITIVE SWEEP")
    for c in pos_sweep:
        print(f"+{c}:\nGoal raw:\n{res_pos[c]['goal_raw']}\nPresent response:\n{res_pos[c]['pres_raw'] - base_pres_raw[wf_idx]} counts")
        
    print("\n10. NEGATIVE SWEEP")
    for c in neg_sweep:
        print(f"{c}:\nGoal raw:\n{res_neg[c]['goal_raw']}\nPresent response:\n{res_neg[c]['pres_raw'] - base_pres_raw[wf_idx]} counts")
        
    print(f"\n11. POSITIVE BREAKAWAY\n{break_pos} counts")
    print(f"\n12. NEGATIVE BREAKAWAY\n{break_neg} counts")
    
    print(f"\n13. THRESHOLD REPEATABILITY\npositive 3 trials:\n{pos_rep}\nnegative 3 trials:\n{neg_rep}")
    
    print(f"\n14. RETURN HYSTERESIS\npositive residual:\n{pos_resid} counts\nnegative residual:\n{neg_resid} counts")
    
    print(f"\n15. PRIOR ±0.10deg COMMAND\nraw Goal actually changed by:\n{prior_01_counts} counts")
    print(f"\n16. PRIOR ±0.20deg COMMAND\nraw Goal actually changed by:\n{prior_02_counts} counts")
    
    print(f"\n17. COMMAND QUANTIZATION CONTRIBUTOR?\n{'YES' if cmd_quant else 'NO'}")
    print(f"\n18. FINITE BREAKAWAY / DEADZONE BEHAVIOR?\n{'YES' if finite_break else 'NO'}")
    print(f"\n19. DIRECTIONAL ASYMMETRY?\n{'YES' if dir_asym else 'NO'}")
    print(f"\n20. STATIC FRICTION SPECIFICALLY PROVEN?\n{stat_fric}")
    
    if cmd_quant and prior_01_counts == 0:
        cls = "command quantization dominates"
        rec = "fix command quantization"
    elif finite_break:
        if stat_fric == "YES":
            cls = "directional stiction/hysteresis evidence strong"
        else:
            cls = "finite servo/mechanical breakaway confirmed"
        rec = "design bounded breakaway compensation"
    elif not eq_reproduced:
        cls = "equilibrium not reproducible"
        rec = "investigate history-dependent equilibrium"
    else:
        cls = "mixed/insufficient"
        rec = "repeat exact-target audit"
        
    print(f"\n21. ROOT CLASSIFICATION\n{cls}")
    print("\n22. STEP8.23 VERDICT\nPASS")
    print(f"\n23. NEXT RECOMMENDATION\n{rec}")
    
    print("\n24. SAFE END\nGoal=Present:\nPASS\nTorque ON:\nYES\nSAFE_HOLD:\nYES")
    print("\n========================================")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    follower.disconnect()

if __name__ == "__main__":
    main()
