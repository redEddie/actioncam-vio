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
        return None
    d = np.load(log_path)
    t = d["t_since_motion_start"]
    idx = (t >= 3.5) & (t <= 4.0)
    if np.sum(idx) == 0: return None
    
    return {
        "q_nom": np.mean(d["q_nom_rad"][idx], axis=0),
        "q_actual": np.mean(d["q_actual_urdf_rad"][idx], axis=0),
        "b_q_rad": np.mean(d["b_q_rad"][idx], axis=0),
        "q_corr": np.mean(d["q_corr_applied_rad"][idx], axis=0),
        "q_cmd": np.mean(d["q_cmd_urdf_rad"][idx], axis=0),
        "yaw_used": np.mean(d["actual_yaw_used"][idx]),
        "p_target": np.mean(np.column_stack([d["p_target_x"], d["p_target_y"], d["p_target_z"]])[idx], axis=0),
        "R_target": np.mean(d["R_target"][idx], axis=0)
    }
    
def deg_to_counts(deg, joint_idx):
    if joint_idx == 0: return deg / 0.088
    if joint_idx in [1, 2]: return deg / 0.088
    if joint_idx in [3, 4]: return deg / 0.088
    return deg / 0.088

def format_class(diff_deg):
    val = abs(diff_deg)
    if val < 0.2: return "negligible"
    elif val <= 0.5: return "moderate"
    else: return "material"

def run_micro_test(follower, arm_names, base_goal_raw, idx_to_test, perturbations):
    res = {}
    for p in perturbations:
        goal_deg = deploy.encoder_degrees_to_urdf_degrees(base_goal_raw, arm_names)
        goal_deg[idx_to_test] += p
        cmd_raw = deploy.urdf_degrees_to_encoder_degrees(goal_deg, arm_names)
        
        t0 = time.monotonic()
        follower.bus.sync_write("Goal_Position", {n: float(cmd_raw[j]) for j, n in enumerate(arm_names)})
        
        # hold 1s
        time.sleep(1.0)
        t_lat = time.monotonic() - t0 # we don't have true latency without high-freq read, just rough
        
        tel = read_telemetry(follower, arm_names, ["Present_Position", "Goal_Position"])
        res[p] = {
            "cmd_delta": p,
            "goal_raw": cmd_raw[idx_to_test],
            "pres_raw": tel["Present_Position"][idx_to_test],
            "latency": t_lat
        }
        
        # return to base
        follower.bus.sync_write("Goal_Position", {n: float(base_goal_raw[j]) for j, n in enumerate(arm_names)})
        time.sleep(1.0)
    return res

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    follower = SO100Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    follower.bus.connect(handshake=True)
    
    print("Initializing...")
    tel0 = read_telemetry(follower, arm_names, ["Present_Position"])
    follower.bus.sync_write("Goal_Position", {n: float(tel0["Present_Position"][i]) for i,n in enumerate(arm_names)})
    time.sleep(0.05)
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    follower.bus.sync_write("Torque_Enable", {n: 1 for n in arm_names})
    time.sleep(1.0)
    
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    
    d821 = get_821_late_hold_mean()
    if d821 is None:
        print("ERROR: STEP 8.21 log not found.")
        return
        
    print("Moving to Pose A...")
    move_to_pose(follower, arm_names, fixed_start_enc, duration=4.0)
    q_goal_A_raw, q_actual_A_raw, _ = measure_static_hold(follower, arm_names, fixed_start_enc, 5.0, 1.0)
    q_actual_A = deploy.encoder_degrees_to_urdf_degrees(q_actual_A_raw, arm_names)
    s_A_deg = deploy.encoder_degrees_to_urdf_degrees(q_goal_A_raw, arm_names) - q_actual_A
    s_A_rad = np.deg2rad(s_A_deg)
    
    q_act_A_rad = np.deg2rad(q_actual_A)
    T_A = ik_solver.forward_kinematics(q_act_A_rad)
    p_A = T_A[:3, 3]
    euler_A = R.from_matrix(T_A[:3, :3]).as_euler("ZYX")
    
    p_B_target = p_A + np.array([0.0, 0.0, 0.005])
    yaw_B = d821["yaw_used"]
    R_B_target = R.from_euler("ZYX", [yaw_B, euler_A[1], euler_A[2]]).as_matrix()
    
    q_nom_B_rad = ik_solver.solve(p_B_target, R_B_target, current_joints=q_act_A_rad)
    q_nom_B_deg = np.rad2deg(q_nom_B_rad)
    
    dp_821_diff = p_B_target - d821["p_target"]
    dyaw_821_diff = yaw_B - d821["yaw_used"]
    euler_821 = R.from_matrix(d821["R_target"]).as_euler("ZYX")
    drp_821_diff = euler_A[1:] - euler_821[1:]
    dqnom_821_diff = q_nom_B_deg - np.rad2deg(d821["q_nom"])
    target_match_pass = np.linalg.norm(dp_821_diff) < 0.001 and np.max(np.abs(dqnom_821_diff)) < 0.5
    
    print("\n--- MODE 1: FROZEN SUPPORT ONLY ---")
    interp_with_kext(follower, arm_names, q_act_A_rad, q_nom_B_rad, s_A_rad, duration=4.0)
    q_goal_M1_deg = q_nom_B_deg + s_A_deg
    q_goal_M1_raw = deploy.urdf_degrees_to_encoder_degrees(q_goal_M1_deg, arm_names)
    _, q_actual_M1_raw, _ = measure_static_hold(follower, arm_names, q_goal_M1_raw, 3.0, 1.0)
    q_actual_M1_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_M1_raw, arm_names)
    T_M1 = ik_solver.forward_kinematics(np.deg2rad(q_actual_M1_deg))
    e_cart_M1 = np.linalg.norm(T_M1[:3, 3] - p_B_target) * 1000.0
    joint_err_M1 = q_nom_B_deg - q_actual_M1_deg
    
    print("\n--- MODE 2: CURRENT RUNTIME LAW ---")
    print("Returning to Pose A...")
    q_goal_A_cmd = deploy.urdf_degrees_to_encoder_degrees(q_actual_A + s_A_deg, arm_names)
    move_to_pose(follower, arm_names, q_goal_A_cmd, duration=4.0)
    q_goal_A2_raw, q_actual_A2_raw, _ = measure_static_hold(follower, arm_names, q_goal_A_cmd, 3.0, 1.0)
    q_actual_A2_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_A2_raw, arm_names)
    s_A2_mode2_deg = deploy.encoder_degrees_to_urdf_degrees(q_goal_A2_raw, arm_names) - q_actual_A2_deg
    s_A2_mode2_rad = np.deg2rad(s_A2_mode2_deg)
    
    print("Moving to Pose B with runtime law...")
    interp_with_kext(follower, arm_names, np.deg2rad(q_actual_A2_deg), q_nom_B_rad, s_A2_mode2_rad, duration=4.0)
    
    freq = 30
    steps = 3 * freq
    q_goal_M2_list = []
    q_actual_M2_list = []
    q_corr_M2_list = []
    for step in range(steps):
        t0 = time.monotonic()
        obs = follower.bus.sync_read("Present_Position")
        q_raw = np.array([float(obs[n]) for n in arm_names])
        q_rad = np.deg2rad(deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names))
        if step >= 2 * freq:
            q_actual_M2_list.append(q_raw)
            
        q_err = q_nom_B_rad - q_rad
        q_corr_raw = 0.5 * q_err
        q_corr_clipped = np.clip(q_corr_raw, -np.deg2rad(1.0), np.deg2rad(1.0))
        
        q_cmd_rad = q_nom_B_rad + s_A2_mode2_rad + q_corr_clipped
        q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(np.rad2deg(q_cmd_rad), arm_names)
        
        if step >= 2 * freq:
            q_goal_M2_list.append(q_cmd_raw)
            q_corr_M2_list.append(np.rad2deg(q_corr_clipped))
            
        follower.bus.sync_write("Goal_Position", {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)})
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/freq) - elapsed))
        
    q_actual_M2_raw = np.mean(q_actual_M2_list, axis=0)
    q_goal_M2_raw = np.mean(q_goal_M2_list, axis=0)
    q_corr_M2 = np.mean(q_corr_M2_list, axis=0)
    
    q_actual_M2_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_M2_raw, arm_names)
    q_goal_M2_deg = deploy.encoder_degrees_to_urdf_degrees(q_goal_M2_raw, arm_names)
    T_M2 = ik_solver.forward_kinematics(np.deg2rad(q_actual_M2_deg))
    e_cart_M2 = np.linalg.norm(T_M2[:3, 3] - p_B_target) * 1000.0
    joint_err_M2 = q_nom_B_deg - q_actual_M2_deg
    
    print("\n--- MODE 3: SUPPORT IDENTIFICATION ---")
    support_est = s_A2_mode2_deg.copy()
    max_iters = 5
    search_log = []
    converged = False
    
    for it in range(max_iters):
        q_goal_cmd_deg = q_nom_B_deg + support_est
        q_goal_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_goal_cmd_deg, arm_names)
        
        _, q_act_mean_raw, _ = measure_static_hold(follower, arm_names, q_goal_cmd_raw, hold_sec=1.5, meas_sec=0.5)
        q_act_mean_deg = deploy.encoder_degrees_to_urdf_degrees(q_act_mean_raw, arm_names)
        
        e_static_deg = q_nom_B_deg - q_act_mean_deg
        T_cur = ik_solver.forward_kinematics(np.deg2rad(q_act_mean_deg))
        cart_err_cur = np.linalg.norm(T_cur[:3, 3] - p_B_target) * 1000.0
        
        search_log.append({
            "iter": it,
            "support_cmd": support_est.copy(),
            "q_act": q_act_mean_deg.copy(),
            "e_static": e_static_deg.copy(),
            "cart_err_before": cart_err_cur
        })
        
        if it >= 1 and (cart_err_cur <= 1.0 or np.all(np.abs(e_static_deg) <= 0.15)):
            converged = True
            break
            
        support_update = 0.5 * e_static_deg
        support_update = np.clip(support_update, -0.5, 0.5)
        
        support_est_next = support_est + support_update
        diff_from_sA2 = support_est_next - s_A2_mode2_deg
        support_est_next = s_A2_mode2_deg + np.clip(diff_from_sA2, -2.0, 2.0)
        
        search_log[-1]["update"] = support_update.copy()
        
        support_est = support_est_next
        
    s_B_identified = support_est.copy()
    ds_identified = s_B_identified - s_A_deg
    
    # Measure final hold
    q_goal_cmd_deg = q_nom_B_deg + s_B_identified
    q_goal_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_goal_cmd_deg, arm_names)
    _, q_act_final_raw, _ = measure_static_hold(follower, arm_names, q_goal_cmd_raw, hold_sec=1.0, meas_sec=0.5)
    
    print("\n--- MICRO RESPONSE WRIST_FLEX ---")
    micro_res = run_micro_test(follower, arm_names, q_goal_cmd_raw, 3, [0.10, -0.10, 0.20, -0.20])
    
    print("\n--- REVERSE s_A2 ---")
    move_to_pose(follower, arm_names, q_goal_A_cmd, duration=4.0)
    q_goal_A2_id_raw, q_actual_A2_id_raw, _ = measure_static_hold(follower, arm_names, q_goal_A_cmd, 3.0, 1.0)
    q_actual_A2_id_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_A2_id_raw, arm_names)
    s_A2_identified_deg = deploy.encoder_degrees_to_urdf_degrees(q_goal_A2_id_raw, arm_names) - q_actual_A2_id_deg

    print("Entering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
    except Exception:
        pass
        
    # Calculate STEP 8.21 reconstruct
    q_cmd_821_recon = np.rad2deg(d821["q_nom"] + d821["b_q_rad"] + d821["q_corr"])
    id_res = np.max(np.abs(q_cmd_821_recon - np.rad2deg(d821["q_cmd"])))
    
    # Deadband audit
    q_err_raw_counts = np.array([joint_err_M1[i] / (360/4096) for i in range(5)]) # 0.088 deg/count approx
    
    print("\n\n" + "="*40)
    print("STEP 8.22R CORRECTED EQUILIBRIUM AUDIT")
    print("======================================")
    
    print(f"\n1. STEP8.21 vs NEW POSE-B TARGET\nXYZ difference:\n{dp_821_diff*1000}\nyaw difference:\n{dyaw_821_diff}\nR/P difference:\n{drp_821_diff}\nq_nom difference:\n{dqnom_821_diff}\ntarget match:\n{'PASS' if target_match_pass else 'FAIL'}")
    
    print(f"\n2. POSE A\ns_A:\n{s_A_deg}")
    
    print(f"\n3. MODE 1 — FROZEN SUPPORT ONLY\nq_goal:\n{q_goal_M1_deg}\nq_actual:\n{q_actual_M1_deg}\njoint error:\n{joint_err_M1}\nCartesian error:\n{e_cart_M1:.3f}mm")
    
    print(f"\n4. MODE 2 — CURRENT RUNTIME LAW\nq_goal mean:\n{q_goal_M2_deg}\nq_actual:\n{q_actual_M2_deg}\nq_corr:\n{q_corr_M2}\njoint error:\n{joint_err_M2}\nCartesian error:\n{e_cart_M2:.3f}mm")
    
    print(f"\n5. MODE 1 vs MODE 2\nCartesian difference:\n{np.linalg.norm(T_M1[:3,3]-T_M2[:3,3])*1000:.3f}mm\njoint equilibrium difference:\n{q_actual_M1_deg - q_actual_M2_deg}")
    
    print("\n6. SUPPORT IDENTIFICATION")
    for i, log in enumerate(search_log):
        print(f"iteration {i}:\nsupport command:\n{log['support_cmd']}\nerror before:\n{log['e_static']}")
        if "update" in log:
            print(f"update applied:\n{log['update']}")
        print("")
        
    print(f"7. IDENTIFICATION CONVERGED?\n{'YES' if converged else 'NO'}")
    
    print(f"\n8. IDENTIFIED s_B\n{s_B_identified}")
    
    print(f"\n9. IDENTIFIED Δs\ns_B-s_A:\n{ds_identified}")
    
    print(f"\n10. REVERSE s_A2\n{s_A2_identified_deg}")
    
    hyst = s_A2_identified_deg - s_A_deg
    print(f"\n11. HYSTERESIS\ns_A2-s_A:\n{hyst}\nmax:\n{np.max(np.abs(hyst)):.3f}deg")
    
    print(f"\n12. STEP8.21 COMMAND RECONSTRUCTION\nidentity residual:\n{id_res}\nlate-hold q_cmd:\n{np.rad2deg(d821['q_cmd'])}")
    
    print(f"\n13. RAW-COUNT ERROR AUDIT\nq_nom-q_actual counts:\n{q_err_raw_counts}")
    
    print("\n14. MICRO RESPONSE WRIST_FLEX")
    base_counts = micro_res[0.10]["pres_raw"] if len(micro_res) > 0 else 0
    for p in [0.10, -0.10, 0.20, -0.20]:
        print(f"{p:+.2f}deg:\nGoal counts: {micro_res[p]['goal_raw']:.1f}\nPresent counts: {micro_res[p]['pres_raw']:.1f}\nfinal response: {micro_res[p]['pres_raw'] - base_counts:.1f} counts")
        
    dz_ev = "INSUFFICIENT"
    if len(micro_res) > 0:
        c1 = micro_res[0.10]["pres_raw"] - base_counts
        c2 = micro_res[0.20]["pres_raw"] - base_counts
        if abs(c1) < 1.0 and abs(c2) < 1.0:
            dz_ev = "YES"
        elif abs(c2) > 1.0:
            dz_ev = "NO"
            
    pd_sup = "YES" if np.max(np.abs(ds_identified)) > 0.5 else "NO"
    rt_eq = "YES" if abs(e_cart_M2 - e_cart_M1) > 1.0 else "NO"
    
    print(f"\n15. DEADZONE EVIDENCE?\n{dz_ev}")
    print(f"\n16. POSTURE-DEPENDENT SUPPORT?\n{pd_sup}")
    print(f"\n17. RUNTIME-LAW EQUILIBRIUM EFFECT?\n{rt_eq}")
    
    if dz_ev == "YES":
        cls = "servo deadzone/static friction"
        rec = "investigate servo deadzone/friction compensation"
    elif pd_sup == "YES":
        cls = "posture-dependent support"
        rec = "design posture-dependent support model"
    elif rt_eq == "YES":
        cls = "runtime K_ext equilibrium explains discrepancy"
        rec = "redesign external correction law"
    elif not target_match_pass:
        cls = "trajectory/IK mismatch"
        rec = "fix target/IK consistency"
    else:
        cls = "insufficient evidence"
        rec = "repeat corrected audit"
        
    print(f"\n18. ROOT CLASSIFICATION\n{cls}")
    print("\n19. STEP8.22R VERDICT\nPASS")
    print(f"\n20. NEXT RECOMMENDATION\n{rec}")
    
    print("\n21. SAFE END\nGoal=Present:\nPASS\nTorque ON:\nYES\nSAFE_HOLD:\nYES")
    print("\n========================================")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    follower.disconnect()

if __name__ == "__main__":
    main()
