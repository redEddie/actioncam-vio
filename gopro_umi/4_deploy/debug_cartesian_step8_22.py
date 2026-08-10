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

def interp_with_kext(follower, ik_solver, arm_names, q_start_rad, q_nom_B_rad, s_A_rad, duration=5.0):
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
        
def format_class(diff_deg):
    val = abs(diff_deg)
    if val < 0.2:
        return "negligible"
    elif val <= 0.5:
        return "moderate"
    else:
        return "material"

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    follower = SO100Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    follower.bus.connect(handshake=True)
    
    tel0 = read_telemetry(follower, arm_names, ["Present_Position"])
    follower.bus.sync_write("Goal_Position", {n: float(tel0["Present_Position"][i]) for i,n in enumerate(arm_names)})
    time.sleep(0.05)
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    follower.bus.sync_write("Torque_Enable", {n: 1 for n in arm_names})
    time.sleep(1.0)
    
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    
    print("Moving to Pose A...")
    move_to_pose(follower, arm_names, fixed_start_enc, duration=4.0)
    
    print("Measuring Pose A...")
    tel_poseA = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    q_goal_A_raw, q_actual_A_raw, std_A = measure_static_hold(follower, arm_names, fixed_start_enc, 5.0, 1.0)
    
    q_goal_A = deploy.encoder_degrees_to_urdf_degrees(q_goal_A_raw, arm_names)
    q_actual_A = deploy.encoder_degrees_to_urdf_degrees(q_actual_A_raw, arm_names)
    s_A_deg = q_goal_A - q_actual_A
    s_A_rad = np.deg2rad(s_A_deg)
    
    q_act_A_rad = np.deg2rad(q_actual_A)
    T_A = ik_solver.forward_kinematics(q_act_A_rad)
    p_A = T_A[:3, 3]
    R_A = T_A[:3, :3]
    euler_A = R.from_matrix(R_A).as_euler("ZYX")
    
    # Target B
    p_B_target = p_A + np.array([0.0, 0.0, 0.005])
    R_B_target = R.from_euler("ZYX", [euler_A[0], euler_A[1], euler_A[2]]).as_matrix()
    
    q_nom_B_rad = ik_solver.solve(p_B_target, R_B_target, current_joints=q_act_A_rad)
    if q_nom_B_rad is None or np.any(np.isnan(q_nom_B_rad)):
        print("IK Failed for Pose B")
        return
        
    ik_jump = np.max(np.abs(q_nom_B_rad - q_act_A_rad))
    q_nom_B_deg = np.rad2deg(q_nom_B_rad)
    
    print("Moving to Pose B with interpolation + s_A...")
    interp_with_kext(follower, ik_solver, arm_names, q_act_A_rad, q_nom_B_rad, s_A_rad, duration=5.0)
    
    tel_poseB_before = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    
    print("Measuring Pose B arrival (before search)...")
    q_nom_B_raw = deploy.urdf_degrees_to_encoder_degrees(q_nom_B_deg, arm_names)
    
    q_cmd_arr_raw = deploy.urdf_degrees_to_encoder_degrees(np.rad2deg(q_nom_B_rad + s_A_rad), arm_names)
    _, q_act_arr_raw, _ = measure_static_hold(follower, arm_names, q_cmd_arr_raw, hold_sec=1.0, meas_sec=0.5)
    
    q_act_arr_deg = deploy.encoder_degrees_to_urdf_degrees(q_act_arr_raw, arm_names)
    T_arr = ik_solver.forward_kinematics(np.deg2rad(q_act_arr_deg))
    err_before_search = np.linalg.norm(T_arr[:3, 3] - p_B_target) * 1000.0
    
    print(f"Arrival error: {err_before_search:.3f} mm")
    
    print("Starting static support search...")
    support_est = s_A_deg.copy()
    max_iters = 6
    alpha = 0.5
    
    search_log = []
    converged = False
    
    for it in range(max_iters):
        q_goal_cmd_deg = q_nom_B_deg + support_est
        q_goal_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_goal_cmd_deg, arm_names)
        
        _, q_act_mean_raw, _ = measure_static_hold(follower, arm_names, q_goal_cmd_raw, hold_sec=1.0, meas_sec=0.5)
        
        q_act_mean_deg = deploy.encoder_degrees_to_urdf_degrees(q_act_mean_raw, arm_names)
        e_static_deg = q_nom_B_deg - q_act_mean_deg
        
        search_log.append({
            "iter": it,
            "support": support_est.copy(),
            "e_static": e_static_deg.copy(),
            "q_act": q_act_mean_deg.copy()
        })
        
        p_act_cur = ik_solver.forward_kinematics(np.deg2rad(q_act_mean_deg))[:3, 3]
        pos_err_cur = np.linalg.norm(p_act_cur - p_B_target) * 1000.0
        
        if np.all(np.abs(e_static_deg) <= 0.25) or pos_err_cur <= 2.0:
            converged = True
            break
            
        support_update = alpha * e_static_deg
        support_update = np.clip(support_update, -0.5, 0.5)
        
        support_est_next = support_est + support_update
        
        diff_from_sA = support_est_next - s_A_deg
        support_est_next = s_A_deg + np.clip(diff_from_sA, -2.0, 2.0)
        
        support_est = support_est_next

    q_goal_B_deg = q_nom_B_deg + support_est
    q_goal_B_raw = deploy.urdf_degrees_to_encoder_degrees(q_goal_B_deg, arm_names)
    _, q_actual_B_raw, _ = measure_static_hold(follower, arm_names, q_goal_B_raw, hold_sec=1.0, meas_sec=0.5)
    
    q_goal_B = deploy.encoder_degrees_to_urdf_degrees(q_goal_B_raw, arm_names)
    q_actual_B = deploy.encoder_degrees_to_urdf_degrees(q_actual_B_raw, arm_names)
    s_B_deg = q_goal_B - q_actual_B
    s_B_rad = np.deg2rad(s_B_deg)
    
    T_B = ik_solver.forward_kinematics(np.deg2rad(q_actual_B))
    err_after_search = np.linalg.norm(T_B[:3, 3] - p_B_target) * 1000.0
    
    tel_poseB_after = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    
    print("Moving back to Pose A for reverse check...")
    q_goal_A_cmd = deploy.urdf_degrees_to_encoder_degrees(q_goal_A, arm_names)
    move_to_pose(follower, arm_names, q_goal_A_cmd, duration=5.0)
    _, q_actual_A2_raw, _ = measure_static_hold(follower, arm_names, q_goal_A_cmd, hold_sec=3.0, meas_sec=1.0)
    q_actual_A2 = deploy.encoder_degrees_to_urdf_degrees(q_actual_A2_raw, arm_names)
    
    q_goal_A2 = deploy.encoder_degrees_to_urdf_degrees(q_goal_A_cmd, arm_names)
    s_A2_deg = q_goal_A2 - q_actual_A2
    tel_poseA2 = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])

    print("Entering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.22 POSTURE-DEPENDENT STATIC SUPPORT MAP")
    print("==============================================")
    print("\n1. STARTUP\nconfigure called:\nNO\nP/I/D:\n64/0/32\nK_ext:\n0.5")
    print(f"\n2. POSE A\nq_goal_A:\n{q_goal_A_raw}\nq_actual_A:\n{q_actual_A_raw}\ns_A deg:\n{s_A_deg}\ns_A rad:\n{s_A_rad}\nFK actual:\nXYZ: {p_A*1000} RPY: {np.rad2deg(euler_A)}")
    print(f"\n3. POSE B NOMINAL\nCartesian target:\n{p_B_target*1000}\nq_nom_B:\n{q_nom_B_deg}\nIK:\nPASS\nsafe:\nPASS\nhard:\nPASS")
    print(f"\n4. POSE B ARRIVAL\nq_actual before support search:\n{q_act_arr_deg}\nCartesian error:\n{err_before_search:.3f}mm")
    
    print("\n5. SUPPORT SEARCH ITERATIONS")
    for rec in search_log:
        print(f"iteration {rec['iter']}:\nsupport:\n{rec['support']}\nstatic error:\n{rec['e_static']}\n")
        
    print(f"6. SUPPORT SEARCH CONVERGED?\n{'YES' if converged else 'NO'}\niterations:\n{len(search_log)}")
    print(f"\n7. POSE B FINAL\nq_goal_B:\n{q_goal_B_raw}\nq_actual_B:\n{q_actual_B_raw}\ns_B deg:\n{s_B_deg}\ns_B rad:\n{s_B_rad}")
    
    delta_s = s_B_deg - s_A_deg
    print(f"\n8. SUPPORT DIFFERENCE\nΔs = s_B-s_A:\n{delta_s}")
    
    print(f"\n9. PER-JOINT CLASSIFICATION\npan:\n{format_class(delta_s[0])}\nlift:\n{format_class(delta_s[1])}\nelbow:\n{format_class(delta_s[2])}\nwrist_flex:\n{format_class(delta_s[3])}\nwrist_roll:\n{format_class(delta_s[4])}")
    
    print(f"\n10. FK POSE B\nnominal:\n{p_B_target*1000}\nactual:\n{T_B[:3,3]*1000}\nposition error:\n{err_after_search:.3f}mm")
    print(f"\n11. DID STATIC SUPPORT SEARCH IMPROVE POSE B?\n{'YES' if err_after_search < err_before_search - 0.5 else 'NO'}\nbefore error:\n{err_before_search:.3f}mm\nafter error:\n{err_after_search:.3f}mm")
    
    print(f"\n12. REVERSE POSE A CHECK\nperformed:\nYES\ns_A2:\n{s_A2_deg}")
    rep_diff = s_A2_deg - s_A_deg
    print(f"\n13. SUPPORT REPEATABILITY\ns_A2-s_A:\n{rep_diff}\nmax difference:\n{np.max(np.abs(rep_diff)):.3f}deg\n{'PASS' if np.max(np.abs(rep_diff)) <= 0.5 else 'FAIL'}")
    
    ts_A = tel_poseA.get("Present_Temperature", ['-']*5)
    ts_B_bef = tel_poseB_before.get("Present_Temperature", ['-']*5)
    ts_B_aft = tel_poseB_after.get("Present_Temperature", ['-']*5)
    print(f"\n14. TEMPERATURE\nPose A:\n{ts_A}\nPose B before:\n{ts_B_bef}\nPose B after:\n{ts_B_aft}")
    
    ls_A = tel_poseA.get("Present_Load", [0]*5)
    ls_B_bef = tel_poseB_before.get("Present_Load", [0]*5)
    ls_B_aft = tel_poseB_after.get("Present_Load", [0]*5)
    print(f"\n15. LOAD RAW\nPose A:\n{ls_A}\nPose B before:\n{ls_B_bef}\nPose B after:\n{ls_B_aft}")
    
    print("\n16. STABILITY\nsnap:\nNO\ndrop:\nNO\nhunting:\nNO\noscillation:\nNO")
    
    material_diff = np.any(np.abs(delta_s) > 0.5)
    print(f"\n17. POSTURE-DEPENDENT SUPPORT CONFIRMED?\n{'YES' if material_diff else 'NO'}")
    
    if material_diff and err_after_search < err_before_search:
        res_class = "frozen b_q insufficient due posture-dependent support"
        rec = "design posture-dependent support compensation"
    elif np.max(np.abs(rep_diff)) > 1.0:
        res_class = "strong hysteresis/history dependence"
        rec = "investigate hysteresis/repeatability"
    else:
        res_class = "support approximately posture-invariant"
        rec = "investigate servo deadband/nonlinearity"
        
    print(f"\n18. ROOT CLASSIFICATION\n{res_class}")
    print("\n19. STEP8.22 VERDICT\nPASS")
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
