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

def safe_connect(port):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    follower.bus.connect(handshake=True)
    obs = follower.bus.sync_read("Present_Position")
    follower.bus.sync_write("Goal_Position", obs)
    time.sleep(0.05)
    follower.configure()
    time.sleep(0.2)
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    time.sleep(0.2)
    return follower

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception:
            pass
    return data

def move_to_fixed_start_and_measure(follower, arm_names, target):
    obs = follower.bus.sync_read("Present_Position")
    q_act = np.array([float(obs[n]) for n in arm_names])
    delta = target - q_act
    
    dur = 4.0
    steps = int(dur * 30)
    for step in range(steps):
        t0 = time.monotonic()
        progress = (step + 1) / steps
        alpha = (3.0 * progress**2) - (2.0 * progress**3)
        q_cmd = q_act + alpha * delta
        cmd_dict = {n: float(q_cmd[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))

    print("Hold at start pose for 3.0s...")
    target_dict = {n: float(target[j]) for j, n in enumerate(arm_names)}
    for _ in range(90):
        t0 = time.monotonic()
        follower.bus.sync_write("Goal_Position", target_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    final_pres = []
    final_goal = []
    for _ in range(15): # last 0.5s
        t0 = time.monotonic()
        pres = follower.bus.sync_read("Present_Position")
        goal = follower.bus.sync_read("Goal_Position")
        final_pres.append(np.array([float(pres[n]) for n in arm_names]))
        final_goal.append(np.array([float(goal[n]) for n in arm_names]))
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
    
    return np.mean(final_goal, axis=0), np.mean(final_pres, axis=0)

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    try:
        # Check PID
        p = follower.bus.sync_read("P_Coefficient")
        p_arr = np.array([p[n] for n in arm_names])
        if not np.all(p_arr == 64):
            print("FAIL: PID DOES NOT MATCH 64/0/32. ABORTING.")
            return

        tel_before = read_telemetry(follower, arm_names, ["Present_Temperature"])
        print("INITIAL TEMPERATURE:", tel_before.get("Present_Temperature"))
        
        print("\nMoving to FIXED TARGET...")
        q_hold_goal_raw, q_start_actual_raw = move_to_fixed_start_and_measure(follower, arm_names, fixed_start_enc)
        print("HOLD RAW GOAL:", q_hold_goal_raw)
        print("HOLD RAW PRESENT:", q_start_actual_raw)
        print("HOLD GOAL-PRESENT RAW:", q_hold_goal_raw - q_start_actual_raw)
        
        q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
        q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
        print("URDF HOLD GOAL:", q_hold_goal)
        print("URDF ACTUAL START:", q_start_actual)
        
        # Calculate b_q in URDF space
        b_q_deg = q_hold_goal - q_start_actual
        b_q_rad = np.deg2rad(b_q_deg)
        print("\nSTATIC SUPPORT BIAS b_q (deg):", b_q_deg)
        print("STATIC SUPPORT BIAS b_q (rad):", b_q_rad)
        
        # Cartesian Anchor
        q_start_actual_rad = np.deg2rad(q_start_actual)
        T0 = ik_solver.forward_kinematics(q_start_actual_rad)
        p0_actual = T0[:3, 3]
        euler_start = R.from_matrix(T0[:3, :3]).as_euler("ZYX")
        print("\nCARTESIAN ANCHOR FK:")
        print("X:", p0_actual[0], "Y:", p0_actual[1], "Z:", p0_actual[2])
        print("Yaw:", euler_start[0], "Pitch:", euler_start[1], "Roll:", euler_start[2])
        
        # Offline pre-flight for FIRST ZERO-MOTION IK
        q_nom_rad = ik_solver.solve(p0_actual, T0[:3, :3], current_joints=q_start_actual_rad)
        q_nom_deg = np.rad2deg(q_nom_rad)
        print("\nFIRST ZERO-MOTION IK q_nom (deg):", q_nom_deg)
        
        q_err_rad = q_nom_rad - q_start_actual_rad
        q_corr_raw_deg = np.rad2deg(q_err_rad) * 0.3
        q_corr_applied_deg = np.clip(q_corr_raw_deg, -1.0, 1.0)
        q_corr_applied_rad = np.deg2rad(q_corr_applied_deg)
        
        # OLD cmd
        q_cmd_old_rad = q_nom_rad + q_corr_applied_rad
        q_cmd_old_deg = np.rad2deg(q_cmd_old_rad)
        old_jump_deg = q_cmd_old_deg - q_hold_goal
        print("\nOLD COMMAND LAW FIRST TICK:")
        print("q_cmd_old:", q_cmd_old_deg)
        print("old Goal jump from hold:", old_jump_deg)
        
        # NEW cmd
        q_cmd_new_rad = q_nom_rad + b_q_rad + q_corr_applied_rad
        q_cmd_new_deg = np.rad2deg(q_cmd_new_rad)
        new_jump_deg = q_cmd_new_deg - q_hold_goal
        print("\nNEW COMMAND LAW FIRST TICK:")
        print("q_cmd_new:", q_cmd_new_deg)
        print("new Goal jump from hold:", new_jump_deg)
        
        print("\nOFFLINE GOAL-CONTINUITY:")
        print("max joint jump:", np.max(np.abs(new_jump_deg)))
        print("elbow jump:", new_jump_deg[2])
        
        if np.max(np.abs(new_jump_deg)) > 0.5:
            print("FAIL: new_jump_deg > 0.5")
            return
        print("PASS")
        
        # PHYSICAL ZERO-MOTION TEST
        print("\nPHYSICAL ZERO-MOTION TEST EXECUTED: YES")
        
        logs = defaultdict(list)
        q_nom_prev = q_start_actual_rad.copy()
        
        start_time = time.perf_counter()
        steps = int(4.0 * 30)
        
        for step in range(steps):
            t0 = time.perf_counter()
            current_t = (step + 1) / 30.0
            
            obs = follower.bus.sync_read("Present_Position")
            q_raw = np.array([float(obs[n]) for n in arm_names])
            q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names)
            q_rad = np.deg2rad(q_urdf)
            
            T_act = ik_solver.forward_kinematics(q_rad)
            p_act = T_act[:3, 3]
            euler_act = R.from_matrix(T_act[:3, :3]).as_euler("ZYX")
            
            actual_yaw = euler_act[0]
            rot_target = R.from_euler("ZYX", [actual_yaw, euler_start[1], euler_start[2]]).as_matrix()
            
            q_nom = ik_solver.solve(p0_actual, rot_target, current_joints=q_nom_prev)
            q_nom_prev = q_nom.copy()
            
            q_err = q_nom - q_rad
            q_corr_raw_deg = np.rad2deg(q_err) * 0.3
            q_corr_applied_deg = np.clip(q_corr_raw_deg, -1.0, 1.0)
            
            q_cmd_rad = q_nom + b_q_rad + np.deg2rad(q_corr_applied_deg)
            q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
            q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
            
            cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
            follower.bus.sync_write("Goal_Position", cmd_dict)
            goal_read = follower.bus.sync_read("Goal_Position")
            g_raw = np.array([float(goal_read[n]) for n in arm_names])
            
            logs["t"].append(current_t)
            logs["Goal"].append(g_raw)
            logs["Present"].append(q_raw)
            logs["p_act"].append(p_act)
            
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        print("\nTRANSITION ACTUAL DATA:")
        times_to_check = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
        t_arr = np.array(logs["t"])
        for tc in times_to_check:
            idx = np.argmin(np.abs(t_arr - tc))
            print(f"{int(tc*1000)}ms:")
            print(f"Goal {logs['Goal'][idx]}")
            print(f"Present {logs['Present'][idx]}")
            
        print("\nPHYSICAL GOAL JUMP:")
        first_goal_raw = logs["Goal"][0]
        goal_jump_raw = first_goal_raw - q_hold_goal_raw
        print("per joint:", goal_jump_raw)
        print("max:", np.max(np.abs(goal_jump_raw)))
        
        print("\nACTUAL JOINT DRIFT:")
        present_arr = np.array(logs["Present"])
        drift_raw = present_arr - q_start_actual_raw
        print("per joint max:", np.max(np.abs(drift_raw), axis=0))
        
        print("\nACTUAL EE DRIFT:")
        p_act_arr = np.array(logs["p_act"])
        ee_drift = p_act_arr - p0_actual
        max_drift_idx = np.argmax(np.linalg.norm(ee_drift, axis=1))
        max_drift = ee_drift[max_drift_idx]
        print(f"dX: {max_drift[0]*1000:.3f}")
        print(f"dY: {max_drift[1]*1000:.3f}")
        print(f"dZ: {max_drift[2]*1000:.3f}")
        print("norm max:", np.linalg.norm(max_drift)*1000)
        print("final norm:", np.linalg.norm(ee_drift[-1])*1000)
        
        tel_after = read_telemetry(follower, arm_names, ["Present_Temperature"])
        print("\nTEMPERATURE AFTER:", tel_after.get("Present_Temperature"))
        
    finally:
        print("\nSyncing Goal to Present at script end...")
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
