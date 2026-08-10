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

def check_pid(follower, arm_names):
    p = follower.bus.sync_read("P_Coefficient")
    i = follower.bus.sync_read("I_Coefficient")
    d = follower.bus.sync_read("D_Coefficient")
    p_arr = np.array([p[n] for n in arm_names])
    i_arr = np.array([i[n] for n in arm_names])
    d_arr = np.array([d[n] for n in arm_names])
    print("PID Read-back:")
    print("P:", p_arr)
    print("I:", i_arr)
    print("D:", d_arr)
    if not (np.all(p_arr == 64) and np.all(i_arr == 0) and np.all(d_arr == 32)):
        print("FAIL: PID DOES NOT MATCH 64/0/32. ABORTING.")
        sys.exit(1)

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

def generate_trajectory(delta_z):
    # 0-1s: Move, 1-2s: Peak Hold, 2-3s: Return, 3-4s: Final Hold
    freq = 60
    total_len = 4 * freq
    traj = []
    for i in range(total_len):
        t = (i + 1) / freq
        if t <= 1.0:
            alpha = (3 * t**2) - (2 * t**3)
            z = delta_z * alpha
        elif t <= 2.0:
            z = delta_z
        elif t <= 3.0:
            norm_t = t - 2.0
            alpha = (3 * norm_t**2) - (2 * norm_t**3)
            z = delta_z * (1.0 - alpha)
        else:
            z = 0.0
        traj.append([0.0, 0.0, z])
    return np.array(traj), np.arange(1, total_len + 1) / freq

def run_preflight(ik_solver, traj_pos, T_start, q_actual_start_rad, b_q_rad):
    print("\nPreflight check for full +2mm Cartesian Trajectory...")
    q_curr = q_actual_start_rad.copy()
    for i, dz in enumerate(traj_pos):
        pos_target = T_start[:3, 3] + dz
        rot_target = T_start[:3, :3]
        
        q_sol = ik_solver.solve(pos_target, rot_target, current_joints=q_curr)
        if q_sol is None:
            print(f"Preflight FAIL: IK failed at step {i}")
            return False
            
        if np.any(np.isnan(q_sol)) or np.any(np.isinf(q_sol)):
            print(f"Preflight FAIL: NaN/Inf at step {i}")
            return False
            
        jump = np.max(np.abs(q_sol - q_curr))
        if jump > 0.5: # rad
            print(f"Preflight FAIL: Branch jump {jump} at step {i}")
            return False
            
        # q_cmd = q_nom + b_q + q_corr (Assume q_corr bounds [-1deg, 1deg])
        # We must verify safe limits on the final command
        q_corr_max = np.deg2rad(1.0)
        q_cmd_max = q_sol + b_q_rad + q_corr_max
        q_cmd_min = q_sol + b_q_rad - q_corr_max
        
        for val_max, val_min, bounds in zip(q_cmd_max, q_cmd_min, ik_solver.safe_limits.values()):
            if val_max > bounds[1] or val_min < bounds[0]:
                print(f"Preflight FAIL: Safe violation near step {i}")
                return False
                
        q_curr = q_sol
        
    print("Preflight PASS.")
    return True

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    try:
        check_pid(follower, arm_names)
        
        tel_before_pos = read_telemetry(follower, arm_names, ["Present_Temperature"])
        print("TEMPERATURE BEFORE POSITIONING:", tel_before_pos.get("Present_Temperature"))
        
        print("\n--- Moving to FIXED TARGET for Phase A ---")
        q_hold_goal_raw, q_start_actual_raw = move_to_fixed_start_and_measure(follower, arm_names, fixed_start_enc)
        
        tel_before_cart = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
        print("TEMPERATURE BEFORE CARTESIAN:", tel_before_cart.get("Present_Temperature"))
        print("LOAD BEFORE CARTESIAN:", tel_before_cart.get("Present_Load"))
        
        q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
        q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
        
        print("\nHOLD RAW GOAL:", q_hold_goal_raw)
        print("HOLD RAW PRESENT:", q_start_actual_raw)
        print("ACTUAL START URDF:", q_start_actual)
        
        b_q_deg = q_hold_goal - q_start_actual
        b_q_rad = np.deg2rad(b_q_deg)
        print("\nFRESH STATIC SUPPORT BIAS:")
        print("deg:", b_q_deg)
        print("rad:", b_q_rad)
        
        ref_b_q = np.array([0.264, 0.615, 2.286, 0.703, 0.352])
        print("difference vs STEP8.11 bias (deg):", b_q_deg - ref_b_q)
        
        q_start_actual_rad = np.deg2rad(q_start_actual)
        T0 = ik_solver.forward_kinematics(q_start_actual_rad)
        p0 = T0[:3, 3]
        euler_start = R.from_matrix(T0[:3, :3]).as_euler("ZYX")
        
        print("\nCARTESIAN ANCHOR FK:")
        print("X:", p0[0], "Y:", p0[1], "Z:", p0[2])
        print("Yaw:", euler_start[0], "Pitch:", euler_start[1], "Roll:", euler_start[2])
        
        # FIRST COMMAND CONTINUITY RECHECK
        q_nom0 = q_start_actual_rad.copy()
        q_corr_raw_deg0 = np.rad2deg(q_nom0 - q_start_actual_rad) * 0.3
        q_corr_applied_deg0 = np.clip(q_corr_raw_deg0, -1.0, 1.0)
        q_cmd0_rad = q_nom0 + b_q_rad + np.deg2rad(q_corr_applied_deg0)
        q_cmd0_deg = np.rad2deg(q_cmd0_rad)
        
        jump = q_cmd0_deg - q_hold_goal
        print("\nFIRST COMMAND CONTINUITY:")
        print("q_hold_goal:", q_hold_goal)
        print("q_cmd_first:", q_cmd0_deg)
        print("jump:", jump)
        print("max jump:", np.max(np.abs(jump)))
        print("elbow jump:", jump[2])
        
        if np.max(np.abs(jump)) > 0.5:
            print("FAIL: FIRST COMMAND JUMP > 0.5 DEG")
            return
            
        delta_z = 0.002
        traj_pos, traj_times = generate_trajectory(delta_z)
        
        if not run_preflight(ik_solver, traj_pos, T0, q_start_actual_rad, b_q_rad):
            print("PREFLIGHT FAIL. Aborting.")
            return
            
        print("\n--- Running +2mm CARTESIAN TRACKING ---")
        logs = defaultdict(list)
        q_nom_prev = q_start_actual_rad.copy()
        clamp_deg = 1.0
        k_ext = 0.3
        clamp_count = 0
        total_ticks = 0
        
        control_freq = 30
        duration = 4.0
        steps = int(duration * control_freq)
        
        start_time = time.perf_counter()
        timing_logs = []
        
        for step in range(steps):
            t0 = time.perf_counter()
            current_t = (step + 1) / control_freq
            
            idx = np.searchsorted(traj_times, current_t, side="left")
            if idx >= len(traj_times):
                idx = len(traj_times) - 1
                
            dz = traj_pos[idx]
            pos_target = p0 + dz
            
            obs = follower.bus.sync_read("Present_Position")
            q_raw = np.array([float(obs[n]) for n in arm_names])
            q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names)
            q_rad = np.deg2rad(q_urdf)
            
            if np.any(np.abs(q_raw - q_start_actual_raw) > 10.0):
                print("EMERGENCY JOINT GATE ABORT!")
                return
                
            T_act = ik_solver.forward_kinematics(q_rad)
            p_act = T_act[:3, 3]
            if np.linalg.norm(p_act - p0) > 0.030:
                print("EMERGENCY CARTESIAN GATE ABORT!")
                return
                
            euler_act = R.from_matrix(T_act[:3, :3]).as_euler("ZYX")
            actual_yaw = euler_act[0]
            rot_target = R.from_euler("ZYX", [actual_yaw, euler_start[1], euler_start[2]]).as_matrix()
            
            q_nom = ik_solver.solve(pos_target, rot_target, current_joints=q_nom_prev)
            q_nom_prev = q_nom.copy()
            
            q_error = q_nom - q_rad
            q_corr_raw_deg = np.rad2deg(q_error) * k_ext
            q_corr_applied_deg = np.clip(q_corr_raw_deg, -clamp_deg, clamp_deg)
            if np.any(np.abs(q_corr_raw_deg) > clamp_deg):
                clamp_count += 1
                
            q_cmd_rad = q_nom + b_q_rad + np.deg2rad(q_corr_applied_deg)
            q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
            q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
            
            cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
            follower.bus.sync_write("Goal_Position", cmd_dict)
            
            logs["t"].append(current_t)
            logs["p_nom"].append(pos_target)
            logs["p_act"].append(p_act)
            logs["euler_act"].append(euler_act)
            logs["q_nom"].append(q_nom)
            logs["q_cmd"].append(q_cmd_rad)
            logs["q_actual"].append(q_rad)
            logs["q_error"].append(q_error)
            logs["q_corr_raw"].append(q_corr_raw_deg)
            logs["q_corr_applied"].append(q_corr_applied_deg)
            
            tel = read_telemetry(follower, arm_names, ["Present_Load", "Present_Temperature"])
            if "Present_Load" in tel: logs["Load"].append(tel["Present_Load"])
            if "Present_Temperature" in tel: logs["Temp"].append(tel["Present_Temperature"])
            
            total_ticks += 1
            
            elapsed = time.perf_counter() - t0
            timing_logs.append(elapsed)
            time.sleep(max(0, (1.0/control_freq) - elapsed))

        # Metrics computation
        p_nom = np.array(logs["p_nom"])
        p_act = np.array(logs["p_act"])
        times = np.array(logs["t"])
        
        idx_peak = (times >= 1.5) & (times <= 2.0)
        p_peak = np.mean(p_act[idx_peak], axis=0)
        dp_peak = p_peak - p0
        print(f"\nPEAK ACTUAL DISPLACEMENT:")
        print(f"dX: {dp_peak[0]*1000:.3f} mm")
        print(f"dY: {dp_peak[1]*1000:.3f} mm")
        print(f"dZ: {dp_peak[2]*1000:.3f} mm")
        print(f"Z amplitude error: {abs(0.002 - dp_peak[2])*1000:.3f} mm")
        
        rel_nom = p_nom - p0
        rel_act = p_act - p0
        e_cart = rel_nom - rel_act
        e_cart_mm = e_cart * 1000.0
        mae_cart = np.mean(np.abs(e_cart_mm), axis=0)
        norms = np.linalg.norm(e_cart_mm, axis=1)
        
        print(f"\nCARTESIAN TRACKING:")
        print(f"X MAE: {mae_cart[0]:.3f} mm")
        print(f"Y MAE: {mae_cart[1]:.3f} mm")
        print(f"Z MAE: {mae_cart[2]:.3f} mm")
        print(f"position norm mean: {np.mean(norms):.3f} mm")
        print(f"P95: {np.percentile(norms, 95):.3f} mm")
        print(f"max: {np.max(norms):.3f} mm")
        
        idx_ret = (times >= 3.5) & (times <= 4.0)
        p_ret = np.mean(p_act[idx_ret], axis=0)
        dp_ret = p_ret - p0
        print(f"\nFINAL RETURN:")
        print(f"dX: {dp_ret[0]*1000:.3f} mm")
        print(f"dY: {dp_ret[1]*1000:.3f} mm")
        print(f"dZ: {dp_ret[2]*1000:.3f} mm")
        print(f"return norm: {np.linalg.norm(dp_ret)*1000:.3f} mm")
        
        q_err_deg = np.rad2deg(np.array(logs["q_error"]))
        print("\nJOINT TRACKING ERROR q_nom-q_actual (deg):")
        print(f"mean signed: {np.mean(q_err_deg, axis=0)}")
        print(f"MAE: {np.mean(np.abs(q_err_deg), axis=0)}")
        print(f"P95: {np.percentile(np.abs(q_err_deg), 95, axis=0)}")
        print(f"max: {np.max(np.abs(q_err_deg), axis=0)}")
        
        def print_rep(idx, name):
            print(f"\n{name}:")
            print("q_nom:", np.rad2deg(logs["q_nom"][idx]))
            print("b_q:", b_q_deg)
            print("q_corr:", logs["q_corr_applied"][idx])
            print("q_cmd:", np.rad2deg(logs["q_cmd"][idx]))
            print("q_actual:", np.rad2deg(logs["q_actual"][idx]))
            print("q_cmd-q_actual:", np.rad2deg(logs["q_cmd"][idx]) - np.rad2deg(logs["q_actual"][idx]))
            
        print_rep(0, "REPRESENTATIVE START VALUES (t≈0)")
        idx_p = np.argmin(np.abs(times - 1.75))
        print_rep(idx_p, "REPRESENTATIVE PEAK VALUES (t≈1.75)")
        idx_f = np.argmin(np.abs(times - 3.75))
        print_rep(idx_f, "REPRESENTATIVE FINAL VALUES (t≈3.75)")
        
        print("\nEXTERNAL CORRECTION (deg):")
        q_corr_raw = np.array(logs["q_corr_raw"])
        q_corr_app = np.array(logs["q_corr_applied"])
        print("raw max:", np.max(np.abs(q_corr_raw), axis=0))
        print("applied max:", np.max(np.abs(q_corr_app), axis=0))
        print("clamp count:", clamp_count)
        print("clamp percentage:", clamp_count/total_ticks*100)
        
        euler_act = np.array(logs["euler_act"])
        pitch_err = np.arctan2(np.sin(euler_act[:,1] - euler_start[1]), np.cos(euler_act[:,1] - euler_start[1]))
        roll_err = np.arctan2(np.sin(euler_act[:,2] - euler_start[2]), np.cos(euler_act[:,2] - euler_start[2]))
        
        print("\nORIENTATION (deg):")
        print("Roll:")
        print(f"MAE: {np.mean(np.abs(np.rad2deg(roll_err))):.3f}")
        print(f"P95: {np.percentile(np.abs(np.rad2deg(roll_err)), 95):.3f}")
        print(f"max: {np.max(np.abs(np.rad2deg(roll_err))):.3f}")
        print("Pitch:")
        print(f"MAE: {np.mean(np.abs(np.rad2deg(pitch_err))):.3f}")
        print(f"P95: {np.percentile(np.abs(np.rad2deg(pitch_err)), 95):.3f}")
        print(f"max: {np.max(np.abs(np.rad2deg(pitch_err))):.3f}")
        
        tel_after = read_telemetry(follower, arm_names, ["Present_Temperature"])
        print("\nTEMPERATURE AFTER CARTESIAN:", tel_after.get("Present_Temperature"))
        
        if "Temp" in logs:
            t_start = logs["Temp"][0][2]
            t_end = logs["Temp"][-1][2]
            print(f"Elbow Temp delta during Cartesian: {t_end - t_start:.1f} °C")
            
        if "Load" in logs:
            print("\nELBOW LOAD RAW:")
            l_arr = np.array(logs["Load"])[:, 2]
            print("before:", l_arr[0])
            print("peak:", np.mean(l_arr[idx_peak]))
            print("final:", np.mean(l_arr[idx_ret]))
            
        timing = np.array(timing_logs) * 1000
        print("\nLOOP TIMING (ms):")
        print(f"mean: {np.mean(timing):.3f}")
        print(f"P95: {np.percentile(timing, 95):.3f}")
        print(f"P99: {np.percentile(timing, 99):.3f}")
        print(f"max: {np.max(timing):.3f}")
        print(f"overruns (>33.3ms): {np.sum(timing > 33.3)}")
        
    finally:
        print("\nSyncing Goal to Present at script end...")
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
