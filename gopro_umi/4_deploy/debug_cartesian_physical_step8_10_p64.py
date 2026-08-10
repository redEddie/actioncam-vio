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

def move_to_fixed_start(follower, arm_names, target):
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

    # Hold for 3s
    print("Hold at start pose for 3.0s...")
    target_dict = {n: float(target[j]) for j, n in enumerate(arm_names)}
    for _ in range(90):
        t0 = time.monotonic()
        follower.bus.sync_write("Goal_Position", target_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    final_obs = []
    for _ in range(15): # last 0.5s
        t0 = time.monotonic()
        obs = follower.bus.sync_read("Present_Position")
        final_obs.append(np.array([float(obs[n]) for n in arm_names]))
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
    
    return np.mean(final_obs, axis=0)

def generate_trajectory(delta_z):
    # 0-1s: Move
    # 1-2s: Peak Hold
    # 2-3s: Return
    # 3-4s: Final Hold
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

def run_preflight(ik_solver, traj_pos, T_start, q_actual_start_rad):
    print("Preflight check...")
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
            
        for val, bounds in zip(q_sol, ik_solver.safe_limits.values()):
            if val < bounds[0] or val > bounds[1]:
                print(f"Preflight FAIL: Safe violation {val} not in {bounds}")
                return False
                
        q_curr = q_sol
        
    print("Preflight PASS.")
    return True

def run_phase(follower, ik_solver, arm_names, delta_z, q_actual_start_raw, phase_name):
    print(f"\n--- Running PHASE {phase_name} (Z {delta_z*1000:+.0f} mm) ---")
    
    q_urdf_start = deploy.encoder_degrees_to_urdf_degrees(q_actual_start_raw, arm_names)
    q_rad_start = np.deg2rad(q_urdf_start)
    T_start = ik_solver.forward_kinematics(q_rad_start)
    p0_actual = T_start[:3, 3]
    euler_start = R.from_matrix(T_start[:3, :3]).as_euler("ZYX")
    
    print(f"p0_actual: {p0_actual}")
    print(f"euler_start: {euler_start}")
    
    traj_pos, traj_times = generate_trajectory(delta_z)
    
    if not run_preflight(ik_solver, traj_pos, T_start, q_rad_start):
        return False
        
    # Telemetry before
    tel_before = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    print("Temperature before:", tel_before.get("Present_Temperature"))
    print("Load before:", tel_before.get("Present_Load"))
    
    logs = defaultdict(list)
    start_time = time.perf_counter()
    q_nom_prev = q_rad_start.copy()
    
    clamp_deg = 1.0
    k_ext = 0.3
    clamp_count = 0
    total_ticks = 0
    
    control_freq = 30
    duration = 4.0
    steps = int(duration * control_freq)
    
    for step in range(steps):
        t0 = time.perf_counter()
        current_t = (step + 1) / control_freq
        
        idx = np.searchsorted(traj_times, current_t, side="left")
        if idx >= len(traj_times):
            idx = len(traj_times) - 1
            
        dz = traj_pos[idx]
        pos_target = p0_actual + dz
        
        # Read actual
        obs = follower.bus.sync_read("Present_Position")
        q_raw = np.array([float(obs[n]) for n in arm_names])
        q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names)
        q_rad = np.deg2rad(q_urdf)
        
        # Emergency Check
        if np.any(np.abs(q_raw - q_actual_start_raw) > 10.0):
            print("EMERGENCY JOINT GATE: >10deg deviation!")
            return False
            
        T_act = ik_solver.forward_kinematics(q_rad)
        p_act = T_act[:3, 3]
        if np.linalg.norm(p_act - p0_actual) > 0.030: # 30mm
            print("EMERGENCY CARTESIAN GATE: >30mm deviation!")
            return False
            
        # Yaw Free
        euler_act = R.from_matrix(T_act[:3, :3]).as_euler("ZYX")
        actual_yaw = euler_act[0]
        rot_target = R.from_euler("ZYX", [actual_yaw, euler_start[1], euler_start[2]]).as_matrix()
        
        # IK
        q_nom = ik_solver.solve(pos_target, rot_target, current_joints=q_nom_prev)
        if q_nom is None:
            print("IK FAIL during runtime!")
            return False
            
        q_nom_prev = q_nom.copy()
        
        # K_ext Correction
        q_err = q_nom - q_rad
        q_corr_raw_deg = np.rad2deg(q_err) * k_ext
        q_corr_applied_deg = np.clip(q_corr_raw_deg, -clamp_deg, clamp_deg)
        
        if np.any(np.abs(q_corr_raw_deg) > clamp_deg):
            clamp_count += 1
            
        q_cmd_rad = q_nom + np.deg2rad(q_corr_applied_deg)
        q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
        q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
        
        # Write Command
        cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        
        # Logging
        logs["time"].append(current_t)
        logs["p_nom"].append(pos_target)
        logs["p_act"].append(p_act)
        logs["euler_act"].append(euler_act)
        logs["q_nom_rad"].append(q_nom)
        logs["q_act_rad"].append(q_rad)
        logs["q_corr_raw_deg"].append(q_corr_raw_deg)
        logs["q_corr_applied_deg"].append(q_corr_applied_deg)
        logs["q_act_raw"].append(q_raw)
        
        tel = read_telemetry(follower, arm_names, ["Present_Load", "Present_Velocity", "Present_Temperature"])
        if "Present_Load" in tel: logs["Load"].append(tel["Present_Load"])
        if "Present_Temperature" in tel: logs["Temp"].append(tel["Present_Temperature"])
        
        total_ticks += 1
        
        elapsed = time.perf_counter() - t0
        time.sleep(max(0, (1.0/control_freq) - elapsed))

    # Metrics computation
    p_nom = np.array(logs["p_nom"])
    p_act = np.array(logs["p_act"])
    q_nom_rad = np.array(logs["q_nom_rad"])
    q_act_rad = np.array(logs["q_act_rad"])
    q_corr_raw_deg = np.array(logs["q_corr_raw_deg"])
    q_corr_applied_deg = np.array(logs["q_corr_applied_deg"])
    times = np.array(logs["time"])
    euler_act = np.array(logs["euler_act"])
    
    idx_peak = (times >= 1.5) & (times <= 2.0)
    p_peak = np.mean(p_act[idx_peak], axis=0)
    dp_peak = p_peak - p0_actual
    print(f"\nPEAK DISPLACEMENT (1.5-2.0s):")
    print(f"dX: {dp_peak[0]*1000:.3f} mm")
    print(f"dY: {dp_peak[1]*1000:.3f} mm")
    print(f"dZ: {dp_peak[2]*1000:.3f} mm")
    print(f"Z amplitude error: {abs(delta_z - dp_peak[2])*1000:.3f} mm")
    
    idx_return = (times >= 3.5) & (times <= 4.0)
    p_ret = np.mean(p_act[idx_return], axis=0)
    dp_ret = p_ret - p0_actual
    print(f"\nRETURN (3.5-4.0s):")
    print(f"dX: {dp_ret[0]*1000:.3f} mm")
    print(f"dY: {dp_ret[1]*1000:.3f} mm")
    print(f"dZ: {dp_ret[2]*1000:.3f} mm")
    print(f"Return norm: {np.linalg.norm(dp_ret)*1000:.3f} mm")
    
    rel_nom = p_nom - p0_actual
    rel_act = p_act - p0_actual
    e_cart = rel_nom - rel_act
    e_cart_mm = e_cart * 1000.0
    
    mae_cart = np.mean(np.abs(e_cart_mm), axis=0)
    norms = np.linalg.norm(e_cart_mm, axis=1)
    
    print(f"\nCARTESIAN TRACKING:")
    print(f"X MAE: {mae_cart[0]:.3f} mm")
    print(f"Y MAE: {mae_cart[1]:.3f} mm")
    print(f"Z MAE: {mae_cart[2]:.3f} mm")
    print(f"Pos norm mean: {np.mean(norms):.3f} mm")
    print(f"P95: {np.percentile(norms, 95):.3f} mm")
    print(f"Max: {np.max(norms):.3f} mm")
    
    print(f"\nJOINT TRACKING (deg):")
    q_err_deg = np.rad2deg(q_nom_rad - q_act_rad)
    print(f"Mean signed: {np.mean(q_err_deg, axis=0)}")
    print(f"MAE: {np.mean(np.abs(q_err_deg), axis=0)}")
    print(f"P95: {np.percentile(np.abs(q_err_deg), 95, axis=0)}")
    print(f"Max: {np.max(np.abs(q_err_deg), axis=0)}")
    
    print(f"\nEXTERNAL CORRECTION:")
    print(f"Raw max: {np.max(np.abs(q_corr_raw_deg), axis=0)}")
    print(f"Applied max: {np.max(np.abs(q_corr_applied_deg), axis=0)}")
    print(f"Clamp count: {clamp_count}")
    print(f"Clamp percentage: {clamp_count/total_ticks*100:.1f}%")
    
    print(f"\nORIENTATION:")
    pitch_err = np.arctan2(np.sin(euler_act[:,1] - euler_start[1]), np.cos(euler_act[:,1] - euler_start[1]))
    roll_err = np.arctan2(np.sin(euler_act[:,2] - euler_start[2]), np.cos(euler_act[:,2] - euler_start[2]))
    print(f"Pitch MAE: {np.mean(np.abs(np.rad2deg(pitch_err))):.3f} deg")
    print(f"Roll MAE: {np.mean(np.abs(np.rad2deg(roll_err))):.3f} deg")
    
    tel_after = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    print("\nTEMPERATURE AFTER:")
    print(tel_after.get("Present_Temperature"))
    
    if "Temp" in logs:
        temp_logs = np.array(logs["Temp"])
        idx_e = arm_names.index("elbow_flex")
        print(f"Elbow Temp Min: {np.min(temp_logs[:, idx_e])}, Max: {np.max(temp_logs[:, idx_e])}")
        
    return True

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    try:
        check_pid(follower, arm_names)
        
        # Thermal Pre-Gate
        tel = read_telemetry(follower, arm_names, ["Present_Temperature"])
        print(f"INITIAL TEMPERATURE: {tel.get('Present_Temperature')}")
        
        # Phase A reset
        print("\n--- Moving to FIXED TARGET for Phase A ---")
        q_act_A = move_to_fixed_start(follower, arm_names, fixed_start_enc)
        print(f"ACTUAL START A: {q_act_A}")
        print(f"RESIDUAL: {fixed_start_enc - q_act_A}")
        
        if not run_phase(follower, ik_solver, arm_names, 0.002, q_act_A, "A"):
            print("Phase A failed. Aborting Phase B.")
            return
            
        # Phase B reset
        print("\n--- Moving to FIXED TARGET for Phase B ---")
        q_act_B = move_to_fixed_start(follower, arm_names, fixed_start_enc)
        print(f"ACTUAL START B: {q_act_B}")
        
        q_urdf_A = deploy.encoder_degrees_to_urdf_degrees(q_act_A, arm_names)
        q_urdf_B = deploy.encoder_degrees_to_urdf_degrees(q_act_B, arm_names)
        p0_A = ik_solver.forward_kinematics(np.deg2rad(q_urdf_A))[:3, 3]
        p0_B = ik_solver.forward_kinematics(np.deg2rad(q_urdf_B))[:3, 3]
        
        print("\nSTART REPEATABILITY:")
        print("Joint diff:", q_act_B - q_act_A)
        print("EE dX/dY/dZ mm:", (p0_B - p0_A)*1000)
        print("Pos norm mm:", np.linalg.norm(p0_B - p0_A)*1000)
        
        if not run_phase(follower, ik_solver, arm_names, 0.005, q_act_B, "B"):
            print("Phase B failed.")
            return
            
    finally:
        print("\nSyncing Goal to Present at script end...")
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
