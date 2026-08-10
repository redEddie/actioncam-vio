import sys
import os
import time
import json
import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

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

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def get_start_pose():
    json_path = DEPLOY_DIR / "yawfree_physical_start.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    # The JSON metadata claims these are URDF degrees, but the existing positioning script
    # (move_to_yawfree_physical_start.py) feeds them directly to the encoder.
    # We must treat them as raw encoder targets to reach the correct physical posture.
    enc_degs = [
        data["arm_joint_degrees"]["shoulder_pan"],
        data["arm_joint_degrees"]["shoulder_lift"],
        data["arm_joint_degrees"]["elbow_flex"],
        data["arm_joint_degrees"]["wrist_flex"],
        data["arm_joint_degrees"]["wrist_roll"]
    ]
    return np.array(enc_degs, dtype=np.float64), str(json_path)

def safe_connect(port):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    print("Executing SAFE STARTUP sequence...")
    # 1. Connect without torque
    follower.bus.connect(handshake=True)
    
    # 2. Read Present_Position
    obs = follower.bus.sync_read("Present_Position")
    
    # 3. Sync Goal_Position
    follower.bus.sync_write("Goal_Position", obs)
    time.sleep(0.05)
    
    # 4. Verify Sync
    goal_obs = follower.bus.sync_read("Goal_Position")
    max_err = 0.0
    for k in obs:
        err = abs(goal_obs[k] - obs[k])
        max_err = max(max_err, err)
    print(f"Goal vs Present Max Diff: {max_err:.2f} deg")
    
    # 5. Enable Torque
    follower.configure()
    
    # Monitor for snap
    time.sleep(0.2)
    new_obs = follower.bus.sync_read("Present_Position")
    max_snap = 0.0
    for k in obs:
        snap = abs(new_obs[k] - obs[k])
        max_snap = max(max_snap, snap)
    
    print(f"Post-enable Max Snap: {max_snap:.2f} deg")
    if max_snap > 5.0:
        print("VISIBLE SNAP DETECTED. Aborting!")
        follower.bus.disconnect(disable_torque=True)
        sys.exit(1)
        
    return follower

def move_to_start(follower, arm_names, q_start_enc_target, ik_solver):
    print("\n--- Moving to PHYSICAL START POSE ---")
    obs = follower.bus.sync_read("Present_Position")
    q_act_enc = np.array([float(obs[n]) for n in arm_names])
    
    delta = q_start_enc_target - q_act_enc
    print("q_start_target (Encoder deg):", q_start_enc_target)
    print("q_current (Encoder deg):", q_act_enc)
    print("delta_to_start (deg):", delta)
    
    q_start_urdf = deploy.encoder_degrees_to_urdf_degrees(q_start_enc_target, arm_names)
    q_start_rad = np.deg2rad(q_start_urdf)
    
    # Check limits for start pose
    for j, j_name in enumerate(arm_names):
        s_min, s_max = ik_solver.safe_limits[j_name]
        if q_start_rad[j] < s_min or q_start_rad[j] > s_max:
            print(f"SAFE LIMIT VIOLATION on {j_name}: {q_start_rad[j]:.3f} rad not in ({s_min:.3f}, {s_max:.3f})")
            sys.exit(1)
            
    # Interpolate in Encoder Space
    dur = 4.0
    steps = int(dur * 30)
    print(f"Moving smoothly over {dur}s ({steps} steps)...")
    for step in range(steps):
        t0 = time.monotonic()
        progress = (step + 1) / steps
        alpha = (1.0 - math.cos(math.pi * progress)) / 2.0
        q_cmd_enc = q_act_enc + alpha * delta
        
        cmd_dict = {n: float(q_cmd_enc[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    print("Hold at start pose for 2.0s...")
    holds = []
    for _ in range(60):
        t0 = time.monotonic()
        obs = follower.bus.sync_read("Present_Position")
        q_enc = np.array([float(obs[n]) for n in arm_names])
        holds.append(q_enc)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    # Get average of last 0.5s (15 frames)
    final_encs = np.array(holds[-15:])
    mean_enc = np.mean(final_encs, axis=0)
    q_actual_start_urdf = deploy.encoder_degrees_to_urdf_degrees(mean_enc, arm_names)
    q_actual_start_rad = np.deg2rad(q_actual_start_urdf)
    
    q_err = q_start_enc_target - mean_enc
    print(f"Start Target vs Actual Error (Encoder deg) - mean: {np.mean(np.abs(q_err)):.3f}, max abs: {np.max(np.abs(q_err)):.3f}")
    
    T_start = ik_solver.forward_kinematics(q_actual_start_rad)
    pos_start = T_start[:3, 3]
    euler_start = R.from_matrix(T_start[:3, :3]).as_euler("ZYX")
    
    print("\nACTUAL START FK (Diagnostic Anchor):")
    print(f"X: {pos_start[0]:.4f}, Y: {pos_start[1]:.4f}, Z: {pos_start[2]:.4f}")
    print(f"Yaw: {euler_start[0]:.4f}, Pitch: {euler_start[1]:.4f}, Roll: {euler_start[2]:.4f}")
    
    return q_actual_start_rad, pos_start, T_start[:3, :3]

def run_cartesian_phase(follower, ik_solver, arm_names, anchor_q_rad, anchor_pos, anchor_rot, z_amp, phase_name):
    print(f"\n--- Running PHASE {phase_name} (Z {z_amp*1000:+.0f} mm) ---")
    
    # 60 Hz trajectory points
    dur = 4.0
    dt_60 = 1.0 / 60.0
    num_pts = int(dur * 60)
    pos_traj = np.zeros((num_pts, 3))
    rot_traj = np.zeros((num_pts, 3, 3))
    
    for i in range(num_pts):
        t = i * dt_60
        pos = anchor_pos.copy()
        
        if t <= 1.0:
            alpha = (1.0 - math.cos(math.pi * t)) / 2.0
            pos[2] += z_amp * alpha
        elif t <= 2.0:
            pos[2] += z_amp
        elif t <= 3.0:
            alpha = (1.0 - math.cos(math.pi * (t - 2.0))) / 2.0
            pos[2] += z_amp * (1.0 - alpha)
            
        pos_traj[i] = pos
        rot_traj[i] = anchor_rot # Nominal rotation stays constant

    # PREFLIGHT
    print("Preflight check...")
    q_seed = anchor_q_rad.copy()
    preflight_q_noms = []
    failures = 0
    hard_violations = 0
    safe_violations = 0
    nan_infs = 0
    anchor_yaw = R.from_matrix(anchor_rot).as_euler("ZYX")[0]
    
    for i in range(0, num_pts, 2):
        euler_nom = R.from_matrix(rot_traj[i]).as_euler("ZYX")
        mod_rot = R.from_euler("ZYX", [anchor_yaw, euler_nom[1], euler_nom[2]]).as_matrix()
        
        q_out = ik_solver.solve(pos_traj[i], mod_rot, current_joints=q_seed)
        if q_out is None or not ik_solver.last_converged:
            failures += 1
            break
        if np.any(np.isnan(q_out)) or np.any(np.isinf(q_out)):
            nan_infs += 1
        for j, j_name in enumerate(arm_names):
            s_min, s_max = ik_solver.safe_limits[j_name]
            if q_out[j] < s_min or q_out[j] > s_max:
                safe_violations += 1
                
        preflight_q_noms.append(q_out)
        q_seed = q_out
        
    if failures > 0 or safe_violations > 0 or nan_infs > 0:
        print(f"PREFLIGHT FAIL: IK fails={failures}, Safe viols={safe_violations}, NaN/Inf={nan_infs}")
        return False
        
    print("Preflight PASS. Executing...")
    
    K_ext = 0.3
    clamp_rad = np.deg2rad(1.0)
    
    metrics = {
        'pos_err': [], 'z_err': [], 'rot_err_pitch': [], 'rot_err_roll': [],
        'q_nom': [], 'q_act': [], 'q_corr_raw': [], 'q_corr_app': [], 'q_cmd': [],
        'periods': [], 'lat_read': [], 'lat_ik': [], 'lat_write': [], 'lat_tot': [],
        'clamp_activations': 0, 'total_ticks': 0
    }
    
    q_seed = anchor_q_rad.copy()
    
    try:
        for tick in range(int(dur * 30)):
            t_start = time.perf_counter()
            idx = tick * 2
            
            # Read
            t_r0 = time.perf_counter()
            obs = follower.bus.sync_read("Present_Position")
            q_act_enc = np.array([float(obs[n]) for n in arm_names])
            q_act_urdf = deploy.encoder_degrees_to_urdf_degrees(q_act_enc, arm_names)
            q_act_rad = np.deg2rad(q_act_urdf)
            t_r1 = time.perf_counter()
            metrics['lat_read'].append((t_r1 - t_r0)*1000)
            
            T_act = ik_solver.forward_kinematics(q_act_rad)
            pos_act = T_act[:3, 3]
            rot_act = T_act[:3, :3]
            
            euler_act = R.from_matrix(rot_act).as_euler("ZYX")
            euler_nom = R.from_matrix(rot_traj[idx]).as_euler("ZYX")
            
            # Safety gate
            if np.linalg.norm(pos_act - anchor_pos) > 0.050: # 50mm fallback
                print("ABORT: 50mm emergency gate exceeded!")
                return False
                
            # IK
            t_ik0 = time.perf_counter()
            mod_rot = R.from_euler("ZYX", [euler_act[0], euler_nom[1], euler_nom[2]]).as_matrix()
            q_nom = ik_solver.solve(pos_traj[idx], mod_rot, current_joints=q_seed)
            if q_nom is None or not ik_solver.last_converged:
                print("IK FAILURE during execution!")
                return False
            q_seed = q_nom
            t_ik1 = time.perf_counter()
            metrics['lat_ik'].append((t_ik1 - t_ik0)*1000)
            
            # External Control
            q_err = q_nom - q_act_rad
            q_corr_raw = K_ext * q_err
            q_corr_app = np.clip(q_corr_raw, -clamp_rad, clamp_rad)
            if np.any(q_corr_raw != q_corr_app):
                metrics['clamp_activations'] += 1
            q_cmd = q_nom + q_corr_app
            
            metrics['q_nom'].append(q_nom)
            metrics['q_act'].append(q_act_rad)
            metrics['q_corr_raw'].append(q_corr_raw)
            metrics['q_corr_app'].append(q_corr_app)
            metrics['q_cmd'].append(q_cmd)
            metrics['pos_err'].append(pos_act - pos_traj[idx])
            metrics['rot_err_pitch'].append(wrap_angle(euler_nom[1] - euler_act[1]))
            metrics['rot_err_roll'].append(wrap_angle(euler_nom[2] - euler_act[2]))
            if 1.0 <= (idx/60.0) <= 2.0:
                metrics['z_err'].append(pos_act[2] - pos_traj[idx][2])
                
            # Write
            t_w0 = time.perf_counter()
            q_cmd_urdf = np.rad2deg(q_cmd)
            q_cmd_enc = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf, arm_names)
            cmd_dict = {n: float(q_cmd_enc[j]) for j, n in enumerate(arm_names)}
            follower.bus.sync_write("Goal_Position", cmd_dict)
            t_w1 = time.perf_counter()
            metrics['lat_write'].append((t_w1 - t_w0)*1000)
            
            t_end = time.perf_counter()
            metrics['lat_tot'].append((t_end - t_start)*1000)
            elapsed = t_end - t_start
            time.sleep(max(0, (1.0/30.0) - elapsed))
            metrics['periods'].append(time.perf_counter() - t_start)
            metrics['total_ticks'] += 1
            
        # Final evaluation
        print("\nTrajectory finished. Processing metrics...")
        
        pos_errs = np.array(metrics['pos_err']) * 1000
        norms = np.linalg.norm(pos_errs, axis=1)
        print(f"X MAE: {np.mean(np.abs(pos_errs[:,0])):.3f} mm, max: {np.max(np.abs(pos_errs[:,0])):.3f} mm")
        print(f"Y MAE: {np.mean(np.abs(pos_errs[:,1])):.3f} mm, max: {np.max(np.abs(pos_errs[:,1])):.3f} mm")
        print(f"Z MAE: {np.mean(np.abs(pos_errs[:,2])):.3f} mm, max: {np.max(np.abs(pos_errs[:,2])):.3f} mm")
        print(f"Pos norm mean: {np.mean(norms):.3f} mm, P95: {np.percentile(norms, 95):.3f} mm, max: {np.max(norms):.3f} mm")
        
        if metrics['z_err']:
            z_errs = np.array(metrics['z_err']) * 1000
            print(f"Steady Peak Z error (MAE): {np.mean(np.abs(z_errs)):.3f} mm")
            
        print(f"Clamp activations: {metrics['clamp_activations']} / {metrics['total_ticks']} ({metrics['clamp_activations']/metrics['total_ticks']*100:.1f}%)")
        
        print("\nFINAL RETURN METRICS:")
        final_pos = pos_errs[-1] / 1000.0 + pos_traj[-1]
        ret_err = final_pos - anchor_pos
        print(f"ΔX: {ret_err[0]*1000:.3f} mm")
        print(f"ΔY: {ret_err[1]*1000:.3f} mm")
        print(f"ΔZ: {ret_err[2]*1000:.3f} mm")
        print(f"Return Norm: {np.linalg.norm(ret_err)*1000:.3f} mm")
        
        q_final_deg = np.rad2deg(metrics['q_act'][-1])
        q_anchor_deg = np.rad2deg(anchor_q_rad)
        print(f"Joint Return Diff (deg): {q_final_deg - q_anchor_deg}")
        
        print("\nTimings:")
        print(f"Control period mean: {np.mean(metrics['periods'])*1000:.3f} ms, max: {np.max(metrics['periods'])*1000:.3f} ms")
        
        return True
        
    except Exception as e:
        print(f"Exception during run: {e}")
        # Exception Safety Policy: sync Goal_Position, do NOT disable torque!
        print("Syncing Goal_Position to Present_Position to prevent snap...")
        try:
            obs = follower.bus.sync_read("Present_Position")
            follower.bus.sync_write("Goal_Position", obs)
        except:
            pass
        return False

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    port = "/dev/ttyACM0"
    
    q_start_target, source_file = get_start_pose()
    print(f"START_POSE_SOURCE_FILE: {source_file}")
    print(f"START_POSE_REPRESENTATION: JSON numeric values treated as RAW ENCODER DEGREES (ignoring URDF label due to metadata mismatch)")
    
    # 1. Safe Startup
    follower = safe_connect(port)
    
    # 2. IK Solver (unmodified limits)
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    try:
        # 3. Move to physical start pose
        q_anchor1, pos_anchor1, rot_anchor1 = move_to_start(follower, arm_names, q_start_target, ik_solver)
        
        # 4. Phase A (+2mm)
        res_A = run_cartesian_phase(follower, ik_solver, arm_names, q_anchor1, pos_anchor1, rot_anchor1, 0.002, "A")
        if not res_A:
            print("PHASE A FAILED. Aborting.")
            return
            
        # 5. Move to physical start pose again for Phase B
        q_anchor2, pos_anchor2, rot_anchor2 = move_to_start(follower, arm_names, q_start_target, ik_solver)
        
        # Repeatability check
        joint_diff = np.rad2deg(q_anchor2 - q_anchor1)
        pos_diff = np.linalg.norm(pos_anchor2 - pos_anchor1) * 1000
        print(f"\nRepeatability from first start -> joint diff (deg): {joint_diff}")
        print(f"Repeatability EE position norm: {pos_diff:.3f} mm")
        
        # 6. Phase B (+5mm)
        res_B = run_cartesian_phase(follower, ik_solver, arm_names, q_anchor2, pos_anchor2, rot_anchor2, 0.005, "B")
        if not res_B:
            print("PHASE B FAILED.")
            
    finally:
        print("\nDisconnecting with disable_torque=False to prevent gravity drop.")
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
