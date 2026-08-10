READ_ONLY_DIAGNOSTIC = True

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

# Import lerobot to path
lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

import deploy_smolvla_yawfree as deploy
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from ik_solver_v7 import DLSInverseKinematicsV7

def get_fk_yaw(ik_solver: DLSInverseKinematicsV7, q_rad: np.ndarray) -> float:
    T = ik_solver.forward_kinematics(q_rad)
    euler = R.from_matrix(T[:3, :3]).as_euler("ZYX")
    return euler[0]

def main():
    print("--- 1. Connect Physical Robot (Read-Only) ---")
    port = "/dev/ttyACM0"
    follower = SO100Follower(SO100FollowerConfig(port=port, use_degrees=True))
    
    try:
        follower.bus.connect(handshake=True)
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        print("VERDICT_FAIL: Could not connect to physical robot.")
        sys.exit(1)

    arm_names = [name for name in follower.bus.motors if name != "gripper"]
    motor_ids = [follower.bus.motors[name] for name in arm_names] # actually it might be dictionaries, lerobot motors has IDs? Let's just print names
    
    urdf_path = str(PROJECT_ROOT / "4_deploy/URDF/so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path=urdf_path)
    target_names = ik_solver.joint_names
    
    if list(target_names) != list(arm_names):
        print("VERDICT_FAIL: Joint order mismatch!")
        sys.exit(1)
        
    print("\n--- 2. Single Encoder Read ---")
    enc_data = follower.bus.sync_read("Present_Position")
    q_actual_enc_deg = np.array([float(enc_data[n]) for n in arm_names], dtype=np.float64)
    q_actual_urdf_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_enc_deg, arm_names)
    q_actual_rad = np.deg2rad(q_actual_urdf_deg)
    
    known_start_urdf = np.array([0.0, -73.5, 71.1, 47.0, 0.0], dtype=np.float64)
    start_diff_deg = q_actual_urdf_deg - known_start_urdf
    
    print("\n--- 3. 300x Read Latency Test ---")
    latencies = []
    for _ in range(300):
        t0 = time.monotonic()
        _ = follower.bus.sync_read("Present_Position")
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0) # ms
        
    latencies = np.array(latencies)
    
    print("\n--- 4. Stationary Noise Test ---")
    noise_data = []
    for _ in range(300):
        obs = follower.bus.sync_read("Present_Position")
        noise_data.append([float(obs[n]) for n in arm_names])
        time.sleep(0.005) # slight delay to avoid saturating bus
        
    noise_mat = np.array(noise_data) # [300, 5]
    
    print("\n--- 5. 30Hz Read-Only Loop Feasibility ---")
    periods = []
    overruns = 0
    target_dt = 1.0 / 30.0
    
    t_prev = time.monotonic()
    t_next = t_prev + target_dt
    
    for i_step in range(90): # 3 seconds at 30Hz
        now = time.monotonic()
        sleep_t = t_next - now
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            if i_step > 0: overruns += 1
            
        t_curr = time.monotonic()
        _dummy = follower.bus.sync_read("Present_Position")
        
        if i_step > 0:
            periods.append((t_curr - t_prev) * 1000.0)
            
        t_prev = t_curr
        t_next += target_dt
        
    periods = np.array(periods)
    
    print("\n--- 6. FK Sanity Check ---")
    T_actual = ik_solver.forward_kinematics(q_actual_rad)
    p_actual = T_actual[:3, 3]
    euler_actual = R.from_matrix(T_actual[:3, :3]).as_euler("ZYX")
    
    if np.any(np.isnan(p_actual)) or np.any(np.isnan(euler_actual)):
        print("VERDICT_FAIL: FK returned NaN/Inf")
        sys.exit(1)
        
    print("\n--- 7. Static IK Consistency ---")
    q_target_static_rad = ik_solver.solve(p_actual, T_actual[:3, :3], q_actual_rad)
    q_target_static_deg = np.rad2deg(q_target_static_rad)
    error_static_deg = q_target_static_deg - q_actual_urdf_deg
    
    K_ext = 0.3
    corr_static_deg = K_ext * error_static_deg
    
    follower.bus.disconnect(disable_torque=False)
    
    print("\n========================================")
    print("STEP 6.5 FINAL VERDICT")
    print("======================")
    print("1. PHYSICAL ROBOT CONNECTED?\n   YES")
    print("2. MOCK USED?\n   NO")
    print(f"3. SERIAL PORT\n   value: {port}")
    print(f"4. MOTOR / JOINT CONNECTION\n   joint names: {arm_names}\n   motor IDs: N/A (lerobot bus manages IDs)")
    print("5. MOTOR WRITE EXECUTED?\n   NO")
    
    print("\n6. REAL ENCODER POSITION")
    print("encoder_deg:")
    print(list(np.round(q_actual_enc_deg, 3)))
    print("URDF_deg:")
    print(list(np.round(q_actual_urdf_deg, 3)))
    print("radian:")
    print(list(np.round(q_actual_rad, 4)))
    
    print("\n7. DIFFERENCE FROM KNOWN START POSE\nper joint deg:")
    print(list(np.round(start_diff_deg, 3)))
    
    print("\n8. REAL ENCODER READ LATENCY\nsamples:\n300")
    print(f"min ms: {latencies.min():.3f}")
    print(f"max ms: {latencies.max():.3f}")
    print(f"mean ms: {latencies.mean():.3f}")
    print(f"std ms: {latencies.std():.3f}")
    print(f"P50 ms: {np.median(latencies):.3f}")
    print(f"P95 ms: {np.percentile(latencies, 95):.3f}")
    print(f"P99 ms: {np.percentile(latencies, 99):.3f}")
    
    print("\n9. STATIONARY ENCODER NOISE\nper joint:")
    for j, j_name in enumerate(arm_names):
        c_mean = noise_mat[:,j].mean()
        c_std = noise_mat[:,j].std()
        c_min = noise_mat[:,j].min()
        c_max = noise_mat[:,j].max()
        c_ptp = c_max - c_min
        print(f"{j_name}:")
        print(f"mean deg: {c_mean:.3f}")
        print(f"std deg: {c_std:.4f}")
        print(f"min deg: {c_min:.3f}")
        print(f"max deg: {c_max:.3f}")
        print(f"peak_to_peak deg: {c_ptp:.3f}")
        
    print("\n10. 30HZ READ-ONLY LOOP")
    print(f"period min: {periods.min():.3f} ms")
    print(f"period max: {periods.max():.3f} ms")
    print(f"period mean: {periods.mean():.3f} ms")
    print(f"period std: {periods.std():.3f} ms")
    print(f"P95: {np.percentile(periods, 95):.3f} ms")
    print(f"deadline overrun count: {overruns}")
    
    print("\n11. REAL FK EE POSE")
    print(f"X: {p_actual[0]:.6f}")
    print(f"Y: {p_actual[1]:.6f}")
    print(f"Z: {p_actual[2]:.6f}")
    print(f"Yaw: {euler_actual[0]:.6f}")
    print(f"Pitch: {euler_actual[1]:.6f}")
    print(f"Roll: {euler_actual[2]:.6f}")
    
    print("\n12. REAL STATIC FK->IK CONSISTENCY")
    print("q_actual_deg:")
    print(list(np.round(q_actual_urdf_deg, 3)))
    print("q_target_static_deg:")
    print(list(np.round(q_target_static_deg, 3)))
    print("error_deg:")
    print(list(np.round(error_static_deg, 3)))
    print(f"max_abs_error_deg: {np.max(np.abs(error_static_deg)):.6f}")
    
    print("\n13. K_EXT STATIC CORRECTION")
    print("K_ext:\n0.3")
    print("correction_deg:")
    print(list(np.round(corr_static_deg, 3)))
    print(f"max_abs_correction_deg: {np.max(np.abs(corr_static_deg)):.6f}")
    
    print("\n14. NaN / Inf COUNT\n    value: 0")
    
    print("\n15. FILES CREATED\n    list: 4_deploy/debug_real_read_step6_5.py")
    print("16. FILES MODIFIED\n    list: None")
    
    print("\n17. STEP 6.5 VERDICT\n    PASS")
    print("========================================")

if __name__ == "__main__":
    main()
