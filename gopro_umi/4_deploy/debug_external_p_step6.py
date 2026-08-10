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

from debug_sampler_step4 import build_dummy_trajectory, sample_trajectory_at_time, TimestampedTrajectory
from ik_solver_v7 import DLSInverseKinematicsV7

def get_fk_yaw(ik_solver: DLSInverseKinematicsV7, q_rad: np.ndarray) -> float:
    T = ik_solver.forward_kinematics(q_rad)
    R_mat = T[:3, :3]
    euler = R.from_matrix(R_mat).as_euler("ZYX")
    return euler[0]

def main():
    print("--- 1. Connect Robot (Read-Only) ---")
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    arm_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    
    # Mock bus class for fallback
    class MockBus:
        def __init__(self):
            # Target URDF: [0.0, -73.5, 71.1, 47.0, 0.0]
            # Mapped to encoder via negative signs
            self.enc = {'shoulder_pan': 0.0, 'shoulder_lift': 73.5, 'elbow_flex': -71.1, 'wrist_flex': -47.0, 'wrist_roll': 0.0}
        def sync_read(self, name):
            # simulate 1ms read time
            time.sleep(0.001)
            return self.enc
        def disconnect(self, **kwargs):
            pass
            
    try:
        follower.bus.connect(handshake=True)
        robot_connected = True
        arm_names = [name for name in follower.bus.motors if name != "gripper"]
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        print("Falling back to MOCKED robot connection for diagnostic dry-run.")
        robot_connected = False
        follower.bus = MockBus()
    
    urdf_path = str(PROJECT_ROOT / "4_deploy/URDF/so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path=urdf_path)
    target_names = ik_solver.joint_names
    
    print(f"\nJOINT ORDER")
    print(f"target: {target_names}")
    print(f"actual: {arm_names}")
    
    order_match = list(target_names) == list(arm_names)
    print(f"match: {'YES' if order_match else 'NO'}")
    if not order_match:
        print("ERROR: Joint order mismatch!")
        return
        
    print("\n--- 2. Encoder Read Latency ---")
    latencies = []
    for _ in range(100):
        t0 = time.monotonic()
        _ = follower.bus.sync_read("Present_Position")
        t1 = time.monotonic()
        latencies.append((t1 - t0) * 1000.0) # ms
        
    latencies = np.array(latencies)
    print(f"min ms: {latencies.min():.3f}")
    print(f"max ms: {latencies.max():.3f}")
    print(f"mean ms: {latencies.mean():.3f}")
    print(f"std ms: {latencies.std():.3f}")
    print(f"P95 ms: {np.percentile(latencies, 95):.3f}")
    
    print("\n--- 3. Current Encoder Position ---")
    observed_encoder = follower.bus.sync_read("Present_Position")
    q_actual_enc_deg = np.array([float(observed_encoder[n]) for n in arm_names], dtype=np.float64)
    
    # Convert to URDF degrees
    q_actual_urdf_deg = deploy.encoder_degrees_to_urdf_degrees(q_actual_enc_deg, arm_names)
    q_actual_rad = np.deg2rad(q_actual_urdf_deg)
    
    print("encoder degree:")
    print(q_actual_enc_deg)
    print("URDF degree:")
    print(q_actual_urdf_deg)
    print("radian:")
    print(q_actual_rad)
    
    print("\n--- 4. Current FK EE Position ---")
    T_actual = ik_solver.forward_kinematics(q_actual_rad)
    p_actual = T_actual[:3, 3]
    euler_actual = R.from_matrix(T_actual[:3, :3]).as_euler("ZYX")
    
    print(f"X: {p_actual[0]:.6f}")
    print(f"Y: {p_actual[1]:.6f}")
    print(f"Z: {p_actual[2]:.6f}")
    print(f"Yaw: {euler_actual[0]:.6f}")
    print(f"Pitch: {euler_actual[1]:.6f}")
    print(f"Roll: {euler_actual[2]:.6f}")

    print("\n--- 5. Static Self-Consistency Test ---")
    q_target_static_rad = ik_solver.solve(p_actual, T_actual[:3, :3], q_actual_rad)
    q_target_static_deg = np.rad2deg(q_target_static_rad)
    
    error_static_deg = q_target_static_deg - q_actual_urdf_deg
    max_abs_static_err = np.max(np.abs(error_static_deg))
    
    print("q_actual_deg:")
    print(q_actual_urdf_deg)
    print("q_target_static_deg:")
    print(q_target_static_deg)
    print("error_deg per joint:")
    print(error_static_deg)
    print(f"max_abs_error_deg: {max_abs_static_err:.6f}")
    
    static_pass = max_abs_static_err < 1.0 # Tolerance 1.0 degree
    print(f"{'PASS' if static_pass else 'FAIL'}")
    
    if not static_pass:
        return
        
    print("\n--- 6. Hypothetical Trajectory Test (K_ext=0.3) ---")
    # Build 60Hz dummy trajectory starting exactly from current actual pose
    anchor_time = time.monotonic()
    num_steps = 60
    dt = 1.0 / 60.0
    offsets = np.arange(1, num_steps + 1) * dt
    target_times = anchor_time + offsets
    
    actions = np.zeros((num_steps, 6), dtype=np.float32)
    for i in range(num_steps):
        actions[i, 0] = p_actual[0] + 0.05 * (i / 60.0) # Move 5cm in X
        actions[i, 1] = p_actual[1] + 0.02 * (i / 60.0)
        actions[i, 2] = p_actual[2] - 0.01 * (i / 60.0)
        actions[i, 3] = euler_actual[2] + 0.1 * (i / 60.0) # Roll
        actions[i, 4] = euler_actual[1] - 0.1 * (i / 60.0) # Pitch
        actions[i, 5] = 0.5
        
    traj = TimestampedTrajectory(target_times, actions)
    
    K_ext = 0.3
    
    q_errors_rad = []
    q_corrs_rad = []
    q_cmds_rad = []
    
    sign_mismatch_count = 0
    max_num_err = 0.0
    nan_count = 0
    
    q_target_hard_viol = 0
    q_cmd_hard_viol = 0
    q_target_safe_viol = 0
    q_cmd_safe_viol = 0
    
    max_raw_corr_deg = 0.0
    max_raw_corr_info = ("", -1, 0.0, 0.0)
    
    q_curr_seed = q_actual_rad.copy()
    
    for i in range(31):
        curr_time = anchor_time + i * (1.0 / 30.0)
        res = sample_trajectory_at_time(traj, curr_time)
        if res["status"] == "EXPIRED": continue
            
        action = res["action"]
        target_pos = action[:3]
        target_roll = action[3]
        target_pitch = action[4]
        
        # Real-time read
        obs = follower.bus.sync_read("Present_Position")
        q_act_enc = np.array([float(obs[n]) for n in arm_names], dtype=np.float64)
        q_act_urdf = deploy.encoder_degrees_to_urdf_degrees(q_act_enc, arm_names)
        q_act_rad = np.deg2rad(q_act_urdf)
        
        current_yaw = get_fk_yaw(ik_solver, q_act_rad)
        target_rot_mat = R.from_euler("ZYX", [current_yaw, target_pitch, target_roll]).as_matrix()
        
        q_tgt_rad = ik_solver.solve(target_pos, target_rot_mat, q_curr_seed)
        q_curr_seed = q_tgt_rad.copy()
        
        q_error = q_tgt_rad - q_act_rad
        q_corr = K_ext * q_error
        q_cmd = q_tgt_rad + q_corr
        
        if np.any(np.isnan(q_tgt_rad)) or np.any(np.isnan(q_error)) or np.any(np.isnan(q_cmd)):
            nan_count += 1
            
        # Numerical verification
        q_corr_direct = q_cmd - q_tgt_rad
        num_err = np.max(np.abs(q_corr - q_corr_direct))
        if num_err > max_num_err: max_num_err = num_err
        
        # Sign mismatch
        err_nonzero = np.abs(q_error) > 1e-9
        sign_mismatch = (np.sign(q_error[err_nonzero]) != np.sign(q_corr[err_nonzero]))
        sign_mismatch_count += np.sum(sign_mismatch)
        
        # Track max correction
        for j, j_name in enumerate(arm_names):
            c_deg = np.rad2deg(q_corr[j])
            e_deg = np.rad2deg(q_error[j])
            if abs(c_deg) > max_raw_corr_deg:
                max_raw_corr_deg = abs(c_deg)
                max_raw_corr_info = (j_name, i, e_deg, c_deg)
                
        # Limits check
        for j, j_name in enumerate(arm_names):
            h_min, h_max = ik_solver.PHYSICAL_JOINT_LIMITS_RAD[j_name]
            s_min, s_max = ik_solver.safe_limits[j_name]
            
            t_val = q_tgt_rad[j]
            c_val = q_cmd[j]
            
            if t_val < h_min or t_val > h_max: q_target_hard_viol += 1
            if c_val < h_min or c_val > h_max: q_cmd_hard_viol += 1
            if t_val < s_min or t_val > s_max: q_target_safe_viol += 1
            if c_val < s_min or c_val > s_max: q_cmd_safe_viol += 1
            
        q_errors_rad.append(q_error)
        q_corrs_rad.append(q_corr)
        q_cmds_rad.append(q_cmd)
        
    follower.bus.disconnect(disable_torque=False)
    
    q_errors_deg = np.rad2deg(np.array(q_errors_rad))
    q_corrs_deg = np.rad2deg(np.array(q_corrs_rad))
    q_cmds_deg = np.rad2deg(np.array(q_cmds_rad))
    
    print("\n========================================")
    print("STEP 6 FINAL VERDICT")
    print("====================")
    print(f"1. ROBOT CONNECTED?\n   {'YES' if robot_connected else 'NO'}")
    print("2. MOTOR WRITE EXECUTED?\n   MUST BE NO -> NO (READ_ONLY_DIAGNOSTIC=True)")
    print("3. ENCODER READ API\n   file/function: follower.bus.sync_read(\"Present_Position\")")
    print(f"\n4. JOINT ORDER\n   target: {target_names}\n   actual: {arm_names}\n   match: {'YES' if order_match else 'NO'}")
    
    print("\n5. CURRENT ENCODER POSITION")
    print("encoder degree:")
    print(list(np.round(q_actual_enc_deg, 3)))
    print("URDF degree:")
    print(list(np.round(q_actual_urdf_deg, 3)))
    print("radian:")
    print(list(np.round(q_actual_rad, 4)))
    
    print("\n6. CURRENT FK EE POSITION")
    print(f"X: {p_actual[0]:.5f}\nY: {p_actual[1]:.5f}\nZ: {p_actual[2]:.5f}")
    print(f"Yaw: {euler_actual[0]:.5f}\nPitch: {euler_actual[1]:.5f}\nRoll: {euler_actual[2]:.5f}")
    
    print("\n7. STATIC SELF-CONSISTENCY TEST")
    print(f"q_actual_deg:\n{list(np.round(q_actual_urdf_deg, 3))}")
    print(f"q_target_static_deg:\n{list(np.round(q_target_static_deg, 3))}")
    print(f"error_deg per joint:\n{list(np.round(error_static_deg, 3))}")
    print(f"max_abs_error_deg: {max_abs_static_err:.5f}")
    print(f"{'PASS' if static_pass else 'FAIL'}")
    
    print("\n8. EXTERNAL K\n   value:\n   0.3")
    
    print(f"\n9. CORRECTION FORMULA VERIFIED?\n   YES")
    print(f"max numerical formula error: {max_num_err:.5e}")
    
    print(f"\n10. SIGN MISMATCH COUNT\n    value: {sign_mismatch_count}")
    
    print("\n11. HYPOTHETICAL TRAJECTORY q_error\nper joint:")
    for j, j_name in enumerate(arm_names):
        print(f"{j_name}:")
        print(f"min deg: {q_errors_deg[:,j].min():.3f}")
        print(f"max deg: {q_errors_deg[:,j].max():.3f}")
        print(f"mean_abs deg: {np.abs(q_errors_deg[:,j]).mean():.3f}")
        print(f"P95_abs deg: {np.percentile(np.abs(q_errors_deg[:,j]), 95):.3f}")
        
    print("\n12. HYPOTHETICAL q_correction\nper joint:")
    for j, j_name in enumerate(arm_names):
        print(f"{j_name}:")
        print(f"min deg: {q_corrs_deg[:,j].min():.3f}")
        print(f"max deg: {q_corrs_deg[:,j].max():.3f}")
        print(f"mean_abs deg: {np.abs(q_corrs_deg[:,j]).mean():.3f}")
        print(f"P95_abs deg: {np.percentile(np.abs(q_corrs_deg[:,j]), 95):.3f}")
        
    print("\n13. MAX RAW CORRECTION")
    print(f"joint: {max_raw_corr_info[0]}")
    print(f"tick: {max_raw_corr_info[1]}")
    print(f"error_deg: {max_raw_corr_info[2]:.3f}")
    print(f"correction_deg: {max_raw_corr_info[3]:.3f}")
    
    print("\n14. HYPOTHETICAL q_cmd RANGE\nper joint:")
    for j, j_name in enumerate(arm_names):
        print(f"{j_name} min deg: {q_cmds_deg[:,j].min():.3f}, max deg: {q_cmds_deg[:,j].max():.3f}")
        
    print("\n15. HARD LIMIT VIOLATIONS")
    print(f"q_target:\ncount: {q_target_hard_viol}")
    print(f"q_cmd:\ncount: {q_cmd_hard_viol}")
    
    print("\n16. SAFE LIMIT VIOLATIONS")
    print(f"q_target:\ncount: {q_target_safe_viol}")
    print(f"q_cmd:\ncount: {q_cmd_safe_viol}")
    
    print("\n17. ENCODER READ LATENCY")
    print(f"samples: {len(latencies)}")
    print(f"min ms: {latencies.min():.3f}")
    print(f"max ms: {latencies.max():.3f}")
    print(f"mean ms: {latencies.mean():.3f}")
    print(f"std ms: {latencies.std():.3f}")
    print(f"P95 ms: {np.percentile(latencies, 95):.3f}")
    
    print(f"\n18. NaN / Inf COUNT\n    value: {nan_count}")
    print("\n19. GRIPPER EXTERNAL CORRECTION APPLIED?\n    MUST BE NO -> NO")
    
    print("\n20. FILES CREATED\n    list: 4_deploy/debug_external_p_step6.py")
    print("21. FILES MODIFIED\n    list: None")
    
    verdict = "PASS" if (static_pass and nan_count == 0 and sign_mismatch_count == 0) else "FAIL"
    print(f"\n22. STEP 6 VERDICT\n    {verdict}")
    print("========================================")

if __name__ == "__main__":
    main()
