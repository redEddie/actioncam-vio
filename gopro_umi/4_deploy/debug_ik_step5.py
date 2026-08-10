import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import copy
import pandas as pd

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
sys.path.insert(0, str(DEPLOY_DIR))

from debug_sampler_step4 import build_dummy_trajectory, sample_trajectory_at_time
from ik_solver_v7 import DLSInverseKinematicsV7

def get_fk_yaw(ik_solver: DLSInverseKinematicsV7, q_rad: np.ndarray) -> float:
    # Get TCP rotation matrix
    T = ik_solver.forward_kinematics(q_rad)
    R_mat = T[:3, :3]
    # "ZYX" order -> [Yaw, Pitch, Roll]
    euler = R.from_matrix(R_mat).as_euler("ZYX")
    return euler[0]

def main():
    print("--- 1. Initialize IK Solver ---")
    urdf_path = str(PROJECT_ROOT / "4_deploy/URDF/so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path=urdf_path)
    
    # 2. Start Configuration (Degree -> Radian)
    start_q_deg = np.array([0.0, -73.5, 71.1, 47.0, 0.0], dtype=np.float64)
    q_seed = np.deg2rad(start_q_deg)
    
    # Check FK to get initial pose
    T_init = ik_solver.forward_kinematics(q_seed)
    p_init = T_init[:3, 3]
    euler_init = R.from_matrix(T_init[:3, :3]).as_euler("ZYX")
    print(f"Start Position: {p_init}")
    print(f"Start Euler ZYX: {euler_init}")
    
    # Create realistic mock absolute trajectory starting from the current pose
    print("\n--- 2. Build 60Hz Trajectory ---")
    anchor_time = time.monotonic()
    num_steps = 60
    dt = 1.0 / 60.0
    offsets = np.arange(1, num_steps + 1) * dt
    target_times = anchor_time + offsets
    
    # Mocking actions (small movements from initial pose)
    actions = np.zeros((num_steps, 6), dtype=np.float32)
    for i in range(num_steps):
        actions[i, 0] = p_init[0] + 0.05 * (i / 60.0) # Move 5cm in X
        actions[i, 1] = p_init[1] + 0.02 * (i / 60.0)
        actions[i, 2] = p_init[2] - 0.01 * (i / 60.0)
        actions[i, 3] = euler_init[2] + 0.1 * (i / 60.0) # Roll target
        actions[i, 4] = euler_init[1] - 0.1 * (i / 60.0) # Pitch target
        actions[i, 5] = 0.5 # Gripper
        
    from debug_sampler_step4 import TimestampedTrajectory
    traj = TimestampedTrajectory(target_times, actions)
    
    print("\n--- 3. Sampling 30Hz Targets & IK ---")
    
    q_curr = q_seed.copy()
    
    # Metrics
    ik_failures = 0
    nan_count = 0
    pos_errors = []
    rot_errors_rad = []
    
    dq_list_deg = []
    q_all_deg = []
    
    hard_limit_violations = 0
    safe_limit_violations = 0
    
    # Safe limits dictionary from solver
    safe_limits = ik_solver.safe_limits
    hard_limits = ik_solver.PHYSICAL_JOINT_LIMITS_RAD
    joint_names = ik_solver.joint_names
    
    print(f"JOINT_ORDER = {joint_names}")
    
    num_sampled_targets = 0
    
    # 30Hz control ticks for 1 second
    for i in range(31):
        curr_time = anchor_time + i * (1.0 / 30.0)
        res = sample_trajectory_at_time(traj, curr_time)
        
        status = res["status"]
        idx = res["index"]
        
        if status == "EXPIRED":
            # EXPIRED TARGET PASSED INTO IK? MUST BE NO
            continue
            
        num_sampled_targets += 1
        action = res["action"]
        
        target_pos = action[:3]
        target_roll = action[3]
        target_pitch = action[4]
        gripper_target = action[5]
        
        # Yaw processing
        current_yaw = get_fk_yaw(ik_solver, q_curr)
        
        target_rot_matrix = R.from_euler("ZYX", [current_yaw, target_pitch, target_roll]).as_matrix()
        
        # Check NaNs
        if np.any(np.isnan(target_pos)) or np.any(np.isnan(target_rot_matrix)) or np.any(np.isnan(q_curr)):
            nan_count += 1
            continue
            
        # IK Solve
        q_target = ik_solver.solve(target_pos, target_rot_matrix, q_curr)
        
        if not ik_solver.last_converged:
            ik_failures += 1
            
        if np.any(np.isnan(q_target)):
            nan_count += 1
            continue
            
        # FK Validation
        T_fk = ik_solver.forward_kinematics(q_target)
        p_fk = T_fk[:3, 3]
        R_fk = T_fk[:3, :3]
        
        if np.any(np.isnan(p_fk)) or np.any(np.isnan(R_fk)):
            nan_count += 1
            continue
            
        pos_err = np.linalg.norm(target_pos - p_fk) * 1000.0 # mm
        pos_errors.append(pos_err)
        
        R_err = target_rot_matrix @ R_fk.T
        angle_err = R.from_matrix(R_err).magnitude() # rad
        rot_errors_rad.append(angle_err)
        
        # Jump check
        dq = q_target - q_curr
        dq_list_deg.append(np.rad2deg(np.abs(dq)))
        q_all_deg.append(np.rad2deg(q_target))
        
        # Limits check
        for j, j_name in enumerate(joint_names):
            val = q_target[j]
            h_min, h_max = hard_limits[j_name]
            s_min, s_max = safe_limits[j_name]
            if val < h_min or val > h_max:
                hard_limit_violations += 1
            if val < s_min or val > s_max:
                safe_limit_violations += 1
                
        # Update q_curr for next step (IK seed continuity)
        q_curr = q_target.copy()

    # Final summary statistics
    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors_rad)
    dq_mat = np.array(dq_list_deg)
    q_mat = np.array(q_all_deg)
    
    print("\n========================================")
    print("STEP 5 FINAL VERDICT")
    print("====================")
    print("1. IK SOLVER")
    print("   file: 4_deploy/ik_solver_v7.py")
    print("   class/function: DLSInverseKinematicsV7.solve")
    
    print(f"\n2. JOINT ORDER\n   value: {joint_names}")
    
    print("\n3. IK INPUT")
    print("   position unit: meters")
    print("   rotation convention: Rotation matrix via 'ZYX' (Yaw from actual/q_curr FK, Pitch/Roll from model)")
    print("   q seed unit: radians")
    
    print(f"\n4. NUMBER OF SAMPLED TARGETS\n   value: {num_sampled_targets}")
    
    print(f"\n5. IK OUTPUT SHAPE\n   value: {q_curr.shape}")
    print("\n6. IK OUTPUT UNIT\n   value: radians")
    
    print(f"\n7. IK FAILURE COUNT\n   value: {ik_failures}")
    print(f"\n8. NaN / Inf COUNT\n   value: {nan_count}")
    
    print("\n9. POSITION FK ERROR")
    print(f"   min: {pos_errors.min():.5f} mm")
    print(f"   max: {pos_errors.max():.5f} mm")
    print(f"   mean: {pos_errors.mean():.5f} mm")
    print(f"   std: {pos_errors.std():.5f} mm")
    print(f"   P95: {np.percentile(pos_errors, 95):.5f} mm")
    
    print("\n10. ORIENTATION FK ERROR")
    print(f"    min rad: {rot_errors.min():.7f}")
    print(f"    max rad: {rot_errors.max():.7f}")
    print(f"    mean rad: {rot_errors.mean():.7f}")
    print(f"    P95 rad: {np.percentile(rot_errors, 95):.7f}")
    
    rot_errors_deg = np.rad2deg(rot_errors)
    print(f"    min deg: {rot_errors_deg.min():.5f}")
    print(f"    max deg: {rot_errors_deg.max():.5f}")
    print(f"    mean deg: {rot_errors_deg.mean():.5f}")
    print(f"    P95 deg: {np.percentile(rot_errors_deg, 95):.5f}")
    
    print("\n11. JOINT CONTINUITY\nper joint:")
    
    max_jump_overall = 0.0
    max_jump_info = ("", -1, 0.0)
    
    for j, j_name in enumerate(joint_names):
        max_dq = dq_mat[:, j].max()
        mean_dq = dq_mat[:, j].mean()
        print(f"    {j_name} max |dq| deg: {max_dq:.4f}")
        print(f"    {j_name} mean |dq| deg: {mean_dq:.4f}")
        
        idx_max = np.argmax(dq_mat[:, j])
        if max_dq > max_jump_overall:
            max_jump_overall = max_dq
            max_jump_info = (j_name, idx_max, max_dq)
            
    print("\nlargest single joint jump:")
    print(f"    joint: {max_jump_info[0]}")
    print(f"    tick: {max_jump_info[1]}")
    print(f"    value: {max_jump_info[2]:.4f} deg")
    
    print("\n12. JOINT RANGE\nper joint:")
    for j, j_name in enumerate(joint_names):
        min_q = q_mat[:, j].min()
        max_q = q_mat[:, j].max()
        print(f"    {j_name} min deg: {min_q:.3f}")
        print(f"    {j_name} max deg: {max_q:.3f}")
        
    print(f"\n13. HARD LIMIT VIOLATIONS\n    count: {hard_limit_violations}")
    print(f"\n14. SAFE LIMIT VIOLATIONS\n    count: {safe_limit_violations}")
    
    print("\n15. YAW SOURCE")
    print("    description: Computed from current actual/seed q_target via forward kinematics (`get_fk_yaw`)")
    
    print("\n16. GRIPPER PASSED INTO IK?\n    MUST BE NO -> NO")
    print("17. EXPIRED TARGET PASSED INTO IK?\n    MUST BE NO -> NO")
    print("18. MOTOR COMMAND EXECUTED?\n    MUST BE NO -> NO")
    
    print("\n19. FILES CREATED\n    list: 4_deploy/debug_ik_step5.py")
    print("20. FILES MODIFIED\n    list: None")
    
    has_jump = max_jump_overall > 30.0 # 30 degree jump is catastrophic
    verdict = "FAIL" if (nan_count > 0 or has_jump) else "PASS"
    print(f"\n21. STEP 5 VERDICT\n    {verdict}")
    print("========================================")

if __name__ == "__main__":
    main()
