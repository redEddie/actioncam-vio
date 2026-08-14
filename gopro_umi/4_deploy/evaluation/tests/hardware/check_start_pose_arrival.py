import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
sys.path.insert(0, str(DEPLOY_DIR))
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

import deploy_smolvla_yawfree as deploy
from ik_solver_v7 import DLSInverseKinematicsV7
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

ARM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def main():
    print("============================================================")
    print(" STEP — VERIFIED START POSE PHYSICAL ARRIVAL CHECK ")
    print("============================================================")
    
    start_v2_path = DEPLOY_DIR / "yawfree_physical_start_v2.json"
    with open(start_v2_path, 'r') as f:
        v2_config = json.load(f)
        
    encoder_zero_deg = np.array(v2_config["encoder_zero_at_urdf_zero_deg"], dtype=np.float64)
    encoder_start_target = np.array(v2_config["encoder_start_deg"], dtype=np.float64)
    urdf_start_target = np.array(v2_config["urdf_start_deg"], dtype=np.float64)
    
    # Verify consistency: encoder_deg = urdf_deg + encoder_zero_deg
    calc_encoder_start = urdf_start_target + encoder_zero_deg
    urdf_to_encoder_match = np.allclose(calc_encoder_start, encoder_start_target)
    
    urdf_path = str(DEPLOY_DIR / "URDF/so_arm_with_gopro_final.urdf")
    ik_solver = DLSInverseKinematicsV7(urdf_path=urdf_path)
    
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.bus.connect(handshake=True)
    
    try:
        # 1. Read initial state
        print("Connecting to robot and reading initial Present_Position...")
        initial_encoder_dict = follower.bus.sync_read("Present_Position")
        initial_encoder_arr = np.array([float(initial_encoder_dict[n]) for n in ARM_NAMES], dtype=np.float64)
        
        # 2. Slow interpolation to start pose over 6.0 seconds
        print("Moving arm slowly to verified start pose (6.0s)...")
        duration_s = 6.0
        steps = 30
        interval = duration_s / steps
        follower.bus.enable_torque()
        
        t_start = time.monotonic()
        for step in range(1, steps + 1):
            ratio = step / steps
            desired_enc = initial_encoder_arr + ratio * (encoder_start_target - initial_encoder_arr)
            desired_dict = dict(zip(ARM_NAMES, desired_enc))
            follower.bus.sync_write("Goal_Position", desired_dict)
            time.sleep(interval)
        t_end = time.monotonic()
        
        move_duration_ms = (t_end - t_start) * 1000.0
        time.sleep(0.5) # Short settling time
        
        # 3. Read single arrival state
        arrival_encoder_dict = follower.bus.sync_read("Present_Position")
        arrival_encoder_arr = np.array([float(arrival_encoder_dict[n]) for n in ARM_NAMES], dtype=np.float64)
        arrival_urdf_deg = arrival_encoder_arr - encoder_zero_deg
        
        joint_errors_deg = arrival_urdf_deg - urdf_start_target
        max_abs_start_error_deg = float(np.max(np.abs(joint_errors_deg)))
        
        legacy_5deg_gate = "PASS" if max_abs_start_error_deg <= 5.0 else "FAIL"
        strict_1deg_gate = "PASS" if max_abs_start_error_deg <= 1.0 else "FAIL"
        
        # 4. Repeated 10 Live Reads (100ms spacing)
        print("Performing 10 repeated live reads (100ms interval)...")
        repeated_urdf_samples = []
        repeated_encoder_samples = []
        
        for _ in range(10):
            d = follower.bus.sync_read("Present_Position")
            enc = np.array([float(d[n]) for n in ARM_NAMES], dtype=np.float64)
            urdf = enc - encoder_zero_deg
            repeated_encoder_samples.append(enc)
            repeated_urdf_samples.append(urdf)
            time.sleep(0.1)
            
        rep_urdf_arr = np.array(repeated_urdf_samples) # [10, 5]
        mean_urdf_deg = np.mean(rep_urdf_arr, axis=0)
        std_urdf_deg = np.std(rep_urdf_arr, axis=0)
        p2p_urdf_deg = np.max(rep_urdf_arr, axis=0) - np.min(rep_urdf_arr, axis=0)
        position_stable = float(np.max(p2p_urdf_deg)) <= 0.25
        
        # 5. FK Diagnostic
        T_target = ik_solver.forward_kinematics(np.deg2rad(urdf_start_target))
        p_target = T_target[:3, 3] * 1000.0 # mm
        rpy_target = R.from_matrix(T_target[:3, :3]).as_euler("ZYX", degrees=True)
        
        T_actual = ik_solver.forward_kinematics(np.deg2rad(mean_urdf_deg))
        p_actual = T_actual[:3, 3] * 1000.0 # mm
        rpy_actual = R.from_matrix(T_actual[:3, :3]).as_euler("ZYX", degrees=True)
        
        dxyz_mm = p_actual - p_target
        pos_error_mm = float(np.linalg.norm(dxyz_mm))
        
        drpy_deg = rpy_actual - rpy_target
        
        # Final Verdict
        if strict_1deg_gate == "PASS":
            verdict = "START_POSE_PHYSICALLY_REACHED"
        elif legacy_5deg_gate == "PASS":
            verdict = "START_POSE_WITHIN_LEGACY_GATE_BUT_NOT_STRICT"
        else:
            verdict = "START_POSE_NOT_REACHED"

        report_data = {
            "START_COMMAND": {
                "COMMAND_TARGET_ENCODER": encoder_start_target.tolist(),
                "COMMAND_TARGET_URDF": urdf_start_target.tolist(),
                "URDF_TO_ENCODER_MATCH": "YES" if urdf_to_encoder_match else "NO"
            },
            "REAL_MOVE": {
                "START_MOVE": "PASS",
                "MOVE_DURATION_MS": move_duration_ms,
                "COMMAND_COUNT": steps
            },
            "ACTUAL_ARRIVE": {
                "ENCODER_TARGET": encoder_start_target.tolist(),
                "ENCODER_ACTUAL": arrival_encoder_arr.tolist(),
                "URDF_TARGET": urdf_start_target.tolist(),
                "URDF_ACTUAL": arrival_urdf_deg.tolist()
            },
            "PER_JOINT_ERROR": {
                "shoulder_pan": {"target": urdf_start_target[0], "actual": arrival_urdf_deg[0], "error": joint_errors_deg[0]},
                "shoulder_lift": {"target": urdf_start_target[1], "actual": arrival_urdf_deg[1], "error": joint_errors_deg[1]},
                "elbow_flex": {"target": urdf_start_target[2], "actual": arrival_urdf_deg[2], "error": joint_errors_deg[2]},
                "wrist_flex": {"target": urdf_start_target[3], "actual": arrival_urdf_deg[3], "error": joint_errors_deg[3]},
                "wrist_roll": {"target": urdf_start_target[4], "actual": arrival_urdf_deg[4], "error": joint_errors_deg[4]},
                "MAX_ABS_START_ERROR_DEG": max_abs_start_error_deg
            },
            "GATES": {
                "LEGACY_5DEG_GATE": legacy_5deg_gate,
                "STRICT_1DEG_DIAGNOSTIC_GATE": strict_1deg_gate
            },
            "ELBOW": {
                "ELBOW_TARGET_URDF_DEG": 79.8242,
                "ELBOW_ACTUAL_URDF_DEG": float(arrival_urdf_deg[2]),
                "ELBOW_ERROR_DEG": float(joint_errors_deg[2])
            },
            "REPEATED_READS": {
                "READ_COUNT": 10,
                "MEAN_URDF_DEG": mean_urdf_deg.tolist(),
                "STD_URDF_DEG": std_urdf_deg.tolist(),
                "P2P_URDF_DEG": p2p_urdf_deg.tolist(),
                "POSITION_STABLE": "YES" if position_stable else "NO"
            },
            "FK": {
                "TARGET_FK_XYZ_MM": p_target.tolist(),
                "ACTUAL_FK_XYZ_MM": p_actual.tolist(),
                "POSITION_ERROR_MM": pos_error_mm,
                "DX_MM": float(dxyz_mm[0]),
                "DY_MM": float(dxyz_mm[1]),
                "DZ_MM": float(dxyz_mm[2]),
                "TARGET_RPY_DEG": rpy_target.tolist(),
                "ACTUAL_RPY_DEG": rpy_actual.tolist(),
                "DROLL_DEG": float(drpy_deg[2]),
                "DPITCH_DEG": float(drpy_deg[1]),
                "DYAW_DEG": float(drpy_deg[0])
            },
            "ISOLATION": {
                "CONTROLLER_PRESETTLE_EXECUTED": "NO",
                "SMOLVLA_INFERENCE_EXECUTED": "NO",
                "SMOLVLA_POLICY_MOTOR_WRITES": 0
            },
            "PHYSICAL_HOLD": {
                "ROBOT_IS_NOW_HOLDING_START_POSE_FOR_USER_INSPECTION": "YES"
            },
            "FINAL_VERDICT": verdict
        }

        print("\n" + "="*60)
        print(" START POSE ARRIVAL DIAGNOSTIC REPORT ")
        print("="*60)
        print(json.dumps(report_data, indent=2))
        print("="*60)

        # Keep connection open for a moment and keep holding position
        time.sleep(1.0)

    finally:
        # Note: disconnect without disabling torque so motor holds position
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)

if __name__ == '__main__':
    main()
