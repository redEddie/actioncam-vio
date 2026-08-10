import sys
import os
import time
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

def safe_connect(port):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    follower.bus.connect(handshake=True)
    obs = follower.bus.sync_read("Present_Position")
    follower.bus.sync_write("Goal_Position", obs)
    time.sleep(0.05)
    follower.configure()
    time.sleep(0.2)
    return follower

def move_to_fixed_start(follower, arm_names, target):
    obs = follower.bus.sync_read("Present_Position")
    q_act = np.array([float(obs[n]) for n in arm_names])
    delta = target - q_act
    
    dur = 3.0
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

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    
    follower = safe_connect("/dev/ttyACM0")
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    try:
        print("--- Moving to FIXED PHYSICAL START POSE ---")
        move_to_fixed_start(follower, arm_names, fixed_start_enc)
        
        print("Holding for 3.0s...")
        for _ in range(90):
            t0 = time.monotonic()
            follower.bus.sync_write("Goal_Position", {n: float(fixed_start_enc[j]) for j, n in enumerate(arm_names)})
            elapsed = time.monotonic() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        print("\n--- HOLD COMPLETED ---")
        goal = follower.bus.sync_read("Goal_Position")
        present = follower.bus.sync_read("Present_Position")
        
        q_goal = np.array([float(goal[n]) for n in arm_names])
        q_present = np.array([float(present[n]) for n in arm_names])
        
        print(f"RAW PRESENT AFTER HOLD:\n{q_present}")
        print(f"RAW GOAL AFTER HOLD:\n{q_goal}")
        print(f"RAW GOAL - RAW PRESENT:\n{q_goal - q_present}")
        
        print("\n--- CONVERSION ROUND TRIP TEST ---")
        q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_present, arm_names)
        q_rad = np.deg2rad(q_urdf)
        q_enc_recovered = deploy.urdf_degrees_to_encoder_degrees(q_urdf, arm_names)
        
        print("Raw encoder:", q_present)
        print("encoder -> URDF deg:", q_urdf)
        print("URDF deg -> encoder recovered:", q_enc_recovered)
        print("Round trip error:", q_enc_recovered - q_present)
        
        print("\n--- IK TO MOTOR OFFLINE TRACE ---")
        T_start = ik_solver.forward_kinematics(q_rad)
        pos = T_start[:3, 3]
        rot = T_start[:3, :3]
        
        pos_target = pos.copy()
        pos_target[2] += 0.002
        
        print("Solving IK for +2mm...")
        q_nom_rad = ik_solver.solve(pos_target, rot, current_joints=q_rad)
        
        if q_nom_rad is not None:
            q_nom_urdf_deg = np.rad2deg(q_nom_rad)
            q_motor_enc = deploy.urdf_degrees_to_encoder_degrees(q_nom_urdf_deg, arm_names)
            
            print("q_nom_rad:", q_nom_rad)
            print("q_nom_urdf_deg:", q_nom_urdf_deg)
            print("motor command representation (encoder deg):", q_motor_enc)
            
            # Inverse trace
            q_recov_urdf = deploy.encoder_degrees_to_urdf_degrees(q_motor_enc, arm_names)
            q_recov_rad = np.deg2rad(q_recov_urdf)
            
            print("Motor round trip error (rad):", q_recov_rad - q_nom_rad)
            print("Motor round trip error (deg):", np.rad2deg(q_recov_rad - q_nom_rad))
            
            T_nom = ik_solver.forward_kinematics(q_nom_rad)
            T_recov = ik_solver.forward_kinematics(q_recov_rad)
            
            pos_diff = np.linalg.norm(T_nom[:3,3] - T_recov[:3,3]) * 1000.0
            print(f"FK Round trip position diff (mm): {pos_diff:.5f}")
            
    finally:
        print("\nDisconnecting with torque enabled...")
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
