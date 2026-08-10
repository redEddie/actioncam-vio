import sys
import os
import time
import numpy as np
from pathlib import Path
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
    new_obs = follower.bus.sync_read("Present_Position")
    max_snap = 0.0
    for k in obs:
        max_snap = max(max_snap, abs(new_obs[k] - obs[k]))
    if max_snap > 5.0:
        print(f"SNAP DETECTED: {max_snap:.2f} deg")
        sys.exit(1)
    return follower

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

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception:
            pass
    return data

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    try:
        # Audit Settings
        print("\n--- SERVO CONFIGURATION AUDIT (READ ONLY) ---")
        settings_regs = ["P_Coefficient", "I_Coefficient", "D_Coefficient"]
        for reg in settings_regs:
            try:
                res = follower.bus.sync_read(reg)
                print(f"{reg}: {[res[n] for n in arm_names]}")
            except Exception as e:
                print(f"{reg}: NOT AVAILABLE ({e})")
                
        print("\n--- Moving to FIXED TARGET ---")
        move_to_fixed_start(follower, arm_names, fixed_start_enc)
        
        # Static Hold 10s
        print("\n--- 10s STATIC HOLD DIAGNOSTIC ---")
        logs = defaultdict(list)
        hold_steps = 10 * 30
        
        target_dict = {n: float(fixed_start_enc[j]) for j, n in enumerate(arm_names)}
        
        telemetry_regs = ["Present_Position", "Present_Velocity", "Present_Load", "Present_Voltage", "Present_Temperature"]
        
        for step in range(hold_steps):
            t0 = time.perf_counter()
            follower.bus.sync_write("Goal_Position", target_dict)
            
            tel = read_telemetry(follower, arm_names, telemetry_regs)
            for k, v in tel.items():
                logs[k].append(v)
            
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        # Analysis
        for k in logs:
            logs[k] = np.array(logs[k])
            
        present = logs["Present_Position"]
        errors = fixed_start_enc - present
        
        print("\n--- STATIC ERROR BY JOINT (deg) ---")
        intervals = [(0,30, "0-1s"), (30,90, "1-3s"), (90,150, "3-5s"), (150,300, "5-10s")]
        for start, end, label in intervals:
            err_mean = np.mean(errors[start:end], axis=0)
            err_std = np.std(errors[start:end], axis=0)
            print(f"[{label}] Mean: {np.round(err_mean,3)}, Std: {np.round(err_std,3)}")
            
        print("\n--- PRESENT VELOCITY (final 1s mean) ---")
        if "Present_Velocity" in logs:
            vel_final = np.mean(logs["Present_Velocity"][-30:], axis=0)
            print(vel_final)
        else:
            print("Not available")
            
        print("\n--- ELBOW TELEMETRY (final 1s mean) ---")
        idx = arm_names.index("elbow_flex")
        for k in telemetry_regs:
            if k in logs:
                val = np.mean(logs[k][-30:, idx])
                print(f"{k}: {val:.3f}")
                
        # Phase 2: Current-Pose Hold
        print("\n--- CURRENT-POSE HOLD COMPARISON (2s) ---")
        curr_obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", curr_obs)
        
        curr_logs = defaultdict(list)
        for _ in range(60):
            t0 = time.perf_counter()
            follower.bus.sync_write("Goal_Position", curr_obs)
            tel = read_telemetry(follower, arm_names, telemetry_regs)
            for k, v in tel.items():
                curr_logs[k].append(v)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        for k in curr_logs:
            curr_logs[k] = np.array(curr_logs[k])
            
        err_curr = np.array([float(curr_obs[n]) for n in arm_names]) - curr_logs["Present_Position"]
        print(f"Position residual mean: {np.mean(err_curr, axis=0)}")
        if "Present_Load" in curr_logs:
            print(f"Elbow Load (Holding Current): {np.mean(curr_logs['Present_Load'][:, idx]):.3f}")
        if "Present_Velocity" in curr_logs:
            print(f"Elbow Velocity (Holding Current): {np.mean(curr_logs['Present_Velocity'][:, idx]):.3f}")
            
    finally:
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
