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
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def safe_connect(port):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    follower.bus.connect(handshake=True)
    obs = follower.bus.sync_read("Present_Position")
    follower.bus.sync_write("Goal_Position", obs)
    time.sleep(0.05)
    follower.configure()
    # Force P=32, I=0, D=32 for Trial A start
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    follower.bus.sync_write("P_Coefficient", {n: 32 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    time.sleep(0.2)
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

def run_trial(follower, arm_names, target, label):
    print(f"\n--- TRIAL {label} ---")
    move_to_fixed_start(follower, arm_names, target)
    
    print("10s STATIC HOLD DIAGNOSTIC...")
    logs = defaultdict(list)
    hold_steps = 10 * 30
    target_dict = {n: float(target[j]) for j, n in enumerate(arm_names)}
    
    telemetry_regs = ["Present_Position", "Goal_Position", "Present_Velocity", "Present_Load", "Present_Temperature"]
    
    for _ in range(hold_steps):
        t0 = time.perf_counter()
        follower.bus.sync_write("Goal_Position", target_dict)
        
        tel = read_telemetry(follower, arm_names, telemetry_regs)
        for k, v in tel.items():
            logs[k].append(v)
            
        elapsed = time.perf_counter() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    for k in logs:
        logs[k] = np.array(logs[k])
        
    errors = np.array(logs["Goal_Position"]) - np.array(logs["Present_Position"])
    final_errors = errors[-150:] # last 5 sec
    
    mean_err = np.mean(final_errors, axis=0)
    mae_err = np.mean(np.abs(final_errors), axis=0)
    std_err = np.std(final_errors, axis=0)
    max_err = np.max(np.abs(final_errors), axis=0)
    
    print("FINAL 5 SEC MEAN ERROR:", mean_err)
    print("MAE:", mae_err)
    print("STD:", std_err)
    
    idx = arm_names.index("elbow_flex")
    print(f"\nELBOW METRICS:")
    print(f"Error (mean): {mean_err[idx]:.3f} deg")
    if "Present_Velocity" in logs:
        print(f"Velocity: {np.mean(logs['Present_Velocity'][-30:, idx]):.3f}")
    if "Present_Load" in logs:
        print(f"Load raw: {np.mean(logs['Present_Load'][-30:, idx]):.1f}")
    if "Present_Temperature" in logs:
        t_start = logs["Present_Temperature"][0, idx]
        t_end = logs["Present_Temperature"][-1, idx]
        print(f"Temperature Start: {t_start:.1f} °C, End: {t_end:.1f} °C")
        
    # Hunting Check
    err_elbow = errors[-150:, idx]
    sign_changes = np.sum(np.diff(np.sign(err_elbow)) != 0)
    print(f"Error sign changes: {sign_changes}")
    
    # Store position std for hunting analysis
    pos_std = np.std(logs["Present_Position"][-150:], axis=0)
    
    return {
        "mean_err": mean_err,
        "mae_err": mae_err,
        "std_err": std_err,
        "pos_std": pos_std,
        "sign_changes": sign_changes,
        "load": np.mean(logs["Present_Load"][-30:]) if "Present_Load" in logs else None,
        "t_start": logs["Present_Temperature"][0, idx] if "Present_Temperature" in logs else None,
        "t_end": logs["Present_Temperature"][-1, idx] if "Present_Temperature" in logs else None
    }

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    try:
        res_P = follower.bus.sync_read("P_Coefficient")
        res_I = follower.bus.sync_read("I_Coefficient")
        res_D = follower.bus.sync_read("D_Coefficient")
        print("Initial PID:")
        print("P:", [res_P[n] for n in arm_names])
        print("I:", [res_I[n] for n in arm_names])
        print("D:", [res_D[n] for n in arm_names])
        
        # Trial A (P=32)
        metrics_A = run_trial(follower, arm_names, fixed_start_enc, "A (P=32)")
        
        # Sync Current
        print("\nSyncing Goal to Present before P change...")
        obs = follower.bus.sync_read("Present_Position")
        q_before = np.array([float(obs[n]) for n in arm_names])
        follower.bus.sync_write("Goal_Position", obs)
        time.sleep(2.0)
        
        # Transient Tracking
        obs_trans = []
        timestamps = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        
        # P Change
        print("\nChanging P to 48...")
        
        # Just before write
        obs_just_before = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs_just_before)
        
        t_start = time.perf_counter()
        follower.bus.sync_write("P_Coefficient", {n: 48 for n in arm_names})
        
        for t_target in timestamps:
            while time.perf_counter() - t_start < t_target:
                time.sleep(0.001)
            obs_curr = follower.bus.sync_read("Present_Position")
            obs_trans.append(np.array([float(obs_curr[n]) for n in arm_names]))
            
        print("\n--- TRANSIENT AFTER P CHANGE ---")
        for i, t in enumerate(timestamps):
            print(f"{int(t*1000)}ms: {obs_trans[i] - q_before}")
            
        max_disp = np.max(np.abs(np.array(obs_trans) - q_before), axis=0)
        print("Max displacement:", max_disp)
        
        if np.any(max_disp > 5.0):
            print("LARGE SNAP/OSCILLATION DETECTED! ABORTING P=48 TRIAL.")
            follower.bus.sync_write("P_Coefficient", {n: 32 for n in arm_names})
            return
            
        res_P2 = follower.bus.sync_read("P_Coefficient")
        print("\nP read-back:", [res_P2[n] for n in arm_names])
        
        # Trial B (P=48)
        metrics_B = run_trial(follower, arm_names, fixed_start_enc, "B (P=48)")
        
        print("\n--- COMPARISON ---")
        idx = arm_names.index("elbow_flex")
        e32 = abs(metrics_A["mean_err"][idx])
        e48 = abs(metrics_B["mean_err"][idx])
        imp = (e32 - e48) / e32 * 100.0 if e32 > 0 else 0
        print(f"Elbow Improvement (P32 -> P48): {imp:.1f}%")
        
        print("\nOther Joints P32 MAE:")
        print(metrics_A["mae_err"])
        print("Other Joints P48 MAE:")
        print(metrics_B["mae_err"])
        print("Other Joints Improvement:")
        print((metrics_A["mae_err"] - metrics_B["mae_err"]) / np.where(metrics_A["mae_err"]>0, metrics_A["mae_err"], 1) * 100)
        
        print("\nStability (Sign Changes):")
        print("P32:", metrics_A["sign_changes"])
        print("P48:", metrics_B["sign_changes"])
        
        print("\nPosition STD:")
        print("P32:", metrics_A["pos_std"])
        print("P48:", metrics_B["pos_std"])
        
    finally:
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
