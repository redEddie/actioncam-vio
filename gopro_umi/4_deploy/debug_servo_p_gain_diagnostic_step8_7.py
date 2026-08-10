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
    
    return {
        "mean_err": mean_err,
        "mae_err": mae_err,
        "std_err": std_err,
        "max_err": max_err,
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
        # Pre-check config
        res_P = follower.bus.sync_read("P_Coefficient")
        res_I = follower.bus.sync_read("I_Coefficient")
        res_D = follower.bus.sync_read("D_Coefficient")
        print("Initial PID:")
        print("P:", [res_P[n] for n in arm_names])
        print("I:", [res_I[n] for n in arm_names])
        print("D:", [res_D[n] for n in arm_names])
        
        # Trial A (P=16)
        metrics_A = run_trial(follower, arm_names, fixed_start_enc, "A (P=16)")
        
        # Sync Current
        print("\nSyncing Goal to Present before P change...")
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        time.sleep(2.0)
        
        # P Change
        print("\nChanging P to 32...")
        follower.bus.sync_write("P_Coefficient", {n: 32 for n in arm_names})
        time.sleep(0.1)
        
        res_P2 = follower.bus.sync_read("P_Coefficient")
        print("P read-back:", [res_P2[n] for n in arm_names])
        
        # 2sec hold after P change
        print("Holding 2 sec after P change...")
        for _ in range(60):
            follower.bus.sync_write("Goal_Position", obs)
            time.sleep(1.0/30.0)
            
        print("Hold complete. Checking for snap/oscillation...")
        obs2 = follower.bus.sync_read("Present_Position")
        max_snap = max([abs(float(obs2[n]) - float(obs[n])) for n in arm_names])
        print(f"Max snap during P=32 hold: {max_snap:.3f} deg")
        if max_snap > 5.0:
            print("SNAP OR OSCILLATION DETECTED. ABORTING TRIAL B.")
            follower.bus.sync_write("P_Coefficient", {n: 16 for n in arm_names})
            return
            
        # Trial B (P=32)
        metrics_B = run_trial(follower, arm_names, fixed_start_enc, "B (P=32)")
        
        # Current-Pose Sync Test
        print("\n--- CURRENT-POSE SYNC TEST (P=32) ---")
        obs3 = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs3)
        time.sleep(0.5)
        goal3 = follower.bus.sync_read("Goal_Position")
        pres3 = follower.bus.sync_read("Present_Position")
        
        goal_arr = np.array([float(goal3[n]) for n in arm_names])
        pres_arr = np.array([float(pres3[n]) for n in arm_names])
        obs3_arr = np.array([float(obs3[n]) for n in arm_names])
        
        print("Goal Write (Raw):", obs3_arr)
        print("Goal Read  (Raw):", goal_arr)
        print("Pres Read  (Raw):", pres_arr)
        print("Residual after sync:", goal_arr - pres_arr)
        
    finally:
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
