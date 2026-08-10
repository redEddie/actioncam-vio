import sys
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
sys.path.insert(0, str(DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'))

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def main():
    port = "/dev/ttyACM0"
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    
    # Connect BUS only, do NOT call follower.configure() yet!
    print("Connecting bus without full follower configuration...")
    follower.bus.connect(handshake=True)
    
    print("Bus connected. Reading Present_Position before any torque enable...")
    obs = follower.bus.sync_read("Present_Position")
    
    print("Present_Position read successfully:")
    for k, v in obs.items():
        print(f"  {k}: {v:.2f}")
        
    print("\nReading Goal_Position before any torque enable...")
    try:
        goal_obs = follower.bus.sync_read("Goal_Position")
        print("Goal_Position read successfully:")
        for k, v in goal_obs.items():
            diff = goal_obs[k] - obs[k]
            print(f"  {k}: {v:.2f} (diff vs Present: {diff:.2f})")
    except Exception as e:
        print(f"Could not read Goal_Position: {e}")
        goal_obs = None
    
    print("\nSynchronizing Goal_Position to Present_Position...")
    follower.bus.sync_write("Goal_Position", obs)
    
    # Read back to verify
    try:
        new_goal_obs = follower.bus.sync_read("Goal_Position")
        print("Goal_Position synced. Verifying:")
        for k, v in new_goal_obs.items():
            diff = new_goal_obs[k] - obs[k]
            print(f"  {k}: {v:.2f} (diff vs Present: {diff:.2f})")
    except Exception:
        pass
        
    print("\nNow calling follower.configure() which will safely enable torque...")
    follower.configure()
    
    # Monitor displacement
    time_points = [0.05, 0.1, 0.2, 0.5]
    print(f"\nMonitoring displacement after torque enable at {time_points}s...")
    start_t = time.time()
    
    results = {}
    for t_target in time_points:
        while time.time() - start_t < t_target:
            pass
        cur_obs = follower.bus.sync_read("Present_Position")
        results[t_target] = cur_obs
        
    for t_target in time_points:
        print(f"\n{int(t_target*1000)}ms displacement:")
        cur_obs = results[t_target]
        for k in obs.keys():
            diff = cur_obs[k] - obs[k]
            print(f"  {k}: {diff:+.2f}")
            
    print("\nDone. Disconnecting.")
    follower.bus.disconnect(disable_torque=True)
    
if __name__ == "__main__":
    main()
