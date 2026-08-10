import sys
import os
from pathlib import Path

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def main():
    config = SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=False)
    follower = SO100Follower(config)
    
    print("=== FORENSIC OUTPUT ===")
    print(f"Norm mode wrist_flex: {follower.bus.motors['wrist_flex'].norm_mode}")
    print(f"Is calibration loaded? {len(follower.bus.calibration) > 0}")
    if len(follower.bus.calibration) > 0:
        c = follower.bus.calibration['wrist_flex']
        print(f"wrist_flex calib min: {c.range_min}, max: {c.range_max}, drive_mode: {c.drive_mode}")
    
    print("Normalized data list:", follower.bus.normalized_data)
    
if __name__ == "__main__":
    main()
