#!/usr/bin/env python3
import sys
from pathlib import Path

# Add lerobot to sys.path
deploy_dir = Path(__file__).parent.resolve()
lerobot_src = deploy_dir / "Teleop" / "lerobot" / "src"
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

def main():
    print("Connecting to SO-100 Follower...")
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.bus.connect(handshake=True)
    
    # Ensure torque is disabled before writing to EPROM
    follower.bus.disconnect(disable_torque=True)
    follower.bus.connect(handshake=True)
    
    target_p = 48
    print(f"Setting P_Coefficient to {target_p} for all motors...")
    
    for name in follower.bus.motors:
        try:
            current_p = follower.bus.read("P_Coefficient", name)
            if current_p != target_p:
                print(f"[{name}] Current P: {current_p}. Updating to {target_p}...")
                follower.bus.write("P_Coefficient", name, target_p)
                new_p = follower.bus.read("P_Coefficient", name)
                print(f"[{name}] Verification -> P_Coefficient is now {new_p}")
            else:
                print(f"[{name}] P_Coefficient is already {target_p}. Skipping.")
        except Exception as e:
            print(f"[{name}] Error accessing motor: {e}")
            
    follower.bus.disconnect(disable_torque=True)
    print("Done! P-gains updated successfully.")

if __name__ == "__main__":
    main()
