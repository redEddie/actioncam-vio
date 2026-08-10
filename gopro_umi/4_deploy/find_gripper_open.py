import time
import numpy as np
import sys
from pathlib import Path

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction

def main():
    print("=" * 50)
    print("🦀 그리퍼 최대 열림 한계(RAW 값) 찾기")
    print("천천히 열립니다. 원하는 만큼 열렸을 때 [Ctrl+C]를 눌러 멈추세요!")
    print("=" * 50)

    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    
    all_m = list(follower.bus.motors.keys())
    
    # 현재 상태에서 출발
    start_deg = float(follower.get_observation()["gripper.pos"])
    current_raw = start_deg * (4096.0 / 360.0)
    
    step_size = -10.0 # 0.1초마다 10씩 감소 (값이 작아질수록 열림)
    
    try:
        # 안전을 위해 0 미만으로는 내려가지 않도록 (필요시 - 범위로 수정 가능)
        while current_raw >= -500:
            action = {}
            # 팔 관절 유지
            current_obs = follower.get_observation()
            for m in all_m:
                if "gripper" not in m:
                    action[f"{m}.pos"] = float(current_obs[f"{m}.pos"])
                    
            # 그리퍼 RAW -> Degree 변환
            cur_deg = current_raw * (360.0 / 4096.0)
            action["gripper.pos"] = float(cur_deg)
            
            follower.send_action(RobotAction(action))
            
            print(f"👐 현재 그리퍼 RAW 값: {int(current_raw)} (원할 때 Ctrl+C를 누르세요)")
            
            current_raw += step_size
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 정지! 적절한 최대 열림 값을 확인했습니다.")
        print(f"✅ 사용자가 멈춘 그리퍼 RAW 값: {int(current_raw - step_size)}")
        
    finally:
        follower.disconnect()

if __name__ == "__main__":
    main()
