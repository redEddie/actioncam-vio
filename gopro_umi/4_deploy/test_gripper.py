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
    print("🦀 그리퍼 최대 열림 / 최대 닫힘 테스트")
    print("=" * 50)

    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    
    all_m = list(follower.bus.motors.keys())
    
    # 그리퍼를 부드럽게 목표값으로 이동시키는 함수 (입력은 RAW 값: 500~3300)
    def move_gripper(target_raw, duration=1.5):
        print(f"🎯 그리퍼 목표 모터값(RAW): {target_raw}")
        
        # 현재 값도 DEGREES로 읽어오므로 RAW로 역변환
        start_deg = float(follower.get_observation()["gripper.pos"])
        start_raw = start_deg * (4096.0 / 360.0)
        
        steps = int(duration * 30) # 30Hz
        
        for step in range(1, steps + 1):
            action = {}
            # 팔 관절은 현재 위치 그대로 유지 (떨림 방지)
            current_obs = follower.get_observation()
            for m in all_m:
                if "gripper" not in m:
                    action[f"{m}.pos"] = float(current_obs[f"{m}.pos"])
            
            # RAW 보간 후 DEGREES로 변환
            cur_raw = start_raw + (target_raw - start_raw) * (step / steps)
            cur_deg = cur_raw * (360.0 / 4096.0)
            
            action["gripper.pos"] = float(cur_deg)
            
            follower.send_action(RobotAction(action))
            time.sleep(1.0 / 30.0)
            
    try:
        for i in range(3):
            print(f"\n[테스트 {i+1}/3]")
            
            print("👐 최대 열림 (500)으로 이동...")
            move_gripper(500, duration=1.5)
            time.sleep(2.0) # 다 열고 2초 대기
            
            print("✊ 최대 닫힘 (3300)으로 이동...")
            move_gripper(3300, duration=1.5)
            time.sleep(2.0) # 다 닫고 2초 대기
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 강제 중지됨.")
        
    print("\n✅ 그리퍼 테스트 완료.")
    follower.disconnect()

if __name__ == "__main__":
    main()
