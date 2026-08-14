import time
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.utils import make_robot_from_config

def main():
    print("🤖 로봇 설정을 불러오는 중...")
    # 팔로워 암 설정 객체 생성 (포트는 상황에 맞게 수정 가능)
    config = SO101FollowerConfig(port="/dev/ttyACM0")
    robot = make_robot_from_config(config)

    try:
        print("🔌 로봇과 연결을 시도합니다...")
        # 임시 테스트이므로 캘리브레이션 과정은 생략하고 바로 연결합니다.
        # (주의: 모든 모터 ID가 제대로 세팅되어 있어야 정상 연결됩니다)
        robot.connect(calibrate=False)
        print("✅ 연결 성공!")
        
        # 현재 로봇 관절 상태 읽어오기
        obs = robot.get_observation()
        current_pos = dict(obs)
        
        print(f"현재 관절 각도: {current_pos}")

        print("\n👉 10초 동안 0.5초 간격으로 그리퍼 각도를 +10도로 유지하며 현재 상태를 계속 출력합니다.")
        
        target_pos = dict(current_pos)
        if "gripper.pos" in target_pos:
            target_pos["gripper.pos"] += 10.0
        
        for i in range(20):
            # 목표 각도 전송
            robot.send_action(target_pos)
            
            # 방금 명령을 내린 후의 현재 실제 상태 읽어오기
            obs = robot.get_observation()
            actual_gripper_pos = obs.get("gripper.pos", 0.0)
            
            print(f"[{i+1}/20] 목표 그리퍼: {target_pos['gripper.pos']:.2f} | 현재 실제 그리퍼: {actual_gripper_pos:.2f}")
            time.sleep(0.5)
            
        # 원래 위치로 복귀
        print("\n원래 위치로 복귀합니다.")
        robot.send_action(current_pos)
        time.sleep(1)

    except Exception as e:
        print(f"\n❌ 에러가 발생했습니다: {e}")
        print("모터 6개의 ID 세팅이 모두 끝나지 않았거나 선이 빠져있으면 연결 에러가 날 수 있습니다.")
    finally:
        if hasattr(robot, 'is_connected') and robot.is_connected:
            robot.disconnect()
            print("테스트 안전 종료 완료.")

if __name__ == "__main__":
    main()
