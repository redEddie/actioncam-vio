import time, sys, random
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction
from ik.ik_solver_v7 import DLSInverseKinematicsV7

URDF_PATH = str(Path(__file__).resolve().parents[3] / "urdf/so_arm_with_gopro_final.urdf")
STEP_M = 0.001
LOOP_HZ = 30

def deg2rad(obs, motors):
    return np.array([np.deg2rad(float(obs[f"{m}.pos"])) for m in motors])

def main():
    print("=" * 70)
    print("🎲 IK v7: 랜덤 위치 이동 테스트 (5초 간격, 총 5번)")
    print("=" * 70)

    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    all_m = list(follower.bus.motors.keys())

    ik = DLSInverseKinematicsV7(urdf_path=URDF_PATH, limit_margin_ratio=0.90)

    obs = follower.get_observation()
    cur = deg2rad(obs, arm_m)

    clamped = ik._clamp_joints_safe(cur)
    if np.linalg.norm(clamped - cur) > 1e-6:
        print("⚠️ 관절이 90% 안전 한계 밖에 있어 초기 강제 이동합니다...")
        for step in range(1, 21):
            action = {}
            for i, m in enumerate(arm_m):
                s = float(obs[f"{m}.pos"])
                t = float(np.rad2deg(clamped[i]))
                action[f"{m}.pos"] = s + (t - s) * (step / 20.0)
            for m in all_m:
                if "gripper" in m:
                    action[f"{m}.pos"] = float(obs[f"{m}.pos"])
            follower.send_action(RobotAction(action))
            time.sleep(1.0 / 30.0)
        cur = clamped.copy()
        print("✅ 강제 이동 완료.")

    init_tcp_pos = ik.get_tcp_position(cur)
    init_tcp_rot = ik.get_tcp_rotation(cur)
    
    # 5번 반복
    for i in range(1, 6):
        # 1. 시작 위치 기준 X, Y, Z 각각 최대 +/- 5cm 랜덤 목표 생성
        delta = np.random.uniform(-0.05, 0.05, size=3)
        target_pos = init_tcp_pos + delta
        target_rot = init_tcp_rot.copy()

        print(f"\n[{i}/5] 🎯 랜덤 목표 지정됨: ΔX={delta[0]*100:+.1f}cm, ΔY={delta[1]*100:+.1f}cm, ΔZ={delta[2]*100:+.1f}cm")
        print(f"       목표 절대좌표: X={target_pos[0]:.4f}, Y={target_pos[1]:.4f}, Z={target_pos[2]:.4f}")

        # 이동 경로 생성 (1mm 단위 보간)
        cur_pos = ik.get_tcp_position(cur)
        dist = np.linalg.norm(target_pos - cur_pos)
        steps = max(int(dist / STEP_M), 1)
        move_dir = (target_pos - cur_pos) / dist
        
        # 중간 과정 출력 생략 (실제 로봇 이동만 부드럽게 실행)
        for step in range(1, steps + 1):
            t_pos = cur_pos + move_dir * (dist * step / steps)
            cur = ik.solve(t_pos, target_rot, cur)
            
            action = {}
            for j, m in enumerate(arm_m):
                action[f"{m}.pos"] = float(np.rad2deg(cur[j]))
            
            obs_now = follower.get_observation()
            for m in all_m:
                if "gripper" in m:
                    action[f"{m}.pos"] = float(obs_now[f"{m}.pos"])
            
            follower.send_action(RobotAction(action))
            time.sleep(1.0 / LOOP_HZ)
        
        # 최종 확인 로그 출력
        final_pos = ik.get_tcp_position(cur)
        error = np.linalg.norm(target_pos - final_pos) * 1000
        print(f"       🤖 최종 도달좌표: X={final_pos[0]:.4f}, Y={final_pos[1]:.4f}, Z={final_pos[2]:.4f}")
        print(f"       📏 도달 오차: {error:.2f} mm")
        
        if i < 5:
            print("       ⏳ 5초 대기 중...")
            time.sleep(5.0)

    print("\n✅ 모든 랜덤 위치 이동 테스트 완료.")
    # 로봇의 안전을 위해 서보 모터 토크 오프 (또는 필요시 유지)
    follower.disconnect()

if __name__ == "__main__":
    main()
