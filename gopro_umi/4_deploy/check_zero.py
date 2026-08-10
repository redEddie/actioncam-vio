import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from ik_solver_v7 import DLSInverseKinematicsV7

def main():
    print("=" * 60)
    print("🔍 모터 캘리브레이션 vs URDF 불일치 검사기")
    print("=" * 60)
    
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    
    ik = DLSInverseKinematicsV7(urdf_path="URDF/so_arm_with_gopro_final.urdf")
    
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    obs = follower.get_observation()
    
    print("\n[1] 현재 실제 모터가 읽고 있는 각도:")
    q_rad = []
    for m in arm_m:
        val_deg = float(obs[f"{m}.pos"])
        q_rad.append(np.deg2rad(val_deg))
        print(f"  {m:16s}: {val_deg:+7.1f}°")
        
    pos = ik.get_tcp_position(np.array(q_rad))
    print(f"\n[2] 이 각도를 URDF에 넣었을 때 컴퓨터가 상상하는 로봇의 위치:")
    print(f"  X (앞뒤) : {pos[0]*100:.1f} cm")
    print(f"  Y (좌우) : {pos[1]*100:.1f} cm")
    print(f"  Z (높이) : {pos[2]*100:.1f} cm")
    
    print("\n🚨 진단 결과:")
    if pos[0] > 0.40:
        print("  ▶ 컴퓨터는 지금 팔이 '앞으로 100% 쫙 펴진 상태(44cm)'라고 굳게 믿고 있습니다.")
        print("  ▶ 만약 지금 눈앞의 로봇이 웅크려(수축해) 있다면, 캘리브레이션(영점)이 완전히 틀어진 것입니다!")
        print("  ▶ 컴퓨터가 쫙 펴진 줄 알고 우회전 계산을 하니까, 관절을 엉뚱하게 엄청 크게 돌려서 30cm씩 휙 날아간 것입니다.")
    
    follower.disconnect()

if __name__ == "__main__":
    main()
