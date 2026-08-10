import os
os.environ["HF_HOME"] = "/home/kimminje/Desktop/project/gopro_umi/smolvla_cache"

import cv2
import sys
import time
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction
from ik_solver_v7 import DLSInverseKinematicsV7
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

URDF_PATH = "URDF/so_arm_with_gopro_final.urdf"

def get_current_state(ik, follower, arm_m, all_m):
    obs = follower.get_observation()
    q = np.array([np.deg2rad(float(obs[f"{m}.pos"])) for m in arm_m])
    
    tcp_pos = ik.get_tcp_position(q)
    tcp_rot = ik.get_tcp_rotation(q)
    euler = R.from_matrix(tcp_rot).as_euler('zyx', degrees=False)
    
    gripper_val_deg = float(obs["gripper.pos"])
    gripper_val_raw = gripper_val_deg * (4096.0 / 360.0)
    
    gripper_norm = (gripper_val_raw - (-27.0)) / (1042.0 - (-27.0))
    gripper_norm = np.clip(gripper_norm, 0.0, 1.0)
    
    state = np.array([
        tcp_pos[0], tcp_pos[1], tcp_pos[2],
        euler[1], euler[2],
        gripper_norm
    ], dtype=np.float32)
    return state, q

def main():
    print("==================================================")
    print("🕵️ AI 예측 격리 테스트 (모터 제어 안 함)")
    print("==================================================")
    print("- 로봇 팔을 직접 손으로 움직이면서 카메라에 물건을 보여주세요.")
    print("- AI가 어떻게 움직이고 싶어하는지 수치로만 확인합니다.")
    print("==================================================")
    
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    
    # 토크 해제 (사람이 손으로 움직일 수 있게 함)
    for m in follower.bus.motors.keys():
        follower.bus.write("Torque_Enable", m, 0)
        
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    all_m = list(follower.bus.motors.keys())
    
    ik = DLSInverseKinematicsV7(urdf_path=URDF_PATH, limit_margin_ratio=0.90)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 PyTorch Device: {device}")
    
    model_path = "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_path).to(device)
    policy.eval()
    
    print("카메라 연결 중...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    for _ in range(5): cap.grab()
        
    print("✅ 준비 완료. 루프 시작...")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 와이드 영상 자르지 않고 원본 프레임을 256x256으로 찌그러뜨려서 압축(Squash)
            img_resized = np.array(Image.fromarray(img_rgb).resize((256, 256)))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            state, cur_q = get_current_state(ik, follower, arm_m, all_m)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            
            obs_dict = {
                "observation.images.camera1": img_tensor,
                "observation.state": state_tensor,
                "observation.language.tokens": torch.zeros((1, 1), dtype=torch.long).to(device),
                "observation.language.attention_mask": torch.ones((1, 1), dtype=torch.bool).to(device)
            }
            
            action = policy.select_action(obs_dict)
            action = action.squeeze(0).cpu().numpy()
            
            # AI 예측값 (6D)
            tx, ty, tz = action[0:3]
            tgrip = action[5]
            
            # 현재값 (6D)
            cx, cy, cz = state[0:3]
            cgrip = state[5]
            
            print(f"👁️ 현재 위치: X:{cx:.2f}, Y:{cy:.2f}, Z:{cz:.2f} | Grip:{cgrip:.2f}")
            print(f"🤖 AI의 목표: X:{tx:.2f}, Y:{ty:.2f}, Z:{tz:.2f} | Grip:{tgrip:.2f}")
            print("-" * 50)
            
            # 절대 모터로 전송하지 않음! (격리)
            time.sleep(0.5)
            
if __name__ == "__main__":
    main()
