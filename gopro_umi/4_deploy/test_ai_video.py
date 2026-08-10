import os
os.environ["HF_HOME"] = "/home/kimminje/Desktop/project/gopro_umi/smolvla_cache"

import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    # 영상 경로 설정 (대소문자 처리)
    video_path = "test_videos/test.mp4"
    if not os.path.exists(video_path):
        video_path = "test_videos/test.MP4"
        
    if not os.path.exists(video_path):
        print(f"❌ 영상을 찾을 수 없습니다: {video_path}")
        print("test_videos 폴더에 test.mp4 또는 test.MP4 파일을 넣어주세요!")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 PyTorch Device: {device}")
    
    # 모델 로드
    model_path = "pretrained_model"
    if not os.path.exists(model_path):
        print(f"❌ 모델을 찾을 수 없습니다: {model_path} 폴더가 없습니다.")
        return
        
    print(f"Loading pretrained SmolVLA from {model_path}...")
    policy = SmolVLAPolicy.from_pretrained(model_path).to(device)
    policy.eval()

    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ 비디오 로드 완료: 총 {total_frames} 프레임, {fps:.1f} FPS")

    # 🚨 [수정 완료: 정상화된 ROS 좌표계 세상] 🚨
    # X=0.0, Y=0.0, Z=0.10(책상 위 10cm), Rx, Ry, Rz=0, Gripper=0.5
    # 시작 위치를 마커 정중앙(X=0.0)으로 변경하여 AI가 물체를 향해 움직이는지(음수) 확인
    dummy_state = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)

    print("==================================================")
    print("🎬 영상 기반 AI 예측 테스트 시작 (양방향 좌표계 변환 적용)")
    print("==================================================")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # 로그가 너무 많이 나오는 것을 방지하기 위해 1초에 2~3프레임 정도만 출력
            if frame_idx % int(fps // 2) != 0:
                continue
                
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img_resized = np.array(Image.fromarray(img_rgb).resize((256, 256)))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            state_tensor = torch.from_numpy(dummy_state).unsqueeze(0).to(device)
            
            obs_dict = {
                "observation.images.camera1": img_tensor,
                "observation.state": state_tensor,
                "observation.language.tokens": torch.zeros((1, 1), dtype=torch.long).to(device),
                "observation.language.attention_mask": torch.ones((1, 1), dtype=torch.bool).to(device)
            }
            
            action = policy.select_action(obs_dict)
            action = action.squeeze(0).cpu().numpy()
            
            # 🚨 [수정 완료: AI의 출력값을 순수 ROS 좌표계로 그대로 사용 (7차원)] 🚨
            ros_y = action[1]
            ros_z = action[2]
            ros_grip = action[6]
            
            # 피드백 루프용 (현재 로봇의 위치)
            cur_ros_y = dummy_state[1]
            cur_ros_z = dummy_state[2]
            cur_ros_grip = dummy_state[6]
            
            print(f"[프레임 {frame_idx:04d}/{total_frames}]")
            print(f"  👁️ 입력 상태(로봇 관점): X:{dummy_state[0]:.2f}, Y:{cur_ros_y:.2f}, Z:{cur_ros_z:.2f} | Grip:{cur_ros_grip:.2f}")
            print(f"  🤖 AI 예측(목표 변환): X:{action[0]:.3f}, Y:{ros_y:.3f}, Z:{ros_z:.3f} | Grip:{ros_grip:.2f}")
            print("-" * 50)
            
            # AI의 왜곡된 세상 좌표 그대로 피드백 루프 생성
            noise = np.random.normal(0, 0.005, size=7)
            # 가중치를 줘서 AI가 너무 확 튀는 것을 방지 (부드러운 추종)
            dummy_state = 0.5 * dummy_state + 0.5 * action + noise
            
    cap.release()
    print("✅ 테스트 완료!")

if __name__ == "__main__":
    main()
