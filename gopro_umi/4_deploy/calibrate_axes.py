import os
import cv2
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["HF_HOME"] = "/home/kimminje/Desktop/project/gopro_umi/smolvla_cache"

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    print("==================================================")
    print("🛠️ AI 좌표계 <-> 로봇 좌표계 영점 캘리브레이션 🛠️")
    print("==================================================")

    video_path = "test_videos/test.MP4"
    if not os.path.exists(video_path):
        video_path = "test_videos/test.mp4"
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("1. 모델 로딩 중...")
    model_path = "pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(model_path).to(device)
    policy.eval()

    print("2. 영상 첫 프레임 추출 중...")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ 영상을 읽을 수 없습니다.")
        return

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = np.array(Image.fromarray(img_rgb).resize((256, 256)))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    dummy_state = np.zeros(6, dtype=np.float32)
    state_tensor = torch.from_numpy(dummy_state).unsqueeze(0).to(device)
    
    obs_dict = {
        "observation.images.camera1": img_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": torch.zeros((1, 1), dtype=torch.long).to(device),
        "observation.language.attention_mask": torch.ones((1, 1), dtype=torch.bool).to(device)
    }
    
    print("3. AI 단일 프레임 추론 중...")
    with torch.no_grad():
        action = policy.select_action(obs_dict)
        action = action.squeeze(0).cpu().numpy()
        
    print("\n==================================================")
    print("🎯 [결과] AI가 생각하는 영상 첫 장면의 절대 좌표 (마커 기준)")
    print("==================================================")
    print(f"X (예측값) : {action[0]:.4f}")
    print(f"Y (예측값) : {action[1]:.4f}")
    print(f"Z (예측값) : {action[2]:.4f}")
    print(f"Grip (그리퍼)   : {action[5]:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
