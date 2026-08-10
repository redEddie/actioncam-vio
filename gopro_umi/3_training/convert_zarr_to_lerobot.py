"""
Zarr → LeRobot v3.0 Dataset Converter
======================================
우리 UMI Zarr replay buffer (2_dataset/replay_buffer.zarr)를
공식 LeRobot SmolVLA 학습 파이프라인에서 사용할 수 있는
LeRobotDataset v3.0 포맷으로 변환합니다.

Feature 스키마:
  - observation.images.top : (224, 224, 3) uint8 RGB 카메라 이미지
  - observation.state      : (7,) float32 [eef_pos(3) + eef_rot_axis_angle(3) + gripper_norm(1)]
  - action                 : (7,) float32 [eef_pos(3) + eef_rot_axis_angle(3) + gripper_norm(1)]

그리퍼 값:
  - Zarr builder가 gripper_width.csv의 width_norm을 저장하므로 이미 [0, 1]로 정규화된 값이다.
  - LeRobot 변환 단계에서 미터 단위 정규화를 다시 적용하지 않는다.
"""

import os
import sys
import shutil
import zarr
import numpy as np
from pathlib import Path

# LeRobot 라이브러리 경로 추가
LEROBOT_PATH = Path("/home/kimminje/lerobot/src")
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================================
# 설정 상수
# ============================================================================
ZARR_PATH = Path(__file__).resolve().parent.parent / "2_dataset" / "replay_buffer.zarr"
DATASET_REPO_ID = "gopro_umi/smolvla_dataset"
DATASET_ROOT = Path(__file__).resolve().parent.parent / "2_dataset" / "lerobot_dataset"
# All 56 source MP4s report 59.94006 FPS. LeRobot metadata uses the nominal
# integer frame rate while preserving every source frame.
FPS = 60
TASK_DESCRIPTION = "pick and place the target object"


def validate_gripper_norm(width_norm: np.ndarray) -> np.ndarray:
    """Validate the already-normalized gripper signal without rescaling it."""
    values = np.asarray(width_norm, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("robot0_gripper_width contains NaN/inf")
    if values.size == 0 or values.min() < -1e-6 or values.max() > 1.0 + 1e-6:
        raise ValueError("robot0_gripper_width must contain normalized values in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def main():
    print("=" * 60)
    print("🔄 Zarr → LeRobot v3.0 Dataset Conversion")
    print("=" * 60)

    # 1. Zarr 데이터셋 로드
    if not ZARR_PATH.exists():
        print(f"❌ Zarr 데이터셋을 찾을 수 없습니다: {ZARR_PATH}")
        sys.exit(1)

    root = zarr.open(str(ZARR_PATH), mode='r')
    if root.attrs.get("pose_frame") != "robot_base_tcp_link_pose":
        raise ValueError("Zarr pose_frame must be robot_base_tcp_link_pose")
    if root.attrs.get("coordinate_transforms_applied_by_builder") != "none":
        raise ValueError("Zarr builder must not apply an additional coordinate transform")
    images = root['data/camera0_rgb']
    eef_pos = root['data/robot0_eef_pos'][:]       # (N, 3) float32
    eef_rot = root['data/robot0_eef_rot_axis_angle'][:]  # (N, 3) float32
    gripper = root['data/robot0_gripper_width'][:]  # (N, 1) float32
    episode_ends = root['meta/episode_ends'][:]     # (56,) int64

    total_frames = images.shape[0]
    num_episodes = len(episode_ends)
    lengths = (len(eef_pos), len(eef_rot), len(gripper))
    if lengths != (total_frames, total_frames, total_frames):
        raise ValueError(f"Zarr array length mismatch: images={total_frames}, arrays={lengths}")
    if images.shape[1:] != (224, 224, 3) or eef_pos.shape[1:] != (3,) or eef_rot.shape[1:] != (3,):
        raise ValueError("unexpected Zarr image/pose schema")
    if num_episodes != 56 or episode_ends[-1] != total_frames or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("expected 56 non-empty episodes ending at the final Zarr frame")

    print(f"📊 Zarr 로드 완료:")
    print(f"   - 총 에피소드: {num_episodes}")
    print(f"   - 총 프레임: {total_frames}")
    print(f"   - 이미지 Shape: {images.shape}")

    # 2. Zarr에 저장된 width_norm을 중복 정규화 없이 검증한다.
    gripper_norm = validate_gripper_norm(gripper)

    # 3. 7D state/action 벡터 구성: eef_pos(3) + eef_rot(3) + gripper_norm(1)
    states_7d = np.concatenate([
        eef_pos,
        eef_rot,
        gripper_norm.reshape(-1, 1)
    ], axis=-1).astype(np.float32)  # (N, 7)

    print(f"   - State 7D Shape: {states_7d.shape}")
    print(f"   - Gripper Norm Range: [{gripper_norm.min():.4f}, {gripper_norm.max():.4f}]")

    # 4. 기존 LeRobot 데이터셋 폴더 정리
    if DATASET_ROOT.exists():
        print(f"🗑️ 기존 LeRobot 데이터셋 삭제: {DATASET_ROOT}")
        shutil.rmtree(DATASET_ROOT)

    # 5. LeRobotDataset 생성
    print(f"\n🚀 LeRobot v3.0 데이터셋 생성 시작...")
    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "axes": ["x", "y", "z", "rx", "ry", "rz", "gripper"]
            },
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features=features,
        robot_type="umi_tcp",
        root=DATASET_ROOT,
        use_videos=False,  # 이미지 모드 사용 (224x224 RGB)
    )

    # 6. 에피소드별 프레임 추가
    ep_start = 0
    for ep_idx, ep_end in enumerate(episode_ends):
        ep_len = ep_end - ep_start
        print(f"⏳ Episode {ep_idx + 1}/{num_episodes} 변환 중... ({ep_len} frames)")

        for t in range(ep_start, ep_end):
            # 현재 프레임의 state
            state = states_7d[t]

            # action = 다음 프레임의 state (마지막 프레임은 현재 state 반복)
            if t + 1 < ep_end:
                action = states_7d[t + 1]
            else:
                action = states_7d[t]

            # 이미지 로드 (Zarr에서 한 프레임씩)
            img = images[t]  # (224, 224, 3) uint8

            from PIL import Image
            img_pil = Image.fromarray(img)

            dataset.add_frame({
                "observation.images.top": img_pil,
                "observation.state": state,
                "action": action,
                "task": TASK_DESCRIPTION,
            })

        dataset.save_episode()
        ep_start = ep_end

    print(f"\n🎉 LeRobot v3.0 데이터셋 변환 완료!")
    print(f"📂 저장 위치: {DATASET_ROOT}")
    print(f"📊 총 {num_episodes}개 에피소드, {total_frames} 프레임")
    training_wrapper = Path(__file__).resolve().parent / "train_smolvla.py"
    print(f"\n💡 학습 전 검증 명령어:")
    print(f"   python {training_wrapper} --validate_only")
    print(f"\n💡 학습 명령어:")
    print(f"   python {training_wrapper} --steps 20000 --batch_size 32")


if __name__ == "__main__":
    main()
