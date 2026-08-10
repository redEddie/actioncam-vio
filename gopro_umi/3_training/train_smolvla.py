"""
SmolVLA 공식 LeRobot 파이프라인 학습 래퍼 스크립트
==================================================
공식 LeRobot lerobot-train CLI를 감싸는 래퍼입니다.
우리 프로젝트의 데이터셋 경로와 하이퍼파라미터를 자동 설정합니다.

사용법:
    python 3_training/train_smolvla.py
    python 3_training/train_smolvla.py --steps 20000 --batch_size 32
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

# LeRobot 라이브러리 경로
LEROBOT_PATH = Path("/home/kimminje/lerobot/src")

# 프로젝트 기본 설정
DATASET_REPO_ID = "gopro_umi/smolvla_dataset"
DATASET_ROOT = str(Path(__file__).resolve().parent.parent / "2_dataset" / "lerobot_dataset")
OUTPUT_DIR = str(Path(__file__).resolve().parent / "smolvla_run")
POLICY_PATH = "lerobot/smolvla_base"


def main():
    parser = argparse.ArgumentParser(description="SmolVLA Official LeRobot Training Wrapper")
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps (default: 20000)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--policy_path", type=str, default=POLICY_PATH, help="Pretrained policy path")
    parser.add_argument("--dataset_repo_id", type=str, default=DATASET_REPO_ID, help="Dataset repo ID")
    parser.add_argument("--validate_only", action="store_true", help="Validate the local dataset and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 SmolVLA Official LeRobot Training Pipeline")
    print(f" - Policy: {args.policy_path}")
    print(f" - Dataset: {args.dataset_repo_id}")
    print(f" - Dataset Root: {DATASET_ROOT}")
    print(f" - Batch Size: {args.batch_size}")
    print(f" - Steps: {args.steps}")
    print(f" - Output: {OUTPUT_DIR}")
    print("=" * 60)

    # LeRobot 데이터셋 존재 확인
    ds_path = Path(DATASET_ROOT)
    info_path = ds_path / "meta" / "info.json"
    if not info_path.is_file():
        print(f"❌ LeRobot 데이터셋이 없습니다: {ds_path}")
        print("💡 먼저 변환 스크립트를 실행해 주세요:")
        print("   python 3_training/convert_zarr_to_lerobot.py")
        sys.exit(1)
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    expected_shapes = {
        "observation.images.top": [224, 224, 3],
        "observation.state": [7],
        "action": [7],
    }
    bad_shapes = {
        key: features.get(key, {}).get("shape")
        for key, shape in expected_shapes.items()
        if features.get(key, {}).get("shape") != shape
    }
    if info.get("total_episodes") != 56 or info.get("total_frames", 0) <= 0 or bad_shapes:
        print(
            f"❌ LeRobot 데이터셋 schema 검증 실패: "
            f"episodes={info.get('total_episodes')}, frames={info.get('total_frames')}, shapes={bad_shapes}"
        )
        sys.exit(1)
    print(
        f"✅ LeRobot 데이터셋 검증 통과: "
        f"episodes={info['total_episodes']}, frames={info['total_frames']}, state/action=7D"
    )
    if args.validate_only:
        return

    output_path = Path(OUTPUT_DIR)
    if output_path.is_dir() and not any(output_path.iterdir()):
        output_path.rmdir()
    elif output_path.exists():
        print(f"❌ 학습 출력 경로가 이미 존재합니다: {output_path}")
        print("💡 이전 run을 보존/이동하거나 --resume 전용 절차를 사용하세요.")
        sys.exit(1)

    # The pretrained SmolVLA config defaults to 6D ALOHA state/action. Override
    # the complete feature dictionaries so this UMI dataset is explicitly 7D.
    policy_input_features = (
        '{"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},'
        '"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},'
        '"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]},'
        '"observation.state":{"type":"STATE","shape":[7]}}'
    )
    policy_output_features = '{"action":{"type":"ACTION","shape":[7]}}'

    # 공식 lerobot-train CLI 명령어 구성
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path={args.policy_path}",
        "--policy.repo_id=gopro_umi/smolvla_finetuned",
        "--policy.push_to_hub=false",
        f"--policy.input_features={policy_input_features}",
        f"--policy.output_features={policy_output_features}",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={DATASET_ROOT}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={OUTPUT_DIR}",
        "--rename_map={\"observation.images.top\": \"observation.images.camera1\"}",
        "--policy.empty_cameras=2",
    ]

    print(f"\n🏃 실행 명령어:")
    print(f"   {' '.join(cmd)}")
    print()

    # 환경 변수에 LeRobot 경로 추가
    env = os.environ.copy()
    if str(LEROBOT_PATH) not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = str(LEROBOT_PATH) + ":" + env.get("PYTHONPATH", "")

    # 학습 실행
    result = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent.parent))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
