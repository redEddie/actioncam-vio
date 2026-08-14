#!/usr/bin/env python3
"""
SmolVLA 10Hz / Chunk-15 / 6D Incremental Delta Training
======================================================
train_delta.sh의 쉘 명령어를 파이썬으로 100% 동일하게 포팅한 단독 실행 스크립트입니다.

사용법:
    # 1. 정식 풀 학습 (20,000 steps, batch size 32)
    python 3_training/train_delta.py

    # 2. 빠른 테스트/스모크 런 (2 steps)
    python 3_training/train_delta.py --smoke

    # 3. 스텝 수 / 배치 사이즈 커스텀 지정
    python 3_training/train_delta.py --steps 10000 --batch-size 16 --gpu 1
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import torch

# 프로젝트 기본 경로
PROJECT_ROOT = Path("/home/kimminje/gopro_umi")
LEROBOT_SRC = Path("/home/kimminje/lerobot/src")

# 기본 데이터셋 및 체크포인트 경로
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "2_dataset" / "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
DEFAULT_SOURCE_CHECKPOINT = PROJECT_ROOT / "3_training" / "smolvla_delta_run_fullft_chunk15_vram_input_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "3_training" / "smolvla_10hz_chunk15_incremental_baseline_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="SmolVLA 10Hz Chunk-15 Delta Action Training")
    parser.add_argument("--steps", type=int, default=20000, help="Total training steps (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--gpu", type=int, default=1, help="Physical CUDA GPU device index to use (default: 1)")
    parser.add_argument("--chunk-size", type=int, default=15, help="Action chunk size (default: 15)")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode (2 steps, episode 0 only)")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for run")
    return parser.parse_args()


def check_cuda(gpu_index: int):
    """CUDA 및 GPU 가용성 확인"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing impractical CPU fallback.")
    if torch.cuda.device_count() <= gpu_index:
        raise RuntimeError(f"Configured CUDA device index {gpu_index} is unavailable (Available count: {torch.cuda.device_count()})")
    print(f"✅ GPU {gpu_index} ({torch.cuda.get_device_name(gpu_index)}) 확인 완료")


def run_preflight_validation(dataset_root: str, checkpoint_path: str, output_dir: str):
    """사전 데이터셋 및 모델 게이트 검증"""
    preflight_script = PROJECT_ROOT / "3_training" / "validate_incremental_training_preflight.py"
    if preflight_script.is_file():
        print("🔍 Preflight 데이터셋 게이트 검증 실행 중...")
        cmd = [
            sys.executable, str(preflight_script),
            "--dataset", dataset_root,
            "--checkpoint", checkpoint_path,
            "--output", output_dir
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("Preflight validation failed. 학습을 중단합니다.")
        print("✅ Preflight 검증 통과!\n")


def main():
    args = parse_args()
    
    # 스모크 모드 설정 조정
    if args.smoke:
        steps = 2
        output_dir = str(PROJECT_ROOT / "3_training" / "smolvla_10hz_chunk15_incremental_baseline_v1_smoke_step9d_b32")
        episodes_arg = ["--dataset.episodes=[0]"]
        extra_train_args = ["--log_freq=1", "--save_freq=2"]
        print("🧪 [SMOKE MODE] 2 스텝 테스트 실행 모드 활성화")
    else:
        steps = args.steps
        output_dir = args.output_dir
        episodes_arg = []
        extra_train_args = []

    print("=" * 60)
    print("🚀 SmolVLA 10Hz Chunk-15 Incremental Delta Training")
    print(f" - Steps         : {steps}")
    print(f" - Batch Size    : {args.batch_size}")
    print(f" - Chunk Size    : {args.chunk_size}")
    print(f" - GPU Index     : {args.gpu}")
    print(f" - Dataset Root  : {args.dataset_root}")
    print(f" - Source Model  : {DEFAULT_SOURCE_CHECKPOINT}")
    print(f" - Output Dir    : {output_dir}")
    print("=" * 60)

    # 1. 하드웨어 및 사전 검증
    check_cuda(args.gpu)
    run_preflight_validation(args.dataset_root, str(DEFAULT_SOURCE_CHECKPOINT), output_dir)

    # 2. LeRobot 학습 CLI 명령어 구성
    train_cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--policy.path={DEFAULT_SOURCE_CHECKPOINT}",
        "--policy.repo_id=gopro_umi/smolvla_10hz_chunk15_incremental_baseline_v1",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=gopro_umi/smolvla_10hz_chunk15_incremental_baseline_v1",
        f"--dataset.root={args.dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={steps}",
        f"--output_dir={output_dir}",
        '--rename_map={"observation.images.top":"observation.images.camera1"}',
        "--policy.empty_cameras=2",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
        "--dataset.image_transforms.enable=true",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.chunk_size}",
    ] + episodes_arg + extra_train_args

    print("\n🏃 실행 명령어:")
    print(" ".join(train_cmd))
    print("-" * 60)

    # 3. 환경 변수 설정 (지정 GPU 및 LeRobot PYTHONPATH 주입)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(LEROBOT_SRC) not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = str(LEROBOT_SRC) + ":" + env.get("PYTHONPATH", "")

    # 4. 학습 서브프로세스 실행
    result = subprocess.run(train_cmd, env=env, cwd=str(PROJECT_ROOT))
    
    if result.returncode == 0:
        print("\n🎉 학습이 성공적으로 완료되었습니다!")
    else:
        print(f"\n❌ 학습 중 에러가 발생했습니다 (Return code: {result.returncode})")
        
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
