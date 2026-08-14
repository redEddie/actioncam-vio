#!/bin/bash
set -euo pipefail

# SmolVLA 10 Hz / chunk-15 / 6D step-to-step incremental training.
# This script is fail-closed: it will not start training until the versioned
# dataset proves timestamp, MAP, resampling, and incremental gates in its
# metadata. Recovery augmentation is intentionally deferred for this clean
# baseline and is not a training blocker. Existing artifacts are never replaced.

PROJECT_ROOT="/home/kimminje/gopro_umi"
PYTHON_BIN="/home/kimminje/miniconda3/envs/gopro_env/bin/python"
DATASET_ROOT="$PROJECT_ROOT/2_dataset/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
OUTPUT_DIR="$PROJECT_ROOT/3_training/smolvla_10hz_chunk15_incremental_baseline_v1"
SOURCE_CHECKPOINT="$PROJECT_ROOT/3_training/smolvla_delta_run_fullft_chunk15_vram_input_v2"
STEPS=20000
BATCH_SIZE=32
EPISODES_ARG=()
EXTRA_TRAIN_ARGS=()
SMOKE_MODE=0

if [[ "${1:-}" == "--smoke" ]]; then
  SMOKE_MODE=1
  STEPS=2
  OUTPUT_DIR="$PROJECT_ROOT/3_training/smolvla_10hz_chunk15_incremental_baseline_v1_smoke_step9d_b32"
  EPISODES_ARG=(--dataset.episodes='[0]')
  EXTRA_TRAIN_ARGS=(--log_freq=1 --save_freq=2)
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--smoke]" >&2
  exit 2
fi

printf 'Resolved training mode: steps=%s batch_size=%s output=%s\n' "$STEPS" "$BATCH_SIZE" "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/3_training/validate_incremental_training_preflight.py" \
  --dataset "$DATASET_ROOT" \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --output "$OUTPUT_DIR"

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; refusing impractical CPU fallback"; assert torch.cuda.device_count() > 1, "configured CUDA device index 1 is unavailable"'

if (( SMOKE_MODE )); then
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" "$PROJECT_ROOT/3_training/measure_smolvla_vram_v2.py" \
      --batch-size 32 \
      --physical-gpu 1
fi

TRAIN_CMD=("$PYTHON_BIN" -m lerobot.scripts.lerobot_train \
  --policy.path="$SOURCE_CHECKPOINT" \
  --policy.repo_id=gopro_umi/smolvla_10hz_chunk15_incremental_baseline_v1 \
  --policy.push_to_hub=false \
  --dataset.repo_id=gopro_umi/smolvla_10hz_chunk15_incremental_baseline_v1 \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --dataset.image_transforms.enable=true \
  --policy.chunk_size=15 \
  --policy.n_action_steps=15 \
  "${EPISODES_ARG[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}")

if (( SMOKE_MODE )); then
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$PROJECT_ROOT/3_training/run_with_nvidia_memory_monitor.py" \
    --physical-gpu 1 \
    --max-process-peak-gib 30 \
    -- "${TRAIN_CMD[@]}"
else
  CUDA_VISIBLE_DEVICES=1 "${TRAIN_CMD[@]}"
fi

if (( SMOKE_MODE )); then
  SMOKE_CHECKPOINT="$OUTPUT_DIR/checkpoints/000002/pretrained_model"
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$PYTHON_BIN" "$PROJECT_ROOT/3_training/validate_smolvla_smoke_checkpoint.py" \
      --checkpoint "$SMOKE_CHECKPOINT" \
      --dataset "$DATASET_ROOT"
fi
