#!/bin/bash
# usage: run_variant.sh <name> <dataset_root> <rename_map_json> <empty_cameras> <steps> [extra lerobot_train args...]
set -euo pipefail
NAME=$1; DS=$2; RENAME=$3; EMPTY=$4; STEPS=$5; shift 5
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PROJECT=~/GoPro_Umi/gopro_umi; SRC_CKPT=$PROJECT/7_storage/Delta_Weights
OUT=$HOME/work/smolvla/variants/runs/$NAME; mkdir -p $(dirname $OUT)
EPISODES="[$(seq -s, 0 89)]"
cd $PROJECT
${TRAIN_ENTRY:-python -m lerobot.scripts.lerobot_train} \
  --policy.path=$SRC_CKPT --policy.repo_id=gopro_umi/$NAME --policy.push_to_hub=false \
  --dataset.repo_id=gopro_umi/$NAME --dataset.root=$DS --dataset.episodes="$EPISODES" \
  --batch_size=32 --steps=$STEPS --output_dir=$OUT \
  --rename_map="$RENAME" --policy.empty_cameras=$EMPTY \
  --policy.chunk_size=15 --policy.n_action_steps=15 \
  --dataset.image_transforms.enable=true \
  --save_freq=1000 --log_freq=50 --num_workers=4 "$@"
