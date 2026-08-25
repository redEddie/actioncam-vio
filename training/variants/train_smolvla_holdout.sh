#!/bin/bash
# SmolVLA training = exact command that training/train_delta.py builds (canonical 10Hz/chunk15/6D incremental
# contract, source checkpoint 7_storage/Delta_Weights (local canonical tree)), plus a fixed hold-out split (episodes 90-98 excluded) so that
# offline trajectory metrics can be computed on unseen demos.  Everything else identical to train_delta.py.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PROJECT=~/GoPro_Umi/gopro_umi
DATASET=$PROJECT/7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1
SRC_CKPT=$PROJECT/7_storage/Delta_Weights
OUT=${OUT:-$HOME/work/smolvla/run_holdout90_$(date +%Y%m%d_%H%M)}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}
EPISODES="[$(seq -s, 0 89)]"
cd $PROJECT
python -m lerobot.scripts.lerobot_train \
  --policy.path=$SRC_CKPT \
  --policy.repo_id=gopro_umi/lerobot_dataset_10hz_chunk15_incremental_baseline_v1 \
  --policy.push_to_hub=false \
  --dataset.repo_id=gopro_umi/lerobot_dataset_10hz_chunk15_incremental_baseline_v1 \
  --dataset.root=$DATASET \
  --dataset.episodes="$EPISODES" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --output_dir=$OUT \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --dataset.image_transforms.enable=true \
  --policy.chunk_size=15 \
  --policy.n_action_steps=15 \
  --save_freq=${SAVE_FREQ:-1000} \
  --log_freq=${LOG_FREQ:-50} \
  --num_workers=${NUM_WORKERS:-4} \
  "$@"
