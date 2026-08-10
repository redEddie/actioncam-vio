#!/bin/bash
# ---------------------------------------------------------
# [버전 A] 절대 좌표(Absolute) 학습 파이프라인
# ---------------------------------------------------------
# - 데이터셋: lerobot_dataset_yawfree (절대 좌표 저장됨)
# - 출력경로: smolvla_yawfree_run_fullft
# - 비전 해동, 전체 VLM 해동 적용
# - 청크 사이즈: 50 (기본값)
# ---------------------------------------------------------

CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=gopro_umi/smolvla_yawfree_fullft \
  --policy.push_to_hub=false \
  --policy.input_features='{"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[6]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[6]}}' \
  --dataset.repo_id=gopro_umi/smolvla_yawfree_dataset \
  --dataset.root=/home/kimminje/gopro_umi/2_dataset/lerobot_dataset_yawfree \
  --batch_size=32 \
  --steps=20000 \
  --output_dir=/home/kimminje/gopro_umi/3_training/smolvla_yawfree_run_fullft \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
