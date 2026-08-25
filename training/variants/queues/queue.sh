#!/bin/bash
# Sequential ablation queue (each 6k steps ≈ 50 min). Eval runs after each training.
cd ~/work/smolvla/variants; mkdir -p logs
STEPS=6000
run() { # name ds rename empty state_mode prev_image extra...
  local NAME=$1 DS=$2 RENAME=$3 EMPTY=$4 SM=$5 PI=$6; shift 6
  echo "=== START $NAME $(date)"; ./run_variant.sh $NAME $DS "$RENAME" $EMPTY $STEPS "$@" > logs/train_$NAME.log 2>&1 || echo "TRAIN FAIL $NAME"
  ./eval_variant.sh $NAME $SM $PI > logs/eval_$NAME.log 2>&1; echo "=== DONE $NAME $(date)"
}
TOP='{"observation.images.top":"observation.images.camera1"}'
TOP2='{"observation.images.top":"observation.images.camera1","observation.images.prev":"observation.images.camera2"}'
STRONG='--dataset.image_transforms.max_num_transforms=6 --dataset.image_transforms.random_order=true --dataset.image_transforms.tfs.brightness.kwargs.brightness=[0.6,1.4] --dataset.image_transforms.tfs.contrast.kwargs.contrast=[0.6,1.4] --dataset.image_transforms.tfs.saturation.kwargs.saturation=[0.5,1.5] --dataset.image_transforms.tfs.hue.kwargs.hue=[-0.08,0.08] --dataset.image_transforms.tfs.affine.kwargs.degrees=[-8.0,8.0] --dataset.image_transforms.tfs.affine.kwargs.translate=[0.08,0.08] --dataset.image_transforms.tfs.affine.kwargs.scale=[0.9,1.1]'
run V1_relvel            ~/work/smolvla/datasets/relvel      "$TOP"  2 relvel 0 --policy.freeze_vision_encoder=false --policy.train_expert_only=false
run V2_relvel_previmg    ~/work/smolvla/datasets/relvel_prev "$TOP2" 1 relvel 1 --policy.freeze_vision_encoder=false --policy.train_expert_only=false
run V3_relvel_strongaug  ~/work/smolvla/datasets/relvel      "$TOP"  2 relvel 0 --policy.freeze_vision_encoder=false --policy.train_expert_only=false $STRONG
run V4_relvel_freezevis  ~/work/smolvla/datasets/relvel      "$TOP"  2 relvel 0 --policy.freeze_vision_encoder=true  --policy.train_expert_only=false
echo QUEUE_DONE $(date)
