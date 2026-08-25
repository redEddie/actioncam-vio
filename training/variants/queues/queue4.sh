#!/bin/bash
cd ~/work/smolvla/variants; NAME=V6_se3relwp_strongaug
echo "=== START $NAME $(date)"
TRAIN_ENTRY="python $HOME/work/smolvla/variants/train_se3.py" ./run_variant.sh $NAME ~/work/smolvla/datasets/se3_relwp '{"observation.images.top":"observation.images.camera1"}' 2 20000 --policy.freeze_vision_encoder=false --policy.train_expert_only=false > logs/train_$NAME.log 2>&1 || echo "TRAIN FAIL $NAME"
./eval_variant.sh $NAME abs 0 se3relwp > logs/eval_$NAME.log 2>&1
echo QUEUE4_DONE $(date)
