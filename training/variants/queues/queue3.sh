#!/bin/bash
cd ~/work/smolvla/variants; STEPS=6000
TOP='{"observation.images.top":"observation.images.camera1"}'
NAME=V3_relvel_strongaug
echo "=== START $NAME $(date)"
TRAIN_ENTRY="python $HOME/work/smolvla/variants/train_strongaug.py" ./run_variant.sh $NAME ~/work/smolvla/datasets/relvel "$TOP" 2 $STEPS --policy.freeze_vision_encoder=false --policy.train_expert_only=false > logs/train_$NAME.log 2>&1 || echo "TRAIN FAIL $NAME"
./eval_variant.sh $NAME relvel 0 > logs/eval_$NAME.log 2>&1
echo QUEUE3_DONE $(date)
