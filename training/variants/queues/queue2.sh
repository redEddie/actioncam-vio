#!/bin/bash
# waits for queue.sh, then runs V5 (relative state + relative-waypoint action)
until grep -q QUEUE_DONE ~/work/smolvla/variants/logs_queue.log; do sleep 60; done
cd ~/work/smolvla/variants; STEPS=6000
TOP='{"observation.images.top":"observation.images.camera1"}'
NAME=V5_relvel_relwp
echo "=== START $NAME $(date)"
TRAIN_ENTRY="python $HOME/work/smolvla/variants/train_relwp.py" ./run_variant.sh $NAME ~/work/smolvla/datasets/relvel_relwp "$TOP" 2 $STEPS --policy.freeze_vision_encoder=false --policy.train_expert_only=false > logs/train_$NAME.log 2>&1 || echo "TRAIN FAIL $NAME"
./eval_variant.sh $NAME relvel 0 relwp > logs/eval_$NAME.log 2>&1; echo "=== DONE $NAME $(date)"
echo QUEUE2_DONE $(date)
