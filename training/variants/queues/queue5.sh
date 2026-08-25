#!/bin/bash
cd ~/work/smolvla/variants
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot
python make_se3_dataset.py --out ~/work/smolvla/datasets/se3_relwp_lat --latency-aug > logs/build_se3_lat.log 2>&1 || { echo BUILD_FAIL; exit 1; }
NAME=V7_se3_lataug
echo "=== START $NAME $(date)"
TRAIN_ENTRY="python $HOME/work/smolvla/variants/train_se3.py" ./run_variant.sh $NAME ~/work/smolvla/datasets/se3_relwp_lat '{"observation.images.top":"observation.images.camera1"}' 2 20000 --policy.freeze_vision_encoder=false --policy.train_expert_only=false > logs/train_$NAME.log 2>&1 || echo "TRAIN FAIL $NAME"
./eval_variant.sh $NAME abs 0 se3relwp > logs/eval_$NAME.log 2>&1
echo QUEUE5_DONE $(date)
