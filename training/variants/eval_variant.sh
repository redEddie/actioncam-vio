#!/bin/bash
# usage: eval_variant.sh <name> <state_mode> <prev_image 0|1>   -> preds/var_<name>_stepN.npz for every checkpoint
NAME=$1; SM=$2; PI=$3; AM=${4:-incr}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; export HF_HUB_OFFLINE=1
cd ~/work/eval; shopt -s nullglob
for d in ~/work/smolvla/variants/runs/$NAME/checkpoints/*/pretrained_model; do
  step=$(basename $(dirname $d)); [ "$step" = "last" ] && continue
  out=preds/var_${NAME}_step$((10#$step)).npz; [ -f "$out" ] && continue
  extra="--action-mode $AM"; [ "$PI" = "1" ] && extra="$extra --prev-image"
  python predict_smolvla.py --ckpt $d --out $out --stride 5 --state-mode $SM $extra 2>&1 | grep -E "saved|Traceback|Error"
done
