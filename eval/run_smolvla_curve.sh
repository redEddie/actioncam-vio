#!/bin/bash
# Evaluate SmolVLA checkpoints (learning curve) + production reference model
source ~/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; export HF_HUB_OFFLINE=1
cd ~/work/eval; mkdir -p preds
RUN=${RUN:-$HOME/work/smolvla/run_holdout90}
[ -f preds/smolvla_prod.npz ] || python predict_smolvla.py --ckpt ~/GoPro_Umi/gopro_umi/7_storage/run/pretrained_model --out preds/smolvla_prod.npz --stride 5 2>&1 | grep -E "saved|Error|Traceback"
[ -f preds/smolvla_deltaweights_step0.npz ] || python predict_smolvla.py --ckpt ~/GoPro_Umi/gopro_umi/7_storage/Delta_Weights --out preds/smolvla_deltaweights_step0.npz --stride 5 2>&1 | grep -E "saved|Error|Traceback"
shopt -s nullglob
for d in $RUN/checkpoints/*/pretrained_model; do [ -d "$d" ] || continue
  step=$(basename $(dirname $d)); [ "$step" = "last" ] && continue
  out=preds/smolvla_step$((10#$step)).npz; [ -f "$out" ] && continue
  python predict_smolvla.py --ckpt $d --out $out --stride 5 2>&1 | grep -E "saved|Error|Traceback"
done
echo SMOLVLA_CURVE_DONE
