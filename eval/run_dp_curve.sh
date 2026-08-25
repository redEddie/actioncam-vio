#!/bin/bash
# Evaluate every stashed DP epoch checkpoint (learning curve) on holdout+train anchors
source ~/miniconda3/etc/profile.d/conda.sh; conda activate umi_dp; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd ~/work/eval; mkdir -p preds
for f in ~/work/umi_dp/outputs/dp_umi_gopro_10hz_h16/ckpt_stash/epoch=*.ckpt; do
  ep=$(basename "$f" | sed -E 's/epoch=([0-9]+).*/\1/'); out=preds/dp_epoch$((10#$ep)).npz
  [ -f "$out" ] && continue
  python predict_dp.py --ckpt "$f" --out "$out" --stride 5 2>&1 | grep -E "saved|Error|Traceback" 
done
echo DP_CURVE_DONE
