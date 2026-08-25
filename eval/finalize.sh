#!/bin/bash
# Final metrics + curves + report build. Usage: finalize.sh <smolvla_step>
set -e
STEP=${1:-20000}
source ~/miniconda3/etc/profile.d/conda.sh; conda activate umi_dp; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
cd ~/work/eval
rm -f results_final results_curves_final; mkdir -p results_final results_curves_final
python compute_metrics.py --pred smolvla_holdout_final=preds/smolvla_step${STEP}.npz umi_dp_ep8=preds/dp_epoch7.npz smolvla_prod_ref=preds/smolvla_prod.npz smolvla_deltaweights_start=preds/smolvla_deltaweights_step0.npz --out results_final 2>&1 | grep -E "^(smol|umi|base|\|)" 
python make_curves.py --out results_curves_final 2>&1 | grep -E "^\*\*|^\| [0-9]"
LOSS=$(grep -E "INFO.*step:" ~/work_logs/train_smolvla.log | tail -1 | sed -E 's/.*loss:([0-9.]+).*/\1/')
cd ~/work/report && python3 build_report.py smol_steps=$STEP smol_loss=$LOSS gen_time="(생성 $(date '+%Y-%m-%d %H:%M'))"
echo FINALIZE_DONE
