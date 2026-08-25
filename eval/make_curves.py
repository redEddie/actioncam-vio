#!/usr/bin/env python3
"""Learning-curve plots + markdown from per-checkpoint prediction files.
Usage: python make_curves.py --out results_final"""
import argparse, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, '/home/jeonchanwook/work/eval')
from eval_common import chunk_metrics, action_std, HOLDOUT, TRAIN_EVAL, HORIZON
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
std = action_std()

def evaluate(path):
    d = np.load(path, allow_pickle=True); res = {}
    for split, eps in [('holdout', HOLDOUT), ('train', TRAIN_EVAL)]:
        sel = np.where(np.isin(d['ep'], eps))[0]
        rows = [chunk_metrics(d['pred_actions'][i], d['gt_actions'][i], d['s0'][i], std) for i in sel]
        res[split] = {k: float(np.nanmean([r[k] for r in rows])) for k in ['action_nmse', 'traj_pos_rmse_m', 'traj_pos_final_err_m', 'traj_rp_rmse_rad', 'traj_grip_rmse', 'dtw_norm', 'dct_cos_k5', 'disp_cos']}
    return res

curves = {'umi_dp (epoch)': [], 'smolvla (step)': []}
for f in sorted(glob.glob('/home/jeonchanwook/work/eval/preds/dp_epoch*.npz'), key=lambda s: int(re.findall(r'epoch(\d+)', s)[0])):
    ep = int(re.findall(r'epoch(\d+)', f)[0]); curves['umi_dp (epoch)'].append((ep + 1, evaluate(f)))   # epochs completed
for f in sorted(glob.glob('/home/jeonchanwook/work/eval/preds/smolvla_step*.npz'), key=lambda s: int(re.findall(r'step(\d+)', s)[0])):
    st = int(re.findall(r'step(\d+)', f)[0]); curves['smolvla (step)'].append((st, evaluate(f)))
if os.path.exists('/home/jeonchanwook/work/eval/preds/smolvla_deltaweights_step0.npz'):
    curves['smolvla (step)'].insert(0, (0, evaluate('/home/jeonchanwook/work/eval/preds/smolvla_deltaweights_step0.npz')))
json.dump(curves, open(os.path.join(args.out, 'curves.json'), 'w'), indent=1)

metrics = [('traj_pos_rmse_m', 'TCP pos RMSE over 1.5 s (cm)', 100), ('traj_pos_final_err_m', 'final-point pos error (cm)', 100), ('action_nmse', 'action nMSE (std-normalized)', 1), ('dtw_norm', 'DTW (normalized 6D)', 1), ('dct_cos_k5', 'DCT low-freq cosine (k=5)', 1), ('traj_rp_rmse_rad', 'roll/pitch RMSE (deg)', 180 / np.pi)]
fig, axes = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 7))
for r, (name, pts) in enumerate(curves.items()):
    if not pts: continue
    x = [p[0] for p in pts]
    for c, (k, title, scale) in enumerate(metrics):
        a = axes[r, c]
        for split, ls in [('holdout', '-o'), ('train', '--s')]:
            a.plot(x, [p[1][split][k] * scale for p in pts], ls, ms=4, label=split)
        a.set_title(f'{name}: {title}', fontsize=9); a.grid(alpha=.3); a.set_xlabel('epochs' if 'epoch' in name else 'steps')
        if k == 'action_nmse': a.set_yscale('log')
    axes[r, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'learning_curves.png'), dpi=120)

lines = []
for name, pts in curves.items():
    if not pts: continue
    lines += [f"\n**{name}**\n", "| progress | split | pos RMSE (cm) | final err (cm) | action nMSE | roll/pitch RMSE (deg) | grip RMSE | DTW | DCT cos k5 | disp cos |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x, res in pts:
        for split in ['holdout', 'train']:
            m = res[split]
            lines.append(f"| {x} | {split} | {m['traj_pos_rmse_m']*100:.2f} | {m['traj_pos_final_err_m']*100:.2f} | {m['action_nmse']:.3f} | {np.degrees(m['traj_rp_rmse_rad']):.2f} | {m['traj_grip_rmse']:.3f} | {m['dtw_norm']:.3f} | {m['dct_cos_k5']:.3f} | {m['disp_cos']:.3f} |")
open(os.path.join(args.out, 'curves.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
