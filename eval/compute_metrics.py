#!/usr/bin/env python3
"""Aggregate metrics for prediction npz files (holdout vs train split), incl. FAST tokens, DTW, DCT.
Usage: python compute_metrics.py --pred label=path.npz [label=path.npz ...] --out results_dir"""
import argparse, json, sys, os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
import numpy as np
sys.path.insert(0, '/home/jeonchanwook/work/eval')
from eval_common import chunk_metrics, action_std, FastTokenizer, HOLDOUT, TRAIN_EVAL, HORIZON, STATE_NAMES, integrate
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--pred', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--no-fast', action='store_true')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
std = action_std()
q = None
fast = None if args.no_fast else FastTokenizer(scale=std * 3)   # /(3*std) ~ maps q1..q99 into ~[-1,1]

models = {}
for item in args.pred:
    label, path = item.split('=', 1)
    d = np.load(path, allow_pickle=True)
    models[label] = {k: d[k] for k in d.files}

# baselines derived from GT of the first model file
ref = next(iter(models.values()))
def add_baseline(name, fn):
    m = {k: ref[k] for k in ['ep', 't', 's0', 'gt_actions', 'gt_states']}
    m['pred_actions'] = np.stack([fn(ga, s0, i) for i, (ga, s0) in enumerate(zip(ref['gt_actions'], ref['s0']))])
    m['pred_states'] = np.stack([integrate(s0, a) for s0, a in zip(m['s0'], m['pred_actions'])])
    models[name] = m
add_baseline('baseline_zero_motion', lambda ga, s0, i: np.zeros_like(ga))
# repeat the *previous* GT action (needs previous action; approximate with first GT action of chunk -> optimistic const-velocity)
add_baseline('baseline_const_velocity(oracle A0)', lambda ga, s0, i: np.repeat(ga[:1], HORIZON, axis=0))
add_baseline('baseline_dataset_mean_action', lambda ga, s0, i: np.repeat(np.mean(ref['gt_actions'], axis=(0, 1))[None], HORIZON, axis=0))

summary = {}
per_step = {}
scalar_keys = ['action_mse', 'action_nmse', 'traj_pos_rmse_m', 'traj_pos_final_err_m', 'traj_rp_rmse_rad', 'traj_grip_rmse', 'dtw_norm', 'dct_cos_k5', 'dct_cos_k3', 'disp_cos', 'fast_exact', 'fast_norm_edit', 'fast_ntok_pred', 'fast_ntok_gt', 'fast_gt_roundtrip_nmse', 'fast_prebpe_match', 'fast_prebpe_match_nonzero']
for label, m in models.items():
    summary[label] = {}
    for split, eps in [('holdout', HOLDOUT), ('train', TRAIN_EVAL)]:
        sel = np.isin(m['ep'], eps)
        if sel.sum() == 0: continue
        rows = []
        for i in np.where(sel)[0]:
            rows.append(chunk_metrics(m['pred_actions'][i], m['gt_actions'][i], m['s0'][i], std, fast=fast))
        agg = {}
        for k in scalar_keys:
            vals = np.array([r[k] for r in rows if k in r], dtype=float)
            if len(vals): agg[k] = float(np.nanmean(vals)); agg[k + '_median'] = float(np.nanmedian(vals))
        agg['action_mse_dim'] = np.mean([r['action_mse_dim'] for r in rows], axis=0).tolist()
        agg['action_rmse_dim'] = np.sqrt(agg['action_mse_dim']).tolist()
        agg['n_chunks'] = int(len(rows))
        per_step[(label, split)] = np.mean([r['traj_pos_err_per_step'] for r in rows], axis=0)
        agg['traj_pos_err_per_step_m'] = per_step[(label, split)].tolist()
        summary[label][split] = agg
        print(f"{label:45s} {split:8s} n={len(rows):4d} nMSE={agg['action_nmse']:.3f} posRMSE={agg['traj_pos_rmse_m']*100:.2f}cm final={agg['traj_pos_final_err_m']*100:.2f}cm rpRMSE={np.degrees(agg['traj_rp_rmse_rad']):.2f}deg grip={agg['traj_grip_rmse']:.3f} DTW={agg['dtw_norm']:.3f} DCTcos5={agg['dct_cos_k5']:.3f} dispcos={agg['disp_cos']:.3f}" + (f" FASTedit={agg['fast_norm_edit']:.3f} FASTexact={agg['fast_exact']:.3f} FASTpreBPE={agg['fast_prebpe_match_nonzero']:.3f}" if fast else ''))

json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=2)

# markdown table
lines = ["| model | split | n | action nMSE | pos RMSE (cm) | final pos err (cm) | roll/pitch RMSE (deg) | grip RMSE | DTW (norm) | DCT cos (k=5) | disp cos | FAST tok edit | FAST exact | FAST pre-BPE match (nonzero) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for label, s in summary.items():
    for split, a in s.items():
        lines.append(f"| {label} | {split} | {a['n_chunks']} | {a['action_nmse']:.3f} | {a['traj_pos_rmse_m']*100:.2f} | {a['traj_pos_final_err_m']*100:.2f} | {np.degrees(a['traj_rp_rmse_rad']):.2f} | {a['traj_grip_rmse']:.3f} | {a['dtw_norm']:.3f} | {a['dct_cos_k5']:.3f} | {a['disp_cos']:.3f} | {a.get('fast_norm_edit', float('nan')):.3f} | {a.get('fast_exact', float('nan')):.3f} | {a.get('fast_prebpe_match_nonzero', float('nan')):.3f} |")
open(os.path.join(args.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# plots: per-step position error
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for j, split in enumerate(['holdout', 'train']):
    for (label, sp), v in per_step.items():
        if sp != split: continue
        ax[j].plot(np.arange(1, HORIZON + 1) * 0.1, v * 100, marker='o', ms=3, label=label)
    ax[j].set_title(f'{split}: mean TCP position error vs horizon'); ax[j].set_xlabel('horizon (s)'); ax[j].set_ylabel('pos error (cm)'); ax[j].grid(alpha=.3)
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(args.out, 'pos_error_vs_horizon.png'), dpi=130)

# example overlays: first 4 holdout chunks of each real model
real = [l for l in models if not l.startswith('baseline')]
for label in real:
    m = models[label]
    idxs = np.where(np.isin(m['ep'], HOLDOUT))[0][::max(1, len(np.where(np.isin(m['ep'], HOLDOUT))[0]) // 6)][:6]
    fig, axes = plt.subplots(len(idxs), 6, figsize=(18, 2.2 * len(idxs)), squeeze=False)
    tt = np.arange(0, HORIZON + 1) * 0.1
    for r, i in enumerate(idxs):
        for d in range(6):
            a = axes[r, d]
            a.plot(tt, np.concatenate([[m['s0'][i][d]], m['gt_states'][i][:, d]]), 'k-', label='GT')
            a.plot(tt, np.concatenate([[m['s0'][i][d]], m['pred_states'][i][:, d]]), 'r--', label='pred')
            if r == 0: a.set_title(STATE_NAMES[d])
            if d == 0: a.set_ylabel(f"ep{m['ep'][i]} t={m['t'][i]}")
            a.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{label}: predicted vs GT 1.5 s trajectories (holdout)'); fig.tight_layout()
    fig.savefig(os.path.join(args.out, f'traj_overlay_{label}.png'), dpi=110); plt.close(fig)
print('done ->', args.out)
