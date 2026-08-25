#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import csv
import time

# Paths
DEPLOY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEPLOY_DIR.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "Delta_Weights"
ZARR_PATH = (
    PROJECT_ROOT / "artifacts" / "zarr" / "replay_buffer_yawfree.zarr"
)

# Force offline
os.environ["HF_HOME"] = str(PROJECT_ROOT / "etc" / "smolvla_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))
replay_src = PROJECT_ROOT / "etc" / "archive" / "patch_history" / "pipeline"
if str(replay_src) not in sys.path:
    sys.path.insert(0, str(replay_src))
legacy_runner_src = PROJECT_ROOT / "etc" / "archive" / "legacy_deploy" / "runners"
if str(legacy_runner_src) not in sys.path:
    sys.path.insert(0, str(legacy_runner_src))
lerobot_src = PROJECT_ROOT / "etc" / "third_party" / "lerobot" / "src"
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from replay_yawfree_shadow import ZarrV2Array
from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def same_sign(a, b):
    return (a >= 0 and b >= 0) or (a < 0 and b < 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sanity-check", action="store_true", help="Run only 3 anchors total")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-md", type=Path, default=PROJECT_ROOT / "etc" / "reports" / "STEP10B_LOCAL_FULL_MODEL_EPISODE_DIRECTION_EVAL.md")
    parser.add_argument("--report-csv", type=Path, default=PROJECT_ROOT / "etc" / "reports" / "evaluation_results.csv")
    parser.add_argument("--report-json", type=Path, default=PROJECT_ROOT / "etc" / "reports" / "evaluation_summary.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Data
    print("Loading Zarr dataset...")
    images = ZarrV2Array(ZARR_PATH / "data/camera0_rgb")
    pos = ZarrV2Array(ZARR_PATH / "data/robot0_eef_pos").all()
    rp = ZarrV2Array(ZARR_PATH / "data/robot0_eef_roll_pitch").all()
    grip = ZarrV2Array(ZARR_PATH / "data/robot0_gripper_width").all()
    ends = ZarrV2Array(ZARR_PATH / "meta/episode_ends").all().astype(int)
    states = np.concatenate((pos, rp, grip), axis=1).astype(np.float32)

    # 2. Load Model
    print(f"Loading policy from {MODEL_PATH} to {device}...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print("Model loaded.")

    horizon = 15
    start = 0
    
    # Statistics storage
    rows = []
    
    episodes_to_eval = min(args.episodes, len(ends))
    print(f"Evaluating episodes 0 to {episodes_to_eval - 1}")
    
    start_time = time.time()
    
    for ep in range(episodes_to_eval):
        end = int(ends[ep])
        valid_anchors = max(0, end - start - horizon)
        
        print(f"Episode {ep}: frames {start} to {end-1} ({end-start} total), valid anchors: {valid_anchors}")
        
        anchor_indices = list(range(start, end - horizon))
        if args.sanity_check:
            anchor_indices = anchor_indices[:min(3, len(anchor_indices))]
            
        for i, anchor_frame in enumerate(anchor_indices):
            S0 = states[anchor_frame].copy()
            rgb = cv2.resize(images.frame(anchor_frame), (256, 256), interpolation=cv2.INTER_AREA)
            
            policy.reset()
            pred_actions = np.zeros((horizon, 6), dtype=np.float64)
            with torch.inference_mode():
                for step_i in range(horizon):
                    act = np.asarray(predict_action(
                        {"observation.images.camera1": rgb, "observation.state": S0},
                        policy, device, pre, post, False,
                        task="pick and place the target object", robot_type="so_follower",
                    ).squeeze(0).detach().cpu(), dtype=np.float64)
                    pred_actions[step_i] = act
            
            for k in range(horizon):
                # Pred incremental
                pred_dx_mm = pred_actions[k, 0] * 1000.0
                pred_dy_mm = pred_actions[k, 1] * 1000.0
                
                # GT incremental
                gt_step_delta = states[anchor_frame + 1 + k] - states[anchor_frame + k]
                gt_dx_mm = gt_step_delta[0] * 1000.0
                gt_dy_mm = gt_step_delta[1] * 1000.0
                
                # X Direction
                if abs(gt_dx_mm) <= 0.1:
                    x_correct = True
                    x_deadzone = True
                else:
                    x_correct = same_sign(pred_dx_mm, gt_dx_mm)
                    x_deadzone = False
                
                # Y Direction
                if abs(gt_dy_mm) <= 0.1:
                    y_correct = True
                    y_deadzone = True
                else:
                    y_correct = same_sign(pred_dy_mm, gt_dy_mm)
                    y_deadzone = False
                    
                # Classify
                if x_correct and y_correct:
                    classification = "BOTH_CORRECT"
                elif x_correct and not y_correct:
                    classification = "X_ONLY"
                elif not x_correct and y_correct:
                    classification = "Y_ONLY"
                else:
                    classification = "BOTH_WRONG"
                    
                # Absolute errors
                x_abs_error_mm = abs(pred_dx_mm - gt_dx_mm)
                y_abs_error_mm = abs(pred_dy_mm - gt_dy_mm)
                
                # Relative errors
                if not x_deadzone:
                    x_rel_err = abs(pred_dx_mm - gt_dx_mm) / abs(gt_dx_mm) * 100.0
                else:
                    x_rel_err = "N/A - DEADZONE"
                    
                if not y_deadzone:
                    y_rel_err = abs(pred_dy_mm - gt_dy_mm) / abs(gt_dy_mm) * 100.0
                else:
                    y_rel_err = "N/A - DEADZONE"
                    
                # Cosine/norm error (optional diagnostic)
                v_pred = pred_actions[k, :3] * 1000.0
                v_gt = gt_step_delta[:3] * 1000.0
                n1 = np.linalg.norm(v_pred)
                n2 = np.linalg.norm(v_gt)
                if n1 < 1e-6 or n2 < 1e-6:
                    cos_sim = 1.0
                else:
                    cos_sim = float(np.dot(v_pred, v_gt) / (n1 * n2))
                    
                rows.append({
                    "episode_id": ep,
                    "anchor_index": anchor_frame,
                    "anchor_timestamp": anchor_frame * 0.1, # approx
                    "chunk_step": k,
                    "future_time_s": 0.1 * (k + 1),
                    "gt_dx_mm": gt_dx_mm,
                    "pred_dx_mm": pred_dx_mm,
                    "gt_dy_mm": gt_dy_mm,
                    "pred_dy_mm": pred_dy_mm,
                    "x_deadzone": x_deadzone,
                    "y_deadzone": y_deadzone,
                    "x_correct": x_correct,
                    "y_correct": y_correct,
                    "classification": classification,
                    "x_abs_error_mm": x_abs_error_mm,
                    "y_abs_error_mm": y_abs_error_mm,
                    "x_relative_error_pct": x_rel_err,
                    "y_relative_error_pct": y_rel_err,
                    "cosine_similarity": cos_sim
                })
                
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"ep {ep} - anchor {i+1}/{len(anchor_indices)} - {len(rows)} total steps eval - {elapsed:.1f}s")
                
            if args.sanity_check and i >= 2:
                break
                
        start = end
        if args.sanity_check and ep >= 0:
            break

    # Summarize Results
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    with open(args.report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    # Computations
    total_steps = len(rows)
    total_anchors = total_steps // 15
    
    def pct(count, total):
        if total == 0: return 0.0
        return (count / total) * 100.0

    def calc_stats(subset):
        if not subset:
            return None
        c_both = sum(1 for r in subset if r["classification"] == "BOTH_CORRECT")
        c_x = sum(1 for r in subset if r["classification"] == "X_ONLY")
        c_y = sum(1 for r in subset if r["classification"] == "Y_ONLY")
        c_wrong = sum(1 for r in subset if r["classification"] == "BOTH_WRONG")
        tot = len(subset)
        return {
            "total": tot,
            "both_correct_count": c_both,
            "both_correct_pct": pct(c_both, tot),
            "x_only_count": c_x,
            "x_only_pct": pct(c_x, tot),
            "y_only_count": c_y,
            "y_only_pct": pct(c_y, tot),
            "both_wrong_count": c_wrong,
            "both_wrong_pct": pct(c_wrong, tot),
            "x_accuracy_pct": pct(c_both + c_x, tot),
            "y_accuracy_pct": pct(c_both + c_y, tot),
            "both_axis_accuracy_pct": pct(c_both, tot)
        }

    global_stats = calc_stats(rows)
    
    ep_stats = {}
    for ep in range(episodes_to_eval):
        ep_rows = [r for r in rows if r["episode_id"] == ep]
        if ep_rows:
            ep_stats[ep] = calc_stats(ep_rows)
            
    ep_means = {}
    if ep_stats:
        metrics = ["both_correct_pct", "x_only_pct", "y_only_pct", "both_wrong_pct", "x_accuracy_pct", "y_accuracy_pct", "both_axis_accuracy_pct"]
        for m in metrics:
            vals = [ep_stats[ep][m] for ep in ep_stats]
            ep_means[m] = {"mean": np.mean(vals), "std": np.std(vals)}
            
    horizon_stats = {}
    for k in range(horizon):
        hk_rows = [r for r in rows if r["chunk_step"] == k]
        if hk_rows:
            horizon_stats[k] = calc_stats(hk_rows)
            
    # Error metrics
    both_correct_rows = [r for r in rows if r["classification"] == "BOTH_CORRECT"]
    x_only_rows = [r for r in rows if r["classification"] == "X_ONLY"]
    y_only_rows = [r for r in rows if r["classification"] == "Y_ONLY"]
    
    error_stats = {
        "both_correct_mean_x_abs_error_mm": np.mean([r["x_abs_error_mm"] for r in both_correct_rows]) if both_correct_rows else 0.0,
        "both_correct_mean_y_abs_error_mm": np.mean([r["y_abs_error_mm"] for r in both_correct_rows]) if both_correct_rows else 0.0,
        "both_correct_median_x_abs_error_mm": np.median([r["x_abs_error_mm"] for r in both_correct_rows]) if both_correct_rows else 0.0,
        "both_correct_median_y_abs_error_mm": np.median([r["y_abs_error_mm"] for r in both_correct_rows]) if both_correct_rows else 0.0,
        
        "x_only_mean_x_abs_error_mm": np.mean([r["x_abs_error_mm"] for r in x_only_rows]) if x_only_rows else 0.0,
        "x_only_median_x_abs_error_mm": np.median([r["x_abs_error_mm"] for r in x_only_rows]) if x_only_rows else 0.0,
        
        "y_only_mean_y_abs_error_mm": np.mean([r["y_abs_error_mm"] for r in y_only_rows]) if y_only_rows else 0.0,
        "y_only_median_y_abs_error_mm": np.median([r["y_abs_error_mm"] for r in y_only_rows]) if y_only_rows else 0.0,
    }
    
    summary = {
        "episodes": list(ep_stats.keys()),
        "total_anchors": total_anchors,
        "total_action_steps": total_steps,
        "global_stats": global_stats,
        "episode_stats": ep_stats,
        "episode_means": ep_means,
        "horizon_stats": horizon_stats,
        "error_stats": error_stats
    }
    
    with open(args.report_json, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Markdown Report Generation
    md_content = ""
    if args.report_md.exists():
        old_content = args.report_md.read_text()
        if "PREVIOUS LIMITED SANITY" not in old_content:
            md_content += "PREVIOUS LIMITED SANITY / DIAGNOSTIC\n15 anchors only\nNOT FINAL DIRECTION EVALUATION\n\n---\n\n"
            md_content += old_content
        else:
            md_content += old_content
            
    md_content += f"""
==================================================
FULL 5-EPISODE 10-HZ X/Y DIRECTION EVALUATION
==================================================

EPISODES:
{list(ep_stats.keys())}

10-HZ VALID ANCHORS:
{total_anchors}

TOTAL ACTION-STEPS:
{total_steps}

DIRECTION DEADZONE:
0.1 mm

--------------------------------
POOLED ALL ACTION-STEPS
--------------------------------

BOTH CORRECT:
{global_stats['both_correct_count']} / {global_stats['total']} = {global_stats['both_correct_pct']:.2f}%

X ONLY:
{global_stats['x_only_count']} / {global_stats['total']} = {global_stats['x_only_pct']:.2f}%

Y ONLY:
{global_stats['y_only_count']} / {global_stats['total']} = {global_stats['y_only_pct']:.2f}%

BOTH WRONG:
{global_stats['both_wrong_count']} / {global_stats['total']} = {global_stats['both_wrong_pct']:.2f}%

X AXIS ACCURACY:
{global_stats['x_accuracy_pct']:.2f}%

Y AXIS ACCURACY:
{global_stats['y_accuracy_pct']:.2f}%

BOTH-AXIS ACCURACY:
{global_stats['both_axis_accuracy_pct']:.2f}%

--------------------------------
5-EPISODE UNWEIGHTED MEAN
--------------------------------

BOTH CORRECT:
mean {ep_means.get('both_correct_pct', {'mean': 0.0})['mean']:.2f}%
std {ep_means.get('both_correct_pct', {'std': 0.0})['std']:.2f}%

X ONLY:
mean {ep_means.get('x_only_pct', {'mean': 0.0})['mean']:.2f}%
std {ep_means.get('x_only_pct', {'std': 0.0})['std']:.2f}%

Y ONLY:
mean {ep_means.get('y_only_pct', {'mean': 0.0})['mean']:.2f}%
std {ep_means.get('y_only_pct', {'std': 0.0})['std']:.2f}%

BOTH WRONG:
mean {ep_means.get('both_wrong_pct', {'mean': 0.0})['mean']:.2f}%
std {ep_means.get('both_wrong_pct', {'std': 0.0})['std']:.2f}%

--------------------------------
A0 IMMEDIATE +0.1s
--------------------------------

BOTH CORRECT: {horizon_stats[0]['both_correct_pct']:.2f}%
X ONLY: {horizon_stats[0]['x_only_pct']:.2f}%
Y ONLY: {horizon_stats[0]['y_only_pct']:.2f}%
BOTH WRONG: {horizon_stats[0]['both_wrong_pct']:.2f}%
X ACCURACY: {horizon_stats[0]['x_accuracy_pct']:.2f}%
Y ACCURACY: {horizon_stats[0]['y_accuracy_pct']:.2f}%

--------------------------------
PER-HORIZON
--------------------------------
"""
    for k in range(horizon):
        st = horizon_stats[k]
        md_content += f"A{k} +{0.1*(k+1):.1f}s: Both {st['both_correct_pct']:.2f}% / X {st['x_only_pct']:.2f}% / Y {st['y_only_pct']:.2f}% / Wrong {st['both_wrong_pct']:.2f}% / X acc {st['x_accuracy_pct']:.2f}% / Y acc {st['y_accuracy_pct']:.2f}%\n"

    md_content += """
--------------------------------
PER-EPISODE
--------------------------------
"""
    for ep, st in ep_stats.items():
        md_content += f"""
EPISODE {ep}:
anchors {st['total'] // 15}
steps {st['total']}
Both {st['both_correct_pct']:.2f}%
X only {st['x_only_pct']:.2f}%
Y only {st['y_only_pct']:.2f}%
Wrong {st['both_wrong_pct']:.2f}%
"""
        
    md_content += f"""
--------------------------------
ERRORS
--------------------------------

Both-correct mean X abs error:
{error_stats['both_correct_mean_x_abs_error_mm']:.3f} mm

Both-correct mean Y abs error:
{error_stats['both_correct_mean_y_abs_error_mm']:.3f} mm

Both-correct median X abs error:
{error_stats['both_correct_median_x_abs_error_mm']:.3f} mm

Both-correct median Y abs error:
{error_stats['both_correct_median_y_abs_error_mm']:.3f} mm

X-only mean X abs error:
{error_stats['x_only_mean_x_abs_error_mm']:.3f} mm

X-only median X abs error:
{error_stats['x_only_median_x_abs_error_mm']:.3f} mm

Y-only mean Y abs error:
{error_stats['y_only_mean_y_abs_error_mm']:.3f} mm

Y-only median Y abs error:
{error_stats['y_only_median_y_abs_error_mm']:.3f} mm
"""

    with open(args.report_md, "w") as f:
        f.write(md_content)
        
    print(f"Reports saved to {args.report_md.parent}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
