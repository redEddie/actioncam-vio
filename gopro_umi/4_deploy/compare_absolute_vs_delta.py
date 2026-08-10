import os
import sys
import glob
import random
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io
import cv2

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'
os.environ['HF_HOME'] = str(PROJECT_ROOT / 'smolvla_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))
sys.path.insert(0, str(DEPLOY_DIR))

from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.0001

def evaluate_model_on_episodes(model_path, is_delta, parquet_paths):
    print(f"\n[{'DELTA' if is_delta else 'ABSOLUTE'}] Loading model from {model_path}...")
    policy = SmolVLAPolicy.from_pretrained(model_path, local_files_only=True).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path),
        preprocessor_overrides={'device_processor': {'device': str(device)}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )
    
    all_ep_results = {}
    
    for p_path in parquet_paths:
        ep_name = Path(p_path).stem
        print(f"  - Evaluating {ep_name}...")
        df = pd.read_parquet(p_path)
        # AI 추론 주기와 동일하게 6프레임 간격으로 전체 에피소드 검증
        indices = list(range(0, len(df)-2, 6))
        
        ep_results = []
        for idx in indices:
            row_t = df.iloc[idx]
            row_t1 = df.iloc[idx+1]
            
            state_t = np.array(row_t["observation.state"], dtype=np.float32)
            state_t1 = np.array(row_t1["observation.state"], dtype=np.float32)
            gt_delta = state_t1 - state_t
            
            img_bytes = row_t["observation.images.top"]["bytes"]
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            
            policy.reset()
            action_tensor = predict_action(
                {'observation.images.camera1': img, 'observation.state': state_t},
                policy, device, pre, post,
                device.type == 'cuda', task='pick and place the target object', robot_type='so_follower',
            )
            
            action = np.asarray(action_tensor.squeeze(0).detach().cpu(), dtype=np.float64)
            
            if is_delta:
                pred_delta = action
            else:
                pred_delta = action - state_t
                
            ep_results.append({
                'step': int(idx),
                'gt_delta': gt_delta.tolist(),
                'pred_delta': pred_delta.tolist(),
            })
        
        all_ep_results[ep_name] = ep_results
        
    del policy
    torch.cuda.empty_cache()
    return all_ep_results

def match_dir(gt_val, pred_val):
    if abs(gt_val) <= THRESHOLD:
        return True
    return (gt_val > 0 and pred_val > 0) or (gt_val < 0 and pred_val < 0)

def compute_metrics(results_abs, results_del):
    abs_match_x = 0; abs_match_y = 0; abs_match_xy_any = 0
    del_match_x = 0; del_match_y = 0; del_match_xy_any = 0
    abs_err_x_sum = 0; abs_err_y_sum = 0
    del_err_x_sum = 0; del_err_y_sum = 0
    
    total = len(results_abs)
    for i in range(total):
        gt = np.array(results_abs[i]['gt_delta'])
        a_del = np.array(results_abs[i]['pred_delta'])
        d_del = np.array(results_del[i]['pred_delta'])
        
        a_mx = match_dir(gt[0], a_del[0]); a_my = match_dir(gt[1], a_del[1])
        d_mx = match_dir(gt[0], d_del[0]); d_my = match_dir(gt[1], d_del[1])
        
        if a_mx: abs_match_x += 1
        if a_my: abs_match_y += 1
        if a_mx or a_my: abs_match_xy_any += 1
        if d_mx: del_match_x += 1
        if d_my: del_match_y += 1
        if d_mx or d_my: del_match_xy_any += 1
        
        abs_err_x_sum += abs(a_del[0]-gt[0])
        abs_err_y_sum += abs(a_del[1]-gt[1])
        del_err_x_sum += abs(d_del[0]-gt[0])
        del_err_y_sum += abs(d_del[1]-gt[1])
        
    return {
        'total': total,
        'abs_match_x': abs_match_x, 'abs_match_y': abs_match_y, 'abs_match_xy_any': abs_match_xy_any,
        'del_match_x': del_match_x, 'del_match_y': del_match_y, 'del_match_xy_any': del_match_xy_any,
        'abs_err_x': abs_err_x_sum, 'abs_err_y': abs_err_y_sum,
        'del_err_x': del_err_x_sum, 'del_err_y': del_err_y_sum
    }

def main():
    dataset_dir = PROJECT_ROOT / "2_dataset/lerobot_dataset_yawfree/data/chunk-000"
    all_parquets = list(dataset_dir.glob("file-*.parquet"))
    if len(all_parquets) < 5:
        print("Not enough parquets found!")
        return
        
    selected_parquets = random.sample(all_parquets, 5)
    selected_names = [p.stem for p in selected_parquets]
    print(f"Selected 5 random episodes: {selected_names}\n")
    
    abs_model = PROJECT_ROOT / "yawfree_server_candidate_new"
    delta_model = PROJECT_ROOT / "yawfree_server_candidate_20260804/3_training/finetuning/delta_weights"
    
    all_abs = evaluate_model_on_episodes(abs_model, False, selected_parquets)
    all_del = evaluate_model_on_episodes(delta_model, True, selected_parquets)
    
    print("\n\n=======================================================")
    print("           EPISODE-BY-EPISODE REPORT                   ")
    print("=======================================================")
    
    overall_metrics = []
    
    for ep_name in selected_names:
        print(f"\n---> Results for Episode: {ep_name}")
        metrics = compute_metrics(all_abs[ep_name], all_del[ep_name])
        overall_metrics.append(metrics)
        t = metrics['total']
        
        print("[1] Axis Direction Match Rate")
        print(f" - ABS   : X {metrics['abs_match_x']}/{t} | Y {metrics['abs_match_y']}/{t}")
        print(f" - DELTA : X {metrics['del_match_x']}/{t} | Y {metrics['del_match_y']}/{t}")
        print("[2] At least ONE Axis Match Rate")
        print(f" - ABS   : {metrics['abs_match_xy_any']}/{t} steps")
        print(f" - DELTA : {metrics['del_match_xy_any']}/{t} steps")
        print("[3] Average Delta Error (mm)")
        print(f" - ABS   : X {metrics['abs_err_x']/t*1000:.3f} mm | Y {metrics['abs_err_y']/t*1000:.3f} mm")
        print(f" - DELTA : X {metrics['del_err_x']/t*1000:.3f} mm | Y {metrics['del_err_y']/t*1000:.3f} mm")
        
    print("\n\n=======================================================")
    print("                GRAND AVERAGE CONCLUSION               ")
    print("=======================================================")
    total_steps = sum(m['total'] for m in overall_metrics)
    
    sum_abs_match_x = sum(m['abs_match_x'] for m in overall_metrics)
    sum_abs_match_y = sum(m['abs_match_y'] for m in overall_metrics)
    sum_abs_match_xy = sum(m['abs_match_xy_any'] for m in overall_metrics)
    
    sum_del_match_x = sum(m['del_match_x'] for m in overall_metrics)
    sum_del_match_y = sum(m['del_match_y'] for m in overall_metrics)
    sum_del_match_xy = sum(m['del_match_xy_any'] for m in overall_metrics)
    
    sum_abs_err_x = sum(m['abs_err_x'] for m in overall_metrics)
    sum_abs_err_y = sum(m['abs_err_y'] for m in overall_metrics)
    sum_del_err_x = sum(m['del_err_x'] for m in overall_metrics)
    sum_del_err_y = sum(m['del_err_y'] for m in overall_metrics)
    
    print(f"Total Steps Tested: {total_steps} (5 Episodes)")
    print(f"Negligible Threshold: {THRESHOLD} m ({THRESHOLD*1000} mm)\n")
    
    print("[1] OVERALL Axis Direction Match Rate")
    print(f" - ABS   : X {sum_abs_match_x}/{total_steps} ({(sum_abs_match_x/total_steps)*100:.1f}%) | Y {sum_abs_match_y}/{total_steps} ({(sum_abs_match_y/total_steps)*100:.1f}%)")
    print(f" - DELTA : X {sum_del_match_x}/{total_steps} ({(sum_del_match_x/total_steps)*100:.1f}%) | Y {sum_del_match_y}/{total_steps} ({(sum_del_match_y/total_steps)*100:.1f}%)")
    
    print("\n[2] OVERALL At least ONE Axis Match Rate")
    print(f" - ABS   : {sum_abs_match_xy}/{total_steps} steps ({(sum_abs_match_xy/total_steps)*100:.1f}%)")
    print(f" - DELTA : {sum_del_match_xy}/{total_steps} steps ({(sum_del_match_xy/total_steps)*100:.1f}%)")
    
    print("\n[3] OVERALL Average Delta Error Magnitude")
    print(f" - ABS   : X {sum_abs_err_x/total_steps*1000:.3f} mm | Y {sum_abs_err_y/total_steps*1000:.3f} mm")
    print(f" - DELTA : X {sum_del_err_x/total_steps*1000:.3f} mm | Y {sum_del_err_y/total_steps*1000:.3f} mm")

if __name__ == '__main__':
    main()
