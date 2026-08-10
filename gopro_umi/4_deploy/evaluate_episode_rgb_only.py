import sys
import os
from pathlib import Path
import io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=69)
    parser.add_argument('--step-interval', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
    DEPLOY_DIR = PROJECT_ROOT / '4_deploy'

    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'smolvla_cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    sys.path.insert(0, str(DEPLOY_DIR))

    import deploy_smolvla_yawfree as deploy
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(args.device)

    # Load Model (Frozen Vision Version)
    MODEL_PATH = PROJECT_ROOT / '3_training/smolvla_yawfree_run/checkpoints/020000/pretrained_model'
    required = ['config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json']
    missing = [f for f in required if not (MODEL_PATH / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")
    print(f"Loading model from {MODEL_PATH}")
    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(MODEL_PATH),
        preprocessor_overrides={'device_processor': {'device': str(device)}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )

    # Load Parquet dataset
    parquet_file = PROJECT_ROOT / f'2_dataset/lerobot_dataset_yawfree/data/chunk-000/file-{args.episode:03d}.parquet'
    df = pd.read_parquet(parquet_file)
    
    # Filter by episode index
    df_ep = df[df['episode_index'] == args.episode].reset_index(drop=True)
    frames_to_eval = list(range(0, len(df_ep), args.step_interval))
    
    # Storage for single-step
    gt_actions = []
    pred_actions_single_step = []
    
    # Storage for open-loop
    pred_actions_open_loop = []
    current_ol_state = None

    print(f"Evaluating {len(frames_to_eval)} frames from episode {args.episode} (interval {args.step_interval})...")
    
    for i, idx in enumerate(frames_to_eval):
        row = df_ep.iloc[idx]
        
        # Extract Image
        img_bytes = row['observation.images.top']['bytes']
        img = np.array(Image.open(io.BytesIO(img_bytes)))
        # Resize to 256x256
        image_256x256_rgb = cv2.resize(img, (256, 256))
        
        gt_state = np.array(row['observation.state'], dtype=np.float32)
        gt_action = np.array(row['action'], dtype=np.float32)
        
        # single step prediction (using GT state)
        policy.reset()
        pred_action_ss = np.asarray(predict_action(
            {'observation.images.camera1': image_256x256_rgb, 'observation.state': gt_state},
            policy, device, pre, post,
            device.type == 'cuda', task='pick and place the target object', robot_type='so_follower',
        ).squeeze(0).detach().cpu(), dtype=np.float64)
        
        gt_actions.append(gt_action)
        pred_actions_single_step.append(pred_action_ss)
        
        # open-loop prediction (using previous predicted action as state)
        if current_ol_state is None:
            current_ol_state = gt_state
            
        policy.reset()
        pred_action_ol = np.asarray(predict_action(
            {'observation.images.camera1': image_256x256_rgb, 'observation.state': current_ol_state},
            policy, device, pre, post,
            device.type == 'cuda', task='pick and place the target object', robot_type='so_follower',
        ).squeeze(0).detach().cpu(), dtype=np.float64)
        
        pred_actions_open_loop.append(pred_action_ol)
        current_ol_state = np.array(pred_action_ol, dtype=np.float32) # use prediction as next state
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames_to_eval)} frames")
            
    gt_actions = np.array(gt_actions)
    pred_actions_single_step = np.array(pred_actions_single_step)
    pred_actions_open_loop = np.array(pred_actions_open_loop)

    # Compute errors (Single Step)
    xyz_gt = gt_actions[:, :3]
    xyz_pred_ss = pred_actions_single_step[:, :3]
    xyz_err_ss = np.linalg.norm(xyz_pred_ss - xyz_gt, axis=1) * 1000 # m to mm
    
    rp_gt = gt_actions[:, 3:5]
    rp_pred_ss = pred_actions_single_step[:, 3:5]
    rp_err_ss = np.linalg.norm(rp_pred_ss - rp_gt, axis=1) * (180.0 / np.pi) # rad to deg
    
    g_gt = gt_actions[:, 5]
    g_pred_ss = pred_actions_single_step[:, 5]
    g_err_ss = np.abs(g_pred_ss - g_gt)
    
    # Compute errors (Open Loop)
    xyz_pred_ol = pred_actions_open_loop[:, :3]
    xyz_err_ol = np.linalg.norm(xyz_pred_ol - xyz_gt, axis=1) * 1000
    
    # Print Stats
    def print_stats(name, arr):
        print(f"{name} - Mean: {np.mean(arr):.2f}, Median: {np.median(arr):.2f}, P95: {np.percentile(arr, 95):.2f}, Max: {np.max(arr):.2f}")

    print("\n--- Single-Step Error Stats ---")
    print_stats("XYZ Error (mm)", xyz_err_ss)
    print_stats("Roll/Pitch Error (deg)", rp_err_ss)
    print_stats("Gripper Error", g_err_ss)
    
    print("\n--- Open-Loop Error Stats ---")
    print_stats("XYZ Cumulative Error (mm)", xyz_err_ol)

    # Plot
    fig = plt.figure(figsize=(15, 6))
    
    # 3D Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], label='Ground Truth', color='black')
    ax1.plot(xyz_pred_ss[:, 0], xyz_pred_ss[:, 1], xyz_pred_ss[:, 2], label='Pred (Single Step)', color='blue')
    ax1.plot(xyz_pred_ol[:, 0], xyz_pred_ol[:, 1], xyz_pred_ol[:, 2], label='Pred (Open Loop)', color='red', linestyle='--')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # Per-step Error Curves
    ax2 = fig.add_subplot(122)
    ax2.plot(xyz_err_ss, label='XYZ Error (mm) - SS', color='blue')
    ax2.plot(xyz_err_ol, label='XYZ Error (mm) - OL', color='red', linestyle='--')
    ax2_2 = ax2.twinx()
    ax2_2.plot(rp_err_ss, label='Roll/Pitch Error (deg) - SS', color='green')
    
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('XYZ Error (mm)')
    ax2_2.set_ylabel('R/P Error (deg)')
    ax2.set_title('Per-step Error Curves')
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    out_img = DEPLOY_DIR / f'episode_{args.episode}_trajectory_eval_frozen.png'
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"\nSaved trajectory plot to {out_img}")

    # Save backlog to CSV
    log_file = DEPLOY_DIR / f'episode_{args.episode}_trajectory_log.csv'
    with open(log_file, 'w') as f:
        f.write("Frame,GT_X,GT_Y,GT_Z,Pred_SS_X,Pred_SS_Y,Pred_SS_Z,Pred_OL_X,Pred_OL_Y,Pred_OL_Z\n")
        for i in range(len(xyz_gt)):
            f.write(f"{frames_to_eval[i]},"
                    f"{xyz_gt[i,0]:.5f},{xyz_gt[i,1]:.5f},{xyz_gt[i,2]:.5f},"
                    f"{xyz_pred_ss[i,0]:.5f},{xyz_pred_ss[i,1]:.5f},{xyz_pred_ss[i,2]:.5f},"
                    f"{xyz_pred_ol[i,0]:.5f},{xyz_pred_ol[i,1]:.5f},{xyz_pred_ol[i,2]:.5f}\n")
    print(f"Saved numerical trajectory log to {log_file}")

if __name__ == '__main__':
    main()
