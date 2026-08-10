import os
import sys
import copy
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

from lerobot.common.control_utils import prepare_observation_for_inference
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def convert_delta_to_absolute(current_state: np.ndarray, predicted_delta_chunk: np.ndarray) -> np.ndarray:
    predicted_absolute_chunk = current_state + np.cumsum(predicted_delta_chunk, axis=0)
    predicted_absolute_chunk[:, 5] = np.clip(predicted_absolute_chunk[:, 5], 0.0, 1.0)
    return predicted_absolute_chunk

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = PROJECT_ROOT / "yawfree_server_candidate_20260804/3_training/finetuning/delta_weights"
    
    print("[1] Loading model...")
    policy = SmolVLAPolicy.from_pretrained(model_path, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(model_path),
        preprocessor_overrides={'device_processor': {'device': str(device)}},
        postprocessor_overrides={'device_processor': {'device': 'cpu'}},
    )
    
    print("[2] Loading observation...")
    p_path = PROJECT_ROOT / "2_dataset/lerobot_dataset_yawfree/data/chunk-000/file-069.parquet"
    df = pd.read_parquet(p_path)
    row_t = df.iloc[0]
    state_t = np.array(row_t["observation.state"], dtype=np.float32)
    img_bytes = row_t["observation.images.top"]["bytes"]
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    raw_obs = {
        'observation.images.camera1': img,
        'observation.state': state_t
    }
    
    # Preprocess
    obs = copy.copy(raw_obs)
    obs = prepare_observation_for_inference(obs, device, task='pick and place the target object', robot_type='so_follower')
    obs = preprocessor(obs)
    
    print("[3] Running raw network prediction...")
    policy.reset()
    
    with torch.inference_mode(), torch.autocast(device_type=device.type) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False):
        batch = policy._prepare_batch(obs)
        # We know policy.predict_action_chunk doesn't exist in control_utils, but it does in SmolVLAPolicy.
        # But predict_action_chunk updates the queue! Let's check its exact behavior.
        # It's better to call _get_action_chunk if we want absolutely NO queue side effects,
        # but predict_action_chunk is the public one (if it exists).
        # We will use predict_action_chunk and see if it populates the queue.
        # Actually, predict_action_chunk in SmolVLAPolicy source we checked earlier:
        # def predict_action_chunk(self, batch, ...):
        #     self._queues = populate_queues(...)
        #     actions = self._get_action_chunk(batch, noise)
        #     return actions
        
        raw_actions = policy.predict_action_chunk(batch)
        print("RAW NETWORK ACTION SHAPE:", raw_actions.shape)
        
    print("[4] Testing Full Chunk Postprocessing...")
    
    # Method A: Existing single-step reference
    ref_chunk = np.zeros((60, 6), dtype=np.float32)
    for t in range(60):
        # raw_actions shape is [1, 60, 6]. We take [:, t, :] -> [1, 6]
        single_action = raw_actions[:, t, :]
        processed = postprocessor(single_action)
        ref_chunk[t] = processed.squeeze(0).cpu().numpy()
        
    # Method B: Full chunk direct postprocessing
    try:
        # Pass the full [1, 60, 6] chunk
        full_processed = postprocessor(raw_actions)
        full_chunk = full_processed.squeeze(0).cpu().numpy()
        supported_BTD = True
    except Exception as e:
        print("Full chunk postprocessing failed:", e)
        supported_BTD = False
        full_chunk = ref_chunk.copy()
        
    if supported_BTD:
        print("FULL POSTPROCESSED CHUNK SHAPE:", full_chunk.shape)
        
    print("[5] Comparing Reference vs Full Chunk...")
    max_err = np.max(np.abs(ref_chunk - full_chunk))
    mean_err = np.mean(np.abs(ref_chunk - full_chunk))
    print(f"Max Abs Error: {max_err}")
    print(f"Mean Abs Error: {mean_err}")
    
    for i, name in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Gripper']):
        err = np.max(np.abs(ref_chunk[:, i] - full_chunk[:, i]))
        print(f"  {name} max error: {err}")
        
    print("\n[6] Delta Statistics (min / max / mean / std)")
    for i, name in enumerate(['X', 'Y', 'Z', 'Roll', 'Pitch', 'Gripper']):
        data = full_chunk[:, i]
        print(f"  {name:7s}: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}, std={data.std():.6f}")
        
    print("\n[7] First 10 Physical Delta Actions")
    print(f"{'step':>4} | {'dx':>8} | {'dy':>8} | {'dz':>8} | {'droll':>8} | {'dpitch':>8} | {'dgrip':>8}")
    for t in range(10):
        d = full_chunk[t]
        print(f"{t:4d} | {d[0]:8.5f} | {d[1]:8.5f} | {d[2]:8.5f} | {d[3]:8.5f} | {d[4]:8.5f} | {d[5]:8.5f}")
        
    print("\n[8] Cumulative Consistency Test")
    absolute_chunk = convert_delta_to_absolute(state_t, full_chunk)
    print("Delta chunk shape:", full_chunk.shape)
    print("Absolute chunk shape:", absolute_chunk.shape)
    
    print("\nPoints:")
    for t in [0, 1, 2, 9, 29, 59]:
        print(f"  absolute[{t}] = {absolute_chunk[t]}")
        
    consistency_passed = True
    gripper_clipped = False
    
    for k in range(60):
        prev = state_t if k == 0 else absolute_chunk[k-1]
        diff = absolute_chunk[k] - prev
        delta = full_chunk[k]
        
        # Check XYZ Roll Pitch
        if not np.allclose(diff[:5], delta[:5], atol=1e-5):
            consistency_passed = False
            print(f"XYZ/RP Consistency failed at {k}: diff={diff[:5]}, delta={delta[:5]}")
            
        # Check Gripper
        if not np.allclose(diff[5], delta[5], atol=1e-5):
            # Might be clipped
            gripper_clipped = True
            expected_abs = prev[5] + delta[5]
            clipped_abs = np.clip(expected_abs, 0.0, 1.0)
            if not np.isclose(absolute_chunk[k][5], clipped_abs, atol=1e-5):
                consistency_passed = False
                print(f"Gripper Consistency failed at {k}: abs={absolute_chunk[k][5]}, expected_clipped={clipped_abs}")
                
    print(f"Cumulative Consistency: {'PASS' if consistency_passed else 'FAIL'}")
    print(f"Gripper Clipping Observed: {'YES' if gripper_clipped else 'NO'}")
    
    print("\n[9] predict_action_chunk API characteristics:")
    print("public/private: public (but not in control_utils)")
    print("input: Dict[str, Tensor] (prepared batch)")
    print("output: Tensor [B, T, D]")
    print("normalization state: Raw (normalized model space)")
    print("queue side effects: Yes (populates observation queue, but does not consume action queue)")

if __name__ == '__main__':
    main()
