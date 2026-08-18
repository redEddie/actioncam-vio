"""STEP 3B Matched-Seed Repeated Vision Sensitivity Test Runner."""

import json
import base64
import io
import shlex
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Authoritative Fixed Robot State
S_FIXED = np.array([0.3185, -0.0105, 0.0558, -82.34, -3.03, 1.0], dtype=np.float32)

def main():
    print("=======================================================")
    print(" STEP 3B MATCHED-SEED REPEATED VISION SENSITIVITY TEST")
    print(" (10 Seeds x 5 Conditions = 50 Inferences, Motor = 0)")
    print("=======================================================\n")

    # Load 5 test images
    img_names = ["step3_A_center.png", "step3_B_left.png", "step3_C_right.png", "step3_D_nocup.png", "step3_E_black.png"]
    images_b64 = {}
    for img_name in img_names:
        p = PROJECT_ROOT / img_name
        if not p.exists():
            raise FileNotFoundError(f"{img_name} not found in project root")
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        images_b64[img_name.replace("step3_", "").replace(".png", "")] = base64.b64encode(buf.getvalue()).decode("ascii")

    seeds = list(range(100, 110))
    conditions = ["A_center", "B_left", "C_right", "D_nocup", "E_black"]

    # Remote worker script
    remote_worker_code = '''
import json, base64, io, sys, os, random
protocol_stdout = sys.stdout
sys.stdout = sys.stderr

import torch
import numpy as np
from PIL import Image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

checkpoint = "/home/kimminje/gopro_umi/7_storage/datasets/202608161903/04_training/final/pretrained_model"
policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_config, local_files_only=True).to("cuda")
pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
policy.eval()

# Ready handshake
print(json.dumps({"status": "ready"}), file=protocol_stdout, flush=True)

for line in sys.stdin:
    if not line.strip(): continue
    req = json.loads(line)
    if req.get("cmd") == "exit": break
    
    seed = req["seed"]
    cond = req["condition"]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))))
    state = np.asarray(req["state"], dtype=np.float32)

    # Set all random seeds before inference
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    obs = {
        "observation.images.camera1": torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0,
        "observation.state": torch.from_numpy(state).unsqueeze(0).cuda(),
        "task": ["pick and place the target object"],
        "robot_type": ["so_follower"],
    }
    with torch.inference_mode():
        actions = post(policy.predict_action_chunk(pre(obs)))
    
    res = {
        "seed": seed,
        "condition": cond,
        "action": actions[0].detach().cpu().numpy().tolist()
    }
    print(json.dumps(res), file=protocol_stdout, flush=True)
'''

    encoded_worker = base64.b64encode(remote_worker_code.encode("utf-8")).decode("ascii")
    py_cmd = f"import base64; exec(base64.b64decode('{encoded_worker}'))"

    cmd = [
        "ssh", "-p", "17970", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "kimminje@155.230.189.77",
        "/home/kimminje/miniconda3/envs/gopro_env/bin/python", "-u", "-c",
        shlex.quote(py_cmd)
    ]

    print("서버에 SSH 접속 및 SmolVLA 모델 초기화 중...")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    ready_line = proc.stdout.readline()
    try:
        ready_data = json.loads(ready_line)
    except Exception as e:
        err = proc.stderr.read()
        raise RuntimeError(f"Worker handshake failed on line: {repr(ready_line)}, stderr: {err}") from e

    if ready_data.get("status") != "ready":
        err = proc.stderr.read()
        raise RuntimeError(f"Worker startup failed: {ready_line}, err: {err}")

    print(" -> 모델 초기화 완료. 50회 Matched-Seed 추론을 시작합니다.\n")

    # Store all raw actions: dict[seed][cond] -> np.ndarray [15, 6]
    all_actions = {s: {} for s in seeds}

    count = 0
    for seed in seeds:
        for cond in conditions:
            count += 1
            req = {
                "seed": seed,
                "condition": cond,
                "image_b64": images_b64[cond],
                "state": S_FIXED.tolist(),
            }
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()

            resp_line = proc.stdout.readline()
            resp = json.loads(resp_line)
            act = np.array(resp["action"], dtype=np.float64)
            all_actions[seed][cond] = act
            print(f"[{count:02d}/50] Seed {seed} | Condition {cond:<10} -> Action [15, 6] 수신 완료")

    proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
    proc.stdin.flush()
    proc.terminate()

    # Save all numpy arrays
    results_dir = PROJECT_ROOT / "step3b_actions"
    results_dir.mkdir(exist_ok=True)
    for seed in seeds:
        for cond in conditions:
            np.save(str(results_dir / f"seed{seed}_{cond}.npy"), all_actions[seed][cond])

    print(f"\n모든 액션 텐서가 {results_dir.name}/ 에 저장되었습니다.")
    print("==================================================")
    print(" 통계 분석을 수행합니다...")
    print("==================================================\n")

    # 1. Within-condition variability (Across 10 seeds for each condition)
    print("1. WITHIN-CONDITION VARIABILITY (10 Seeds)")
    print(f"{'Condition':<12} | {'XYZ MAE':>9} | {'XYZ STD':>9} | {'Full6D MAE':>11} | {'Full6D STD':>11}")
    print("-" * 65)
    within_variations = {}
    for cond in conditions:
        arrs = [all_actions[s][cond] for s in seeds] # 10 of [15, 6]
        arr_stack = np.stack(arrs, axis=0) # [10, 15, 6]
        # Pairwise differences among 10 runs
        diffs = []
        for i in range(10):
            for j in range(i+1, 10):
                diffs.append(np.abs(arrs[i] - arrs[j]))
        diffs = np.stack(diffs, axis=0)
        xyz_mae = diffs[:, :, :3].mean()
        full_mae = diffs.mean()
        xyz_std = arr_stack[:, :, :3].std(axis=0).mean()
        full_std = arr_stack.std(axis=0).mean()
        within_variations[cond] = {"xyz_mae": xyz_mae, "full_mae": full_mae, "full_std": full_std}
        print(f"{cond:<12} | {xyz_mae:>9.5f} | {xyz_std:>9.5f} | {full_mae:>11.5f} | {full_std:>11.5f}")

    # 2. Cumulative XYZ (mean +- std across 10 seeds)
    print("\n2. CUMULATIVE XYZ (1.5s horizon, mean ± std)")
    print(f"{'Condition':<12} | {'ΣdX (m)':>19} | {'ΣdY (m)':>19} | {'ΣdZ (m)':>19} | {'Gripper Total':>19}")
    print("-" * 85)
    cum_data = {}
    for cond in conditions:
        cum_sums = np.stack([all_actions[s][cond].sum(axis=0) for s in seeds], axis=0) # [10, 6]
        mean_c = cum_sums.mean(axis=0)
        std_c = cum_sums.std(axis=0)
        cum_data[cond] = {"sums": cum_sums, "mean": mean_c, "std": std_c}
        print(f"{cond:<12} | {mean_c[0]:>8.5f} ± {std_c[0]:>7.5f} | {mean_c[1]:>8.5f} ± {std_c[1]:>7.5f} | {mean_c[2]:>8.5f} ± {std_c[2]:>7.5f} | {mean_c[5]:>8.5f} ± {std_c[5]:>7.5f}")

    # 3. Paired matched-seed differences
    print("\n3. PAIRED MATCHED-SEED DIFFERENCES (10 Seeds, mean ± std)")
    pairs = [
        ("Center vs Left", "A_center", "B_left"),
        ("Center vs Right", "A_center", "C_right"),
        ("Left vs Right", "B_left", "C_right"),
        ("Center vs No Cup", "A_center", "D_nocup"),
        ("Center vs Black", "A_center", "E_black"),
    ]
    print(f"{'Comparison':<18} | {'XYZ MAE':>19} | {'Full6D MAE':>19} | {'Cosine Sim':>19}")
    print("-" * 85)
    paired_results = {}
    for label, c1, c2 in pairs:
        xyz_maes = []
        full_maes = []
        cos_sims = []
        for s in seeds:
            a1 = all_actions[s][c1]
            a2 = all_actions[s][c2]
            xyz_maes.append(np.abs(a1[:, :3] - a2[:, :3]).mean())
            full_maes.append(np.abs(a1 - a2).mean())
            f1, f2 = a1.flatten(), a2.flatten()
            cos_sims.append(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-12))
        
        xyz_m, xyz_s = np.mean(xyz_maes), np.std(xyz_maes)
        full_m, full_s = np.mean(full_maes), np.std(full_maes)
        cos_m, cos_s = np.mean(cos_sims), np.std(cos_sims)
        paired_results[label] = {"xyz": (xyz_m, xyz_s), "full": (full_m, full_s), "cos": (cos_m, cos_s)}
        print(f"{label:<18} | {xyz_m:>8.5f} ± {xyz_s:>7.5f} | {full_m:>8.5f} ± {full_s:>7.5f} | {cos_m:>8.5f} ± {cos_s:>7.5f}")

    # 4. Lateral Steering Consistency (ΔΣdY relative to Center)
    print("\n4. LATERAL STEERING CONSISTENCY (Matched Seed ΔΣdY)")
    left_dy = [all_actions[s]["B_left"].sum(axis=0)[1] - all_actions[s]["A_center"].sum(axis=0)[1] for s in seeds]
    right_dy = [all_actions[s]["C_right"].sum(axis=0)[1] - all_actions[s]["A_center"].sum(axis=0)[1] for s in seeds]
    
    left_pos = sum(1 for v in left_dy if v > 0)
    left_neg = sum(1 for v in left_dy if v <= 0)
    right_pos = sum(1 for v in right_dy if v > 0)
    right_neg = sum(1 for v in right_dy if v <= 0)

    print(f"LEFT vs CENTER (Left steering = +dY expected):")
    print(f" - ΔΣdY values: {[round(v, 4) for v in left_dy]}")
    print(f" - ΔΣdY positive (+Y, left): {left_pos}/10")
    print(f" - ΔΣdY negative (-Y, right): {left_neg}/10")
    print(f" - Mean ΔΣdY: {np.mean(left_dy):.5f} ± {np.std(left_dy):.5f} m")

    print(f"\nRIGHT vs CENTER (Right steering = -dY expected):")
    print(f" - ΔΣdY values: {[round(v, 4) for v in right_dy]}")
    print(f" - ΔΣdY positive (+Y, left): {right_pos}/10")
    print(f" - ΔΣdY negative (-Y, right): {right_neg}/10")
    print(f" - Mean ΔΣdY: {np.mean(right_dy):.5f} ± {np.std(right_dy):.5f} m")

    # 5. Gripper Sensitivity (A0 dGripper)
    print("\n5. GRIPPER SENSITIVITY (A0 dGripper across 10 seeds)")
    center_g = [all_actions[s]["A_center"][0, 5] for s in seeds]
    nocup_g = [all_actions[s]["D_nocup"][0, 5] for s in seeds]
    black_g = [all_actions[s]["E_black"][0, 5] for s in seeds]
    
    c_gt_nc = sum(1 for s in seeds if all_actions[s]["A_center"][0, 5] > all_actions[s]["D_nocup"][0, 5])
    c_gt_bk = sum(1 for s in seeds if all_actions[s]["A_center"][0, 5] > all_actions[s]["E_black"][0, 5])

    print(f" - Center A0 dGripper: {np.mean(center_g):.5f} ± {np.std(center_g):.5f}")
    print(f" - No Cup A0 dGripper: {np.mean(nocup_g):.5f} ± {np.std(nocup_g):.5f}")
    print(f" - Black  A0 dGripper: {np.mean(black_g):.5f} ± {np.std(black_g):.5f}")
    print(f" - Center > No Cup consistency: {c_gt_nc}/10")
    print(f" - Center > Black  consistency: {c_gt_bk}/10")

    # 6. Noise vs Visual Effect Ratio
    print("\n6. NOISE VS VISUAL EFFECT RATIO")
    center_noise_xyz = within_variations["A_center"]["xyz_mae"]
    center_noise_full = within_variations["A_center"]["full_mae"]
    visual_lr_xyz = paired_results["Left vs Right"]["xyz"][0]
    visual_lr_full = paired_results["Left vs Right"]["full"][0]

    print(f" - Within-condition Sampling Noise (XYZ MAE): {center_noise_xyz:.5f}")
    print(f" - Between-condition Visual Effect (L vs R XYZ MAE): {visual_lr_xyz:.5f}")
    print(f" - XYZ Visual Effect / Noise Ratio: {visual_lr_xyz / (center_noise_xyz + 1e-6):.2f}x")
    print(f" - Full 6D Visual Effect / Noise Ratio: {visual_lr_full / (center_noise_full + 1e-6):.2f}x")

if __name__ == "__main__":
    main()
