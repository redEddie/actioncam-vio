"""STEP 3B Matched-Seed Repeated Vision Sensitivity Test with Fixed Gripper Scale."""

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

# Authoritative Fixed Robot State with Fixed Gripper Scale
S_FIXED = np.array([0.3185, -0.0105, 0.0558, -82.34, -3.03, 0.441305], dtype=np.float32)

def main():
    print("=======================================================")
    print(" STEP 3B MATCHED-SEED REPEATED VISION SENSITIVITY TEST")
    print(" (Fixed Gripper Scale = 0.441305, Motor = 0)")
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

checkpoint = "/home/kimminje/gopro_umi/artifacts/datasets/202608161903/04_training/final/pretrained_model"
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
    results_dir = PROJECT_ROOT / "step3b_fixed_gripper"
    results_dir.mkdir(exist_ok=True)
    for seed in seeds:
        for cond in conditions:
            np.save(str(results_dir / f"seed{seed}_{cond}.npy"), all_actions[seed][cond])

    print(f"\n모든 액션 텐서가 {results_dir.name}/ 에 저장되었습니다.")
    print("==================================================")
    print(" 통계 분석을 수행합니다...")
    print("==================================================\n")

    # 1. Within-Condition Variability (Stochastic Flow Matching noise across seeds)
    print("1. WITHIN-CONDITION VARIABILITY (10 Seeds)")
    print(f"{'Condition':<12} | {'XYZ MAE':>10} | {'XYZ STD':>10} | {'Full6D MAE':>12} | {'Full6D STD':>12}")
    print("-" * 65)
    within_var = {}
    for cond in conditions:
        cond_actions = np.stack([all_actions[s][cond] for s in seeds]) # [10, 15, 6]
        mean_act = cond_actions.mean(axis=0) # [15, 6]
        diffs = cond_actions - mean_act # [10, 15, 6]
        xyz_mae = np.mean(np.abs(diffs[:, :, :3]))
        xyz_std = np.mean(np.std(cond_actions[:, :, :3], axis=0))
        full_mae = np.mean(np.abs(diffs))
        full_std = np.mean(np.std(cond_actions, axis=0))
        within_var[cond] = {"xyz_mae": xyz_mae, "full_mae": full_mae, "xyz_std": xyz_std, "full_std": full_std}
        print(f"{cond:<12} | {xyz_mae:10.5f} | {xyz_std:10.5f} | {full_mae:12.5f} | {full_std:12.5f}")

    # 2. Cumulative XYZ per condition
    print("\n2. CUMULATIVE XYZ (1.5s horizon, mean ± std)")
    print(f"{'Condition':<12} | {'ΣdX (m)':>25} | {'ΣdY (m)':>25} | {'ΣdZ (m)':>25} | {'Gripper Total':>15}")
    print("-" * 110)
    cum_stats = {}
    for cond in conditions:
        sum_dx = [all_actions[s][cond][:, 0].sum() for s in seeds]
        sum_dy = [all_actions[s][cond][:, 1].sum() for s in seeds]
        sum_dz = [all_actions[s][cond][:, 2].sum() for s in seeds]
        sum_dg = [all_actions[s][cond][:, 5].sum() for s in seeds]
        cum_stats[cond] = {"sum_dx": sum_dx, "sum_dy": sum_dy, "sum_dz": sum_dz, "sum_dg": sum_dg}
        print(f"{cond:<12} | {np.mean(sum_dx):+8.5f} ± {np.std(sum_dx):7.5f} ({min(sum_dx):+.4f}..{max(sum_dx):+.4f}) | "
              f"{np.mean(sum_dy):+8.5f} ± {np.std(sum_dy):7.5f} ({min(sum_dy):+.4f}..{max(sum_dy):+.4f}) | "
              f"{np.mean(sum_dz):+8.5f} ± {np.std(sum_dz):7.5f} ({min(sum_dz):+.4f}..{max(sum_dz):+.4f}) | "
              f"{np.mean(sum_dg):+8.5f} ± {np.std(sum_dg):7.5f}")

    # 3. Matched-Seed Pairwise Differences
    print("\n3. MATCHED-SEED PAIRWISE DIFFERENCES (10 Seeds mean ± std)")
    print(f"{'Comparison':<22} | {'XYZ MAE':>18} | {'Full6D MAE':>18} | {'Cosine Sim':>18}")
    print("-" * 85)
    pairs = [
        ("Center vs Left", "A_center", "B_left"),
        ("Center vs Right", "A_center", "C_right"),
        ("Left vs Right", "B_left", "C_right"),
        ("Center vs NoCup", "A_center", "D_nocup"),
        ("Center vs Black", "A_center", "E_black"),
    ]
    pairwise_stats = {}
    for label, c1, c2 in pairs:
        xyz_maes = []
        full_maes = []
        cos_sims = []
        for s in seeds:
            a1 = all_actions[s][c1]
            a2 = all_actions[s][c2]
            xyz_maes.append(np.mean(np.abs(a1[:, :3] - a2[:, :3])))
            full_maes.append(np.mean(np.abs(a1 - a2)))
            # Flat cosine similarity
            v1, v2 = a1.flatten(), a2.flatten()
            cos_sims.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        pairwise_stats[label] = {
            "xyz_mae": (np.mean(xyz_maes), np.std(xyz_maes)),
            "full_mae": (np.mean(full_maes), np.std(full_maes)),
            "cos_sim": (np.mean(cos_sims), np.std(cos_sims)),
        }
        print(f"{label:<22} | {np.mean(xyz_maes):8.5f} ± {np.std(xyz_maes):7.5f} | "
              f"{np.mean(full_maes):8.5f} ± {np.std(full_maes):7.5f} | "
              f"{np.mean(cos_sims):8.5f} ± {np.std(cos_sims):7.5f}")

    # 4. Lateral (dY) Reaching Direction Analysis per Seed
    print("\n4. LATERAL STEERING (ΣdY) PER SEED")
    print(f"{'Seed':<6} | {'Center':>10} | {'Left':>10} | {'Right':>10} | {'Left - Center':>15} | {'Right - Center':>15} | {'Left - Right':>15}")
    print("-" * 95)
    d_left_center = []
    d_right_center = []
    d_left_right = []
    for s in seeds:
        dy_c = all_actions[s]["A_center"][:, 1].sum()
        dy_l = all_actions[s]["B_left"][:, 1].sum()
        dy_r = all_actions[s]["C_right"][:, 1].sum()
        dlc = dy_l - dy_c
        drc = dy_r - dy_c
        dlr = dy_l - dy_r
        d_left_center.append(dlc)
        d_right_center.append(drc)
        d_left_right.append(dlr)
        print(f"{s:<6} | {dy_c:+10.5f} | {dy_l:+10.5f} | {dy_r:+10.5f} | {dlc:+15.5f} | {drc:+15.5f} | {dlr:+15.5f}")
    
    pos_lc = sum(1 for x in d_left_center if x > 0)
    neg_lc = sum(1 for x in d_left_center if x < 0)
    pos_rc = sum(1 for x in d_right_center if x > 0)
    neg_rc = sum(1 for x in d_right_center if x < 0)
    pos_lr = sum(1 for x in d_left_right if x > 0)
    neg_lr = sum(1 for x in d_left_right if x < 0)

    print("-" * 95)
    print(f"Left vs Center  ΔΣdY: {np.mean(d_left_center):+.5f} ± {np.std(d_left_center):.5f} (Positive: {pos_lc}/10, Negative: {neg_lc}/10)")
    print(f"Right vs Center ΔΣdY: {np.mean(d_right_center):+.5f} ± {np.std(d_right_center):.5f} (Positive: {pos_rc}/10, Negative: {neg_rc}/10)")
    print(f"Left vs Right   ΔΣdY: {np.mean(d_left_right):+.5f} ± {np.std(d_left_right):.5f} (Positive: {pos_lr}/10, Negative: {neg_lr}/10)")

    # 5. Gripper Step 0 Action Comparison
    print("\n5. FIRST ACTION STEP (A0) GRIPPER SENSITIVITY")
    a0_c = [all_actions[s]["A_center"][0, 5] for s in seeds]
    a0_l = [all_actions[s]["B_left"][0, 5] for s in seeds]
    a0_r = [all_actions[s]["C_right"][0, 5] for s in seeds]
    a0_n = [all_actions[s]["D_nocup"][0, 5] for s in seeds]
    a0_b = [all_actions[s]["E_black"][0, 5] for s in seeds]
    print(f"Center A0 dGripper: {np.mean(a0_c):+.6f} ± {np.std(a0_c):.6f}")
    print(f"Left   A0 dGripper: {np.mean(a0_l):+.6f} ± {np.std(a0_l):.6f}")
    print(f"Right  A0 dGripper: {np.mean(a0_r):+.6f} ± {np.std(a0_r):.6f}")
    print(f"NoCup  A0 dGripper: {np.mean(a0_n):+.6f} ± {np.std(a0_n):.6f}")
    print(f"Black  A0 dGripper: {np.mean(a0_b):+.6f} ± {np.std(a0_b):.6f}")
    
    # Check consistency Center > NoCup, Center > Black (or more negative / positive)
    c_vs_n_count = sum(1 for s in range(len(seeds)) if a0_c[s] != a0_n[s])
    c_vs_b_count = sum(1 for s in range(len(seeds)) if a0_c[s] != a0_b[s])
    print(f"Center vs NoCup A0 different: {c_vs_n_count}/10")
    print(f"Center vs Black A0 different: {c_vs_b_count}/10")

    # 6. OLD vs NEW Comparison
    old_dir = PROJECT_ROOT / "step3b_actions"
    if old_dir.exists():
        print("\n6. OLD (Gripper=1.0) vs NEW (Gripper=0.441305) DIRECT COMPARISON")
        print(f"{'Condition':<12} | {'XYZ MAE Diff':>14} | {'Full6D MAE Diff':>16} | {'OLD ΣdX':>18} | {'NEW ΣdX':>18} | {'OLD ΣdY':>18} | {'NEW ΣdY':>18}")
        print("-" * 125)
        for cond in conditions:
            diffs_xyz = []
            diffs_full = []
            old_dxs = []
            old_dys = []
            for s in seeds:
                old_act = np.load(str(old_dir / f"seed{s}_{cond}.npy"))
                new_act = all_actions[s][cond]
                diffs_xyz.append(np.mean(np.abs(new_act[:, :3] - old_act[:, :3])))
                diffs_full.append(np.mean(np.abs(new_act - old_act)))
                old_dxs.append(old_act[:, 0].sum())
                old_dys.append(old_act[:, 1].sum())
            
            new_dxs = cum_stats[cond]["sum_dx"]
            new_dys = cum_stats[cond]["sum_dy"]
            print(f"{cond:<12} | {np.mean(diffs_xyz):14.5f} | {np.mean(diffs_full):16.5f} | "
                  f"{np.mean(old_dxs):+7.4f} ± {np.std(old_dxs):.4f} | {np.mean(new_dxs):+7.4f} ± {np.std(new_dxs):.4f} | "
                  f"{np.mean(old_dys):+7.4f} ± {np.std(old_dys):.4f} | {np.mean(new_dys):+7.4f} ± {np.std(new_dys):.4f}")

    # 7. Visual Effect vs Noise Ratio Calculation
    avg_xyz_noise = np.mean([within_var[c]["xyz_mae"] for c in conditions])
    lr_xyz_effect = pairwise_stats["Left vs Right"]["xyz_mae"][0]
    new_ratio = lr_xyz_effect / (avg_xyz_noise + 1e-9)
    print(f"\n7. VISUAL EFFECT / NOISE RATIO")
    print(f"Average Within-Condition Sampling Noise (XYZ MAE): {avg_xyz_noise:.5f}")
    print(f"Left vs Right Matched-Seed Visual Effect (XYZ MAE): {lr_xyz_effect:.5f}")
    print(f"NEW Visual Effect / Noise Ratio:                   {new_ratio:.4f}")

if __name__ == "__main__":
    main()
