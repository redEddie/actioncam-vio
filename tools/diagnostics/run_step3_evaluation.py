"""STEP 3 Vision Ablation - Hold Start Pose & Interactive Evaluation."""

import json
import base64
import io
import shlex
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"

import sys
sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping
from control.motor_io import MotorIO
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator

def capture_live_frame(cap, remove_pillarbox=True):
    for _ in range(10):  # flush buffer
        ret, bgr = cap.read()
    if not ret or bgr is None:
        raise RuntimeError("Failed to capture from /dev/video2")
    
    # Process 8:7 active FOV squeeze
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if remove_pillarbox:
        col_means = rgb.mean(axis=(0, 2))
        active_cols = np.where(col_means > 8.0)[0]
        if len(active_cols) > 0:
            x_min, x_max = active_cols[0], active_cols[-1] + 1
            rgb = rgb[:, x_min:x_max]
    
    return cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)

def run_remote_inference(images_dict: dict[str, np.ndarray], fixed_state: np.ndarray) -> dict[str, np.ndarray]:
    """Execute remote inference on server over SSH and return unnormalized action chunks."""

    remote_worker_code = f'''
import json, base64, io, sys, os
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
print(json.dumps({{"status": "ready"}}), file=protocol_stdout, flush=True)

for line in sys.stdin:
    if not line.strip(): continue
    req = json.loads(line)
    if req.get("cmd") == "exit": break
    
    name = req["name"]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))))
    state = np.asarray(req["state"], dtype=np.float32)
    obs = {{
        "observation.images.camera1": torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0,
        "observation.state": torch.from_numpy(state).unsqueeze(0).cuda(),
        "task": ["pick and place the target object"],
        "robot_type": ["so_follower"],
    }}
    with torch.inference_mode():
        actions = post(policy.predict_action_chunk(pre(obs)))
    
    res = {{
        "name": name,
        "action": actions[0].detach().cpu().numpy().tolist()
    }}
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

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait for ready handshake
    ready_line = proc.stdout.readline()
    try:
        ready_data = json.loads(ready_line)
    except Exception as e:
        err = proc.stderr.read()
        raise RuntimeError(f"Worker handshake failed on line: {repr(ready_line)}, stderr: {err}") from e

    if ready_data.get("status") != "ready":
        err = proc.stderr.read()
        raise RuntimeError(f"Worker startup failed: {ready_line}, err: {err}")

    results = {}
    for name, img_rgb in images_dict.items():
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="PNG")
        req = {
            "name": name,
            "image_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "state": fixed_state.tolist(),
        }
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        
        resp_line = proc.stdout.readline()
        resp = json.loads(resp_line)
        results[resp["name"]] = np.array(resp["action"], dtype=np.float64)

    # Send exit
    proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
    proc.stdin.flush()
    proc.terminate()

    return results

def main():
    print("\n=======================================================")
    print(" STEP 3 VISION ABLATION (HOLD START POSE & EVALUATE)")
    print("=======================================================\n")

    config = load_runtime_config(DEPLOY_DIR / "config" / "deployment.yaml")
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    gravity = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    motor = MotorIO(
        config.deployment["motor"]["port"],
        config.motor_calibration,
        config.deployment["motor"]["pid"],
        disable_torque_on_disconnect=config.deployment["motor"]["disable_torque_on_disconnect"],
    )

    # Move to start pose and maintain hold in background thread
    print("1. 로봇을 시작 자세로 부드럽게 이동시킵니다...")
    motor.connect(permit_motion=True)
    start_ticks = mapping.arm_degrees_to_raw_ticks(config.start_raw_deg)
    motor_cfg = config.deployment["motor"]
    motor.move_to_start_ticks(
        start_ticks,
        int(motor_cfg["start_move_steps"]),
        float(motor_cfg["start_move_duration_s"]),
    )
    time.sleep(float(motor_cfg["settle_duration_s"]))
    print(" -> 로봇이 시작 자세에 도달했습니다. 자세 유지를 시작합니다.\n")

    stop_hold = threading.Event()

    def hold_loop():
        while not stop_hold.is_set():
            try:
                deg_state = motor.read_present_positions(normalized=True)
                tick_state = motor.read_present_positions(normalized=False)
                raw_deg = [deg_state.positions[name] for name in mapping.joint_order]
                q_urdf = mapping.raw_degrees_to_urdf_degrees(raw_deg)
                q_hold = controller.compute_hold(config.start_urdf_deg, q_urdf)
                raw_goals = mapping.urdf_degrees_to_raw_degrees(q_hold)
                goals = mapping.arm_degrees_to_raw_ticks(raw_goals)
                goals["gripper"] = mapping.gripper_raw_from_width(1.0)
                motor.write_goal_ticks(goals)
            except Exception:
                pass
            time.sleep(config.control_period_s)

    hold_thread = threading.Thread(target=hold_loop, name="motor-hold", daemon=True)
    hold_thread.start()

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    images = {}

    try:
        # 1. Center
        input("\n[1/4] 컵을 [중앙(Center)]에 놓고 엔터(Enter)를 누르세요...")
        images["A_center"] = capture_live_frame(cap)
        cv2.imwrite(str(PROJECT_ROOT / "step3_A_center.png"), cv2.cvtColor(images["A_center"], cv2.COLOR_RGB2BGR))
        print("  -> step3_A_center.png 캡처 완료!")

        # 2. Left
        input("\n[2/4] 컵을 [왼쪽(Left, 5~10cm)]으로 이동시킨 후 엔터(Enter)를 누르세요...")
        images["B_left"] = capture_live_frame(cap)
        cv2.imwrite(str(PROJECT_ROOT / "step3_B_left.png"), cv2.cvtColor(images["B_left"], cv2.COLOR_RGB2BGR))
        print("  -> step3_B_left.png 캡처 완료!")

        # 3. Right
        input("\n[3/4] 컵을 [오른쪽(Right, 5~10cm)]으로 이동시킨 후 엔터(Enter)를 누르세요...")
        images["C_right"] = capture_live_frame(cap)
        cv2.imwrite(str(PROJECT_ROOT / "step3_C_right.png"), cv2.cvtColor(images["C_right"], cv2.COLOR_RGB2BGR))
        print("  -> step3_C_right.png 캡처 완료!")

        # 4. No Cup
        input("\n[4/4] 컵을 카메라 시야에서 [완전히 치운(No Cup)] 후 엔터(Enter)를 누르세요...")
        images["D_nocup"] = capture_live_frame(cap)
        cv2.imwrite(str(PROJECT_ROOT / "step3_D_nocup.png"), cv2.cvtColor(images["D_nocup"], cv2.COLOR_RGB2BGR))
        print("  -> step3_D_nocup.png 캡처 완료!")

        # 5. Black Image (Auto)
        images["E_black"] = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(PROJECT_ROOT / "step3_E_black.png"), cv2.cvtColor(images["E_black"], cv2.COLOR_RGB2BGR))
        print("\n[5/5] 암전(Black) 이미지는 자동 생성되었습니다.")

    finally:
        cap.release()
        stop_hold.set()
        hold_thread.join(timeout=1.0)
        motor.disconnect()
        print("\n모터 연결이 안전하게 해제되었습니다.")

    # Fixed state: Start TCP state
    s_fixed = np.array([0.3185, -0.0105, 0.0558, -82.34, -3.03, 1.0], dtype=np.float32)

    # Repeatability test payload (Center 5 times)
    inference_payload = {}
    for i in range(5):
        inference_payload[f"A_center_rep{i}"] = images["A_center"].copy()
    inference_payload["A_center"] = images["A_center"]
    inference_payload["B_left"] = images["B_left"]
    inference_payload["C_right"] = images["C_right"]
    inference_payload["D_nocup"] = images["D_nocup"]
    inference_payload["E_black"] = images["E_black"]

    print("\n-------------------------------------------------------")
    print(" 서버(GPU)로 5개 조건에 대한 SmolVLA 추론 요청 중...")
    print("-------------------------------------------------------")
    actions = run_remote_inference(inference_payload, s_fixed)
    print(" 추론 완료! 분석 결과를 계산합니다.\n")

    # Save action npy files
    for name in ["A_center", "B_left", "C_right", "D_nocup", "E_black"]:
        npy_path = PROJECT_ROOT / f"step3_{name}_action.npy"
        np.save(str(npy_path), actions[name])

    # Check Repeatability
    center_reps = [actions[f"A_center_rep{i}"] for i in range(5)]
    rep_diffs = []
    for i in range(len(center_reps)):
        for j in range(i+1, len(center_reps)):
            rep_diffs.append(np.abs(center_reps[i] - center_reps[j]))
    rep_diffs = np.array(rep_diffs)
    within_mae = rep_diffs.mean()
    within_max = rep_diffs.max()

    print("==================================================")
    print(" STEP 3 VISION ABLATION — RESULT TABLE")
    print("==================================================\n")

    print(f"1. INFERENCE REPEATABILITY (5 CENTER RUNS)")
    print(f" - Within-condition MAE: {within_mae:.6e}")
    print(f" - Within-condition MAX: {within_max:.6e}")
    print(f" - Determinism: {'DETERMINISTIC' if within_max < 1e-6 else 'STOCHASTIC'}\n")

    print("2. A0 IMMEDIATE ACTION COMPARISON")
    print(f"{'Condition':<12} | {'dX (m)':>9} | {'dY (m)':>9} | {'dZ (m)':>9} | {'dRoll':>9} | {'dPitch':>9} | {'dGripper':>9}")
    print("-" * 75)
    for name, label in [("A_center", "Center"), ("B_left", "Left"), ("C_right", "Right"), ("D_nocup", "No Cup"), ("E_black", "Black")]:
        a0 = actions[name][0]
        print(f"{label:<12} | {a0[0]:>9.5f} | {a0[1]:>9.5f} | {a0[2]:>9.5f} | {a0[3]:>9.5f} | {a0[4]:>9.5f} | {a0[5]:>9.5f}")

    print("\n3. 1.5s CUMULATIVE DISPLACEMENT (ΣdX, ΣdY, ΣdZ)")
    print(f"{'Condition':<12} | {'ΣdX (m)':>9} | {'ΣdY (m)':>9} | {'ΣdZ (m)':>9} | {'Gripper Total':>14}")
    print("-" * 65)
    for name, label in [("A_center", "Center"), ("B_left", "Left"), ("C_right", "Right"), ("D_nocup", "No Cup"), ("E_black", "Black")]:
        sums = actions[name].sum(axis=0)
        print(f"{label:<12} | {sums[0]:>9.5f} | {sums[1]:>9.5f} | {sums[2]:>9.5f} | {sums[5]:>14.5f}")

    print("\n4. PAIRWISE ACTION DIFFERENCES")
    pairs = [
        ("Center vs Left", actions["A_center"], actions["B_left"]),
        ("Center vs Right", actions["A_center"], actions["C_right"]),
        ("Left vs Right", actions["B_left"], actions["C_right"]),
        ("Center vs No Cup", actions["A_center"], actions["D_nocup"]),
        ("Center vs Black", actions["A_center"], actions["E_black"]),
    ]
    print(f"{'Comparison':<18} | {'XYZ MAE':>9} | {'Full 6D MAE':>11} | {'Max Diff':>9} | {'Cosine Sim':>11}")
    print("-" * 65)
    for label, a1, a2 in pairs:
        xyz_mae = np.abs(a1[:, :3] - a2[:, :3]).mean()
        full_mae = np.abs(a1 - a2).mean()
        max_diff = np.abs(a1 - a2).max()
        f1, f2 = a1.flatten(), a2.flatten()
        cos_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-12)
        print(f"{label:<18} | {xyz_mae:>9.5f} | {full_mae:>11.5f} | {max_diff:>9.5f} | {cos_sim:>11.6f}")

    print("\n5. AXIS SENSITIVITY")
    axis_names = ["dX", "dY", "dZ", "dRoll", "dPitch", "dGripper"]
    for i, axis in enumerate(axis_names):
        diff_lr = np.abs(actions["B_left"][:, i] - actions["C_right"][:, i]).mean()
        diff_nc = np.abs(actions["A_center"][:, i] - actions["D_nocup"][:, i]).mean()
        diff_bk = np.abs(actions["A_center"][:, i] - actions["E_black"][:, i]).mean()
        print(f" {axis:<10}: Left-vs-Right diff = {diff_lr:.5f}, NoCup diff = {diff_nc:.5f}, Black diff = {diff_bk:.5f}")

    print("\n==================================================")

if __name__ == "__main__":
    main()
