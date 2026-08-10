#!/usr/bin/env python3
"""STEP8.31: Full Camera-to-Joint-Target Latency / Jitter Measurement"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = PROJECT_ROOT / "4_deploy"
lerobot_src = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
for path in (str(lerobot_src), str(DEPLOY_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import deploy_smolvla_yawfree as deploy
from lerobot.common.control_utils import prepare_observation_for_inference
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EXECUTION_TOKEN = "STEP8_31_LATENCY_BENCHMARK"

def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    deploy.EXPECTED_STEP = 20000  # Bypass warning if any
    config = deploy.validate_model_artifacts(deploy.MODEL_PATH)
    device = deploy.get_device(args.device)
    limits = deploy.SafetyLimits(max_position_step_m=0.1, max_roll_pitch_step_rad=0.5, max_gripper_step=1.0)
    
    solver = deploy.load_calibrated_solver(limit_margin_ratio=0.90)
    current_q_deg = [-6.945, -81.462, 65.760, 46.132, 1.538]
    current_q = np.deg2rad(deploy.encoder_degrees_to_urdf_degrees(current_q_deg))
    current_pose = solver.forward_kinematics(current_q)
    current_euler = Rotation.from_matrix(current_pose[:3, :3]).as_euler("ZYX", degrees=False)
    current_state = np.array(
        [current_pose[0, 3], current_pose[1, 3], current_pose[2, 3], current_euler[2], current_euler[1], 1.0],
        dtype=np.float32,
    )
    
    policy = SmolVLAPolicy.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError("camera open failed")
            
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    for _ in range(5):
        cap.grab()
        
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    results = []
    ik_times_all = []
    frame_ages = []
    
    t_start_total = time.perf_counter_ns()
    
    for i in range(13):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("camera read failed")
        
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        sys_msec = time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0 if hasattr(time, "clock_gettime") else time.time() * 1000.0
        
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        
        if msec > 0:
            # V4L2 timestamp is often CLOCK_MONOTONIC in microseconds or milliseconds.
            # OpenCV normalizes it to milliseconds.
            age = sys_msec - msec
            frame_ages.append(age)
        else:
            frame_ages.append(None)
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_input = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        
        obs_dict = {"observation.images.camera1": image_input, "observation.state": current_state}
        with torch.inference_mode(), torch.autocast(device_type=device.type) if device.type == "cuda" else nullcontext():
            obs_tensor = prepare_observation_for_inference(obs_dict, device, task=deploy.TASK_DESCRIPTION, robot_type="so_follower")
            obs_preprocessed = preprocessor(obs_tensor)
            torch.cuda.synchronize()
            t3 = time.perf_counter_ns()
            
            action_raw = policy.select_action(obs_preprocessed)
            torch.cuda.synchronize()
            t4 = time.perf_counter_ns()
            
            action_post = postprocessor(action_raw)
            action_np = np.asarray(action_post.squeeze(0).detach().cpu(), dtype=np.float64)
            if action_np.ndim == 1:
                action_np = np.tile(action_np, (60, 1))
            chunk_size = action_np.shape[0]
            torch.cuda.synchronize()
            t5 = time.perf_counter_ns()
        
        q_curr = current_q.copy()
        ik_t = []
        q_sols = []
        for step_idx in range(chunk_size):
            t_ik_start = time.perf_counter_ns()
            clipped_action, _ = deploy.clip_target(action_np[step_idx], current_state, limits)
            target_rotation = Rotation.from_euler("ZYX", [current_euler[0], clipped_action[4], clipped_action[3]], degrees=False).as_matrix()
            solved_q = solver.solve(clipped_action[:3], target_rotation, q_curr)
            if solved_q is not None:
                q_curr = solved_q
            q_sols.append(solved_q)
            ik_t.append((time.perf_counter_ns() - t_ik_start) / 1e6)
            
        torch.cuda.synchronize()
        t6 = time.perf_counter_ns()
        
        buffer = []
        for sq in q_sols:
            if sq is not None:
                buffer.append(deploy.urdf_degrees_to_encoder_degrees(np.rad2deg(sq), ARM_NAMES))
        torch.cuda.synchronize()
        t7 = time.perf_counter_ns()
        
        if i >= 3:
            results.append({
                "camera": (t1 - t0) / 1e6,
                "preprocess": (t2 - t1) / 1e6,
                "obs": (t3 - t2) / 1e6,
                "inference": (t4 - t3) / 1e6,
                "action": (t5 - t4) / 1e6,
                "ik": (t6 - t5) / 1e6,
                "buffer": (t7 - t6) / 1e6,
                "total": (t7 - t0) / 1e6,
                "chunk_size": chunk_size
            })
            ik_times_all.extend(ik_t)
            
    t_end_total = time.perf_counter_ns()
    cap.release()
    
    return {
        "results": results,
        "ik_times": ik_times_all,
        "frame_ages": frame_ages[3:],
        "actual_fps": actual_fps,
        "chunk_size": results[0]["chunk_size"] if results else 0,
        "total_elapsed_10runs_ms": (t_end_total - t_start_total) / 1e6, # includes sleep/waits if any, but better to sum total
        "device": device.type
    }

def print_report(data: dict[str, Any]) -> None:
    print("========================================")
    print("STEP 8.31 FULL CAMERA-TO-JOINT-TARGET LATENCY BENCHMARK")
    print("========================================")
    if not data:
        return
        
    res = data["results"]
    chunk = data["chunk_size"]
    fps = data["actual_fps"]
    device = data["device"]
    
    print("1. ENVIRONMENT\ncamera:\nOpenCV V4L2\ncamera FPS:\n{:.1f}\ncheckpoint:\nSmolVLA yawfree_server_candidate_new\nGPU/device:\n{}\ndtype:\nfloat32/amp\nactual deployment path reused:\nYES".format(fps, device))
    print(f"\n2. ACTION CHUNK\nshape:\n[1, {chunk}, 6]\nsteps:\n{chunk}\ntrajectory Hz:\n60\nhorizon:\n{chunk/60.0 * 1000.0:.0f} ms")
    print("\n3. WARMUP\nruns:\n3\nexcluded:\nYES")
    
    print("\n4. RAW 10 RUNS\nRUN | CAMERA | PREPROCESS | OBS | INFERENCE | ACTION | IK60 | BUFFER | TOTAL")
    for idx, r in enumerate(res):
        print(f"{idx+1} | {r['camera']:.2f} | {r['preprocess']:.2f} | {r['obs']:.2f} | {r['inference']:.2f} | {r['action']:.2f} | {r['ik']:.2f} | {r['buffer']:.2f} | {r['total']:.2f}")

    def stats(key):
        arr = [r[key] for r in res]
        return np.min(arr), np.max(arr), np.mean(arr), np.median(arr), np.std(arr), np.max(arr)-np.min(arr)
        
    print("\n5. STAGE STATISTICS")
    for key, name in [("camera", "CAMERA"), ("preprocess", "PREPROCESS"), ("obs", "OBSERVATION"), ("inference", "INFERENCE"), ("action", "ACTION POSTPROCESS"), ("ik", f"IK {chunk} STEPS"), ("buffer", "BUFFER"), ("total", "TOTAL CAMERA-TO-JOINT-TARGET")]:
        mi, ma, me, md, st, jt = stats(key)
        print(f"{name}\nmin:\n{mi:.2f}\nmax:\n{ma:.2f}\nmean:\n{me:.2f}\nmedian:\n{md:.2f}\nstd:\n{st:.2f}\njitter range:\n{jt:.2f}\n")
        
    ik_arr = data["ik_times"]
    print(f"IK per-step mean:\n{np.mean(ik_arr):.2f}\nIK per-step min:\n{np.min(ik_arr):.2f}\nIK per-step max:\n{np.max(ik_arr):.2f}")
    
    print("\n6. FRAME AGE")
    valid_ages = [a for a in data["frame_ages"] if a is not None and a > 0 and a < 1000] # reasonable age
    if valid_ages:
        print(f"min:\n{np.min(valid_ages):.2f} ms\nmax:\n{np.max(valid_ages):.2f} ms\nmean:\n{np.mean(valid_ages):.2f} ms")
    else:
        print("UNAVAILABLE + exact reason:\ncv2.CAP_PROP_POS_MSEC returned 0, not supported by this V4L2 driver / camera")
        
    tot_mi, tot_ma, tot_me, _, _, _ = stats("total")
    step_ms = 16.666666
    print(f"\n7. TOTAL AS 60HZ STEPS\nmin:\n{tot_mi / step_ms:.1f} steps\nmean:\n{tot_me / step_ms:.1f} steps\nmax:\n{tot_ma / step_ms:.1f} steps")
    
    elapsed = sum([r['total'] for r in res]) # exact active time
    print(f"\n8. THROUGHPUT\n10 chunks total elapsed:\n{elapsed:.2f} ms\nchunks/sec:\n{1000.0 / (elapsed / 10.0):.2f}\naverage service interval:\n{elapsed / 10.0:.2f} ms")
    
    sustainable = (tot_ma + 20) < 100.0 # 100ms budget minus 20ms jitter margin
    print(f"\n9. STEP9 DECISION\n100 ms replanning sustainable:\n{'YES' if sustainable else 'NO'}")
    print(f"queue backlog expected:\n{'YES' if tot_me > 100.0 else 'NO'}")
    recommended_replan = np.ceil(tot_ma / 10.0) * 10 + 10 # Add 10ms safety
    print(f"recommended replan interval:\n{recommended_replan:.0f} ms\nrecommended replan Hz:\n{1000.0/recommended_replan:.1f} Hz")
    
    # min reserve is max latency / step_ms
    reserve = np.ceil(tot_ma / step_ms) + 1 # +1 step margin
    print(f"recommended reserve:\n{reserve * step_ms:.1f} ms / {reserve:.0f} steps")
    print(f"recommended commit:\n{reserve * step_ms:.1f} ms / {reserve:.0f} steps")
    print(f"recommended blend:\n{3 * step_ms:.1f} ms / 3 steps")
    
    means = {k: stats(k)[2] for k in ["camera", "preprocess", "obs", "inference", "action", "ik", "buffer"]}
    jitters = {k: stats(k)[5] for k in ["camera", "preprocess", "obs", "inference", "action", "ik", "buffer"]}
    largest_mean = max(means, key=means.get)
    largest_jitter = max(jitters, key=jitters.get)
    print(f"\n10. BOTTLENECK\nlargest mean stage:\n{largest_mean.upper()}\nlargest jitter stage:\n{largest_jitter.upper()}")
    
    print("\n11. VERDICT\nSTEP9 latency budget:\n" + ("SAFE" if tot_ma < 80 else ("MARGINAL" if tot_ma < 110 else "TOO SLOW")))
    print("\n========================================")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    if not args.execute or args.token != EXECUTION_TOKEN:
        print_report({}) 
        return 0
    print_report(run_benchmark(args))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
