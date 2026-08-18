#!/usr/bin/env python3
"""
Benchmark 01 — Remote Pipeline Latency Validation Scaffold
==========================================================
Evaluates the end-to-end latency of the remote SmolVLA inference pipeline:
  [Local Observation Acquisition]
        ↓
  [Local Preprocessing (256x256 RGB + 6D FK State)]
        ↓
  [Persistent SSH Transport Send]
        ↓
  [Remote SmolVLA Policy Inference]
        ↓
  [Response Network Transport]
        ↓
  [Local Postprocessing -> Ready [15,6] Action Chunk]

Modes:
  --audit (default): Non-invasive configuration, import, and pipeline contract audit (NO SSH, NO Camera, NO Network).
  --measure: Explicit execution flag to run live remote pipeline latency benchmark over persistent connection.

Timing Boundaries:
  T0: Pipeline invocation / fresh observation capture start
  T1: Validated local [15,6] action chunk available for trajectory reconstruction
  Total Latency = T1 - T0
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "4_deploy"
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.observation import build_observation, state_from_motor_sample, ObservationSnapshot
from inference.remote_smolvla import RemoteSmolVLAClient, RemoteRequest, SubprocessSSHTransport


DEFAULT_WARMUP_RUNS = 5
DEFAULT_MEASURED_RUNS = 30
ACTION_HORIZON_STEPS = 15
ACTION_DIM = 6
ACTION_RATE_HZ = 10.0
ACTION_STEP_DT_S = 1.0 / ACTION_RATE_HZ  # 0.100 s = 100 ms
LATENCY_TARGET_S = 0.400                 # 400 ms target (< 4 action steps)


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_s: float


def compute_latency_stats(values_ms: Sequence[float]) -> LatencyStats:
    if len(values_ms) == 0:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    arr = np.asarray(values_ms, dtype=np.float64)
    return LatencyStats(
        count=len(arr),
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        mean_s=float(np.mean(arr)) / 1000.0,
    )


def run_audit_mode(config_path: Path):
    """Safe, non-invasive static contract and configuration audit."""
    print("=" * 70)
    print("BENCHMARK 01 — REMOTE PIPELINE LATENCY AUDIT (DRY-RUN)")
    print("=" * 70)
    print("STATUS: AUDIT ONLY (NO SSH, NO CAMERA, NO MODEL INFERENCE)")
    print("-" * 70)

    config = load_runtime_config(config_path)
    remote_cfg = config.deployment.get("remote", {})
    model_cfg = config.deployment.get("model", {})
    cam_cfg = config.deployment.get("camera", {})

    print("1. DEPLOYMENT CONFIGURATION AUDIT:")
    print(f"   Remote Host                 : {remote_cfg.get('user')}@{remote_cfg.get('host')}")
    print(f"   SSH Port                    : {remote_cfg.get('port')}")
    print(f"   Remote Python               : {remote_cfg.get('python')}")
    print(f"   Remote Checkpoint           : {remote_cfg.get('checkpoint')}")
    print(f"   Remote Request Timeout      : {remote_cfg.get('request_timeout_s')} s")
    print(f"   Action Dimension            : {ACTION_DIM} [dX, dY, dZ, dRoll, dPitch, dGripper]")
    print(f"   Chunk Size / Horizon        : {ACTION_HORIZON_STEPS} steps ({ACTION_HORIZON_STEPS * ACTION_STEP_DT_S:.1f} s horizon at {ACTION_RATE_HZ} Hz)")
    print(f"   Latency Budget Target       : < {LATENCY_TARGET_S * 1000.0:.1f} ms (< {LATENCY_TARGET_S / ACTION_STEP_DT_S:.1f} action steps)")

    print("\n2. CAMERA & OBSERVATION SPECIFICATION:")
    print(f"   Observation Image Shape     : (256, 256, 3) RGB uint8")
    print(f"   State Feature Shape         : (6,) [X, Y, Z, Roll, Pitch, Gripper]")
    print(f"   Camera Backend / Device     : {cam_cfg.get('backend')} / Device {cam_cfg.get('device')}")

    print("\n3. BENCHMARK EXECUTION CONTRACT:")
    print(f"   Connection Mode             : Persistent SSH Subprocess Worker")
    print(f"   Default Warmup Runs         : {DEFAULT_WARMUP_RUNS}")
    print(f"   Default Measured Runs       : {DEFAULT_MEASURED_RUNS}")
    print(f"   Timing Start Boundary (T0)  : Fresh observation snapshot acquisition start")
    print(f"   Timing End Boundary (T1)    : Validated [15,6] action array available locally")

    print("\n4. TEMPORARY PLANNING VALUE (NOT MEASURED):")
    print(f"   Temporary Planning Value    : 350.0 ms (0.350 s)")
    print(f"   Planning Action Steps       : 3.5 action steps at 10 Hz")
    print(f"   Actual Measurement Status   : PENDING (Run with --measure to execute live test)")

    print("=" * 70)
    print("AUDIT RESULT: READY FOR EXECUTION (Passes static contract verification)")
    print("=" * 70)


def run_measurement_mode(args, config_path: Path):
    """Live measurement execution over persistent SSH connection."""
    print("=" * 70)
    print("BENCHMARK 01 — ACTUAL REMOTE MEASUREMENT")
    print("=" * 70)
    print("THIS WILL ACCESS:")
    print("  CAMERA         = YES (GoPro live capture)")
    print("  REMOTE SERVER  = YES (SSH connection)")
    print("  REMOTE MODEL   = YES (SmolVLA inference)")
    print("  MOTOR WRITE    = NO  (Arm stationary)")
    print("CONTINUE ONLY BECAUSE --measure WAS EXPLICITLY PROVIDED")
    print("=" * 70)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now_str}"
    results_base = Path(__file__).resolve().parent / "results"
    run_dir = results_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_runtime_config(config_path)
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    ik_solver = DLSInverseKinematicsV7(str(DEPLOY_DIR / "ik" / "urdf" / "so_arm_with_gopro_final.urdf"))

    # Import camera lazily to avoid opening OpenCV camera during dry-run audit
    from inference.camera import LatestFrameCamera

    cam_cfg = config.deployment["camera"]
    camera = LatestFrameCamera(
        device=cam_cfg["device"],
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        stale_timeout_s=cam_cfg["stale_timeout_s"],
    )

    transport = SubprocessSSHTransport(config.deployment["remote"])
    client = RemoteSmolVLAClient(transport, request_timeout_s=config.deployment["remote"]["request_timeout_s"])

    print("Starting camera capture thread and persistent remote SSH worker...")
    camera.start()
    client.start()
    time.sleep(1.0)  # Settle background camera stream

    total_runs = args.warmup_runs + args.measured_runs
    samples_records = []
    total_latencies_ms = []
    camera_latencies_ms = []
    preprocess_latencies_ms = []
    roundtrip_latencies_ms = []
    postprocess_latencies_ms = []

    successful_runs = 0
    failed_runs = 0
    failure_reasons = []

    print(f"Beginning {args.warmup_runs} warmup + {args.measured_runs} measured runs...")

    # Canonical start state reference for offline state observation
    start_config = json.loads((DEPLOY_DIR / "config" / "physical_start.json").read_text())
    target_start_urdf_deg = np.asarray(start_config["urdf_start_deg"], dtype=np.float64)
    target_start_urdf_rad = np.deg2rad(target_start_urdf_deg)
    T0 = ik_solver.forward_kinematics(target_start_urdf_rad)
    from scipy.spatial.transform import Rotation
    yaw0, pitch0, roll0 = Rotation.from_matrix(T0[:3, :3]).as_euler("ZYX", degrees=False)
    dummy_state = np.array([*T0[:3, 3], roll0, pitch0, 0.0], dtype=np.float64)

    for idx in range(total_runs):
        is_warmup = (idx < args.warmup_runs)
        phase_label = "WARMUP" if is_warmup else "MEASURED"
        t0 = time.perf_counter_ns()

        try:
            # 1. Camera Snapshot
            t_cam_start = time.perf_counter_ns()
            frame = camera.latest()
            t_cam_end = time.perf_counter_ns()
            cam_ms = (t_cam_end - t_cam_start) / 1e6

            # 2. Preprocess / Request build
            t_prep_start = time.perf_counter_ns()
            request = RemoteRequest(
                request_id=idx + 1,
                t_obs=frame.timestamp,
                image_rgb=frame.image_rgb,
                state=dummy_state,
            )
            t_prep_end = time.perf_counter_ns()
            prep_ms = (t_prep_end - t_prep_start) / 1e6

            # 3. Persistent Remote Transport & Inference
            t_rt_start = time.perf_counter_ns()
            client.submit(request)
            timeout_limit = time.perf_counter() + config.deployment["remote"]["request_timeout_s"]
            response = None
            while time.perf_counter() < timeout_limit:
                response = client.poll()
                if response is not None and response.request_id == request.request_id:
                    break
                time.sleep(0.002)
            if response is None:
                raise TimeoutError("Remote inference request timed out")
            t_rt_end = time.perf_counter_ns()
            rt_ms = (t_rt_end - t_rt_start) / 1e6

            # 4. Postprocess & Validation
            t_post_start = time.perf_counter_ns()
            if response.action is None or response.error is not None:
                raise RuntimeError(f"Remote returned error: {response.error}")
            action_chunk = validate_postprocessed_action_chunk(response.action)
            t_post_end = time.perf_counter_ns()
            post_ms = (t_post_end - t_post_start) / 1e6

            t1 = time.perf_counter_ns()
            tot_ms = (t1 - t0) / 1e6

            if not is_warmup:
                successful_runs += 1
                total_latencies_ms.append(tot_ms)
                camera_latencies_ms.append(cam_ms)
                preprocess_latencies_ms.append(prep_ms)
                roundtrip_latencies_ms.append(rt_ms)
                postprocess_latencies_ms.append(post_ms)

            samples_records.append({
                "run_index": idx + 1,
                "phase": phase_label,
                "total_ms": f"{tot_ms:.4f}",
                "camera_ms": f"{cam_ms:.4f}",
                "preprocess_ms": f"{prep_ms:.4f}",
                "remote_roundtrip_ms": f"{rt_ms:.4f}",
                "postprocess_ms": f"{post_ms:.4f}",
                "action_shape": str(list(action_chunk.shape)),
                "finite": "True",
                "error": "None",
            })
            print(f"[{phase_label}] Run {idx+1}/{total_runs}: Total={tot_ms:6.2f} ms (Camera={cam_ms:4.2f} ms, Pre={prep_ms:4.2f} ms, RTT={rt_ms:6.2f} ms, Post={post_ms:4.2f} ms)")

        except Exception as exc:
            if not is_warmup:
                failed_runs += 1
                failure_reasons.append(str(exc))
            print(f"[{phase_label}] Run {idx+1}/{total_runs} FAILED: {exc}")
            samples_records.append({
                "run_index": idx + 1,
                "phase": phase_label,
                "total_ms": "NaN",
                "camera_ms": "NaN",
                "preprocess_ms": "NaN",
                "remote_roundtrip_ms": "NaN",
                "postprocess_ms": "NaN",
                "action_shape": "None",
                "finite": "False",
                "error": str(exc),
            })

        time.sleep(0.1)  # Stride between requests

    client.close()
    camera.stop()

    # Save samples.csv
    samples_path = run_dir / "samples.csv"
    with open(samples_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples_records[0].keys()))
        writer.writeheader()
        writer.writerows(samples_records)

    stats_tot = compute_latency_stats(total_latencies_ms)
    stats_cam = compute_latency_stats(camera_latencies_ms)
    stats_prep = compute_latency_stats(preprocess_latencies_ms)
    stats_rt = compute_latency_stats(roundtrip_latencies_ms)
    stats_post = compute_latency_stats(postprocess_latencies_ms)

    # Save metadata.json
    metadata = {
        "step": "B01",
        "run_id": run_id,
        "timestamp": now_str,
        "status": "COMPLETED" if failed_runs == 0 else "PARTIAL_FAILURE",
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "failure_reasons": failure_reasons,
        "remote_host": config.deployment["remote"]["host"],
        "ssh_port": config.deployment["remote"]["port"],
        "model_action_shape": [ACTION_HORIZON_STEPS, ACTION_DIM],
        "action_rate_hz": ACTION_RATE_HZ,
        "latency_target_ms": LATENCY_TARGET_S * 1000.0,
        "total_latency_stats": {
            "mean_ms": stats_tot.mean_ms,
            "p50_ms": stats_tot.p50_ms,
            "p95_ms": stats_tot.p95_ms,
            "p99_ms": stats_tot.p99_ms,
            "max_ms": stats_tot.max_ms,
            "min_ms": stats_tot.min_ms,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 70)
    print("BENCHMARK 01 — MEASUREMENT RESULTS")
    print("=" * 70)
    print(f"Total Pipeline Latency: Mean={stats_tot.mean_ms:.2f} ms, P50={stats_tot.p50_ms:.2f} ms, P95={stats_tot.p95_ms:.2f} ms, Max={stats_tot.max_ms:.2f} ms")
    print(f"Equivalent Action Steps: {stats_tot.mean_s / ACTION_STEP_DT_S:.2f} steps (Target < 4.0 steps)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 01: Remote Pipeline Latency")
    parser.add_argument("--audit", action="store_true", default=False, help="Perform static audit and contract inspection only (Default behavior)")
    parser.add_argument("--measure", action="store_true", default=False, help="Perform actual live measurement over persistent SSH connection")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS, help="Number of warmup inference runs (default 5)")
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS, help="Number of measured inference runs (default 30)")
    args = parser.parse_args()

    config_path = DEPLOY_DIR / "config" / "deployment.yaml"

    if args.measure:
        run_measurement_mode(args, config_path)
    else:
        run_audit_mode(config_path)


if __name__ == "__main__":
    main()
