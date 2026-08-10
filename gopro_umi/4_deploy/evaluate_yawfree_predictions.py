#!/usr/bin/env python3
"""Evaluate yaw-free SmolVLA predictions against recorded next-frame poses.

Inputs are the exact canonical Zarr RGB image and 6D state used for training.
This is an offline model-accuracy measurement only: no camera, robot, IK, or
motor-control API is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from replay_yawfree_shadow import ZarrV2Array
import deploy_smolvla_yawfree as deploy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--report", type=Path, default=Path("/tmp/yawfree_prediction_eval.json"))
    return parser.parse_args()


def summary(values: list[float], scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> int:
    args = parse_args()
    if args.samples_per_episode <= 0:
        raise ValueError("--samples-per-episode must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    dataset = deploy.PROJECT_ROOT / "2_dataset/replay_buffer_yawfree.zarr"
    images = ZarrV2Array(dataset / "data/camera0_rgb")
    pos = ZarrV2Array(dataset / "data/robot0_eef_pos").all()
    roll_pitch = ZarrV2Array(dataset / "data/robot0_eef_roll_pitch").all()
    gripper = ZarrV2Array(dataset / "data/robot0_gripper_width").all()
    ends = ZarrV2Array(dataset / "meta/episode_ends").all().astype(int)
    states = np.concatenate((pos, roll_pitch, gripper), axis=1).astype(np.float32)

    device = torch.device(args.device)
    deploy.validate_model_artifacts(deploy.MODEL_PATH)
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    samples: list[tuple[int, int, int]] = []
    start = 0
    fractions = np.linspace(0.25, 0.75, args.samples_per_episode)
    for episode, end in enumerate(ends):
        end = int(end)
        for fraction in fractions:
            frame = start + min(int((end - start - 1) * fraction), end - start - 2)
            samples.append((episode, frame, end))
        start = end

    rows: list[dict[str, object]] = []
    xyz_errors: list[float] = []
    rp_errors: list[float] = []
    gripper_errors: list[float] = []
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for ordinal, (episode, frame, episode_end) in enumerate(samples):
        torch.manual_seed(args.seed + ordinal)
        policy.reset()  # exactly one prediction; never consume a 50-action queue
        rgb = images.frame(frame)
        model_rgb = __import__("cv2").resize(rgb, (256, 256), interpolation=__import__("cv2").INTER_AREA)
        raw = np.asarray(
            predict_action(
                {"observation.images.camera1": model_rgb, "observation.state": states[frame]},
                policy, device, preprocessor, postprocessor, False,
                task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
            ).squeeze(0).cpu(),
            dtype=np.float64,
        )
        recorded = states[min(frame + 1, episode_end - 1)].astype(np.float64)
        xyz_error = float(np.linalg.norm(raw[:3] - recorded[:3]))
        rp_delta = (raw[3:5] - recorded[3:5] + np.pi) % (2 * np.pi) - np.pi
        rp_error = float(np.linalg.norm(rp_delta))
        gripper_error = float(abs(raw[5] - recorded[5]))
        xyz_errors.append(xyz_error)
        rp_errors.append(rp_error)
        gripper_errors.append(gripper_error)
        rows.append({
            "episode": episode + 1,
            "frame": frame,
            "input_state": states[frame].tolist(),
            "recorded_next_action": recorded.tolist(),
            "raw_model_action": raw.tolist(),
            "xyz_error_m": xyz_error,
            "roll_pitch_error_rad": rp_error,
            "gripper_abs_error": gripper_error,
        })
        if (ordinal + 1) % 5 == 0 or ordinal + 1 == len(samples):
            partial = {
                "completed": ordinal + 1,
                "total": len(samples),
                "xyz_median_cm": float(np.median(xyz_errors) * 100),
                "rp_median_deg": float(np.median(rp_errors) * 180 / np.pi),
            }
            print(partial, flush=True)
            args.report.write_text(json.dumps({"status": "running", "samples": rows, "partial": partial}, indent=2) + "\n")

    per_episode: dict[str, dict[str, float]] = {}
    for episode in range(1, len(ends) + 1):
        episode_rows = [row for row in rows if row["episode"] == episode]
        per_episode[str(episode)] = {
            "xyz_median_cm": float(np.median([row["xyz_error_m"] for row in episode_rows]) * 100),
            "roll_pitch_median_deg": float(np.median([row["roll_pitch_error_rad"] for row in episode_rows]) * 180 / np.pi),
            "gripper_median_abs": float(np.median([row["gripper_abs_error"] for row in episode_rows])),
        }
    report = {
        "status": "complete",
        "mode": "exact_zarr_rgb_and_6d_state_offline_model_evaluation",
        "samples_per_episode": args.samples_per_episode,
        "total_samples": len(rows),
        "xyz_error_cm": summary(xyz_errors, 100),
        "roll_pitch_error_deg": summary(rp_errors, 180 / np.pi),
        "gripper_abs_error": summary(gripper_errors),
        "per_episode": per_episode,
        "samples": rows,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("total_samples", "xyz_error_cm", "roll_pitch_error_deg", "gripper_abs_error")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import sys
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
