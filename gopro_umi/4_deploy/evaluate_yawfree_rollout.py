#!/usr/bin/env python3
"""Offline recursive yaw-free rollout; no camera or motor API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from replay_yawfree_shadow import ZarrV2Array
import deploy_smolvla_yawfree as deploy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--frame-stride", type=int, default=60,
                   help="Recorded-frame advance per decision; 60 matches a 1 Hz loop at 60 fps")
    p.add_argument("--start-fraction", type=float, default=0.25)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--report", type=Path, default=Path("/tmp/yawfree_rollout_77.json"))
    p.add_argument("--teacher-forced", action="store_true",
                   help="Also run the independent teacher-forced baseline (doubles inference time)")
    a = p.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = deploy.PROJECT_ROOT / "2_dataset/replay_buffer_yawfree.zarr"
    images = ZarrV2Array(root / "data/camera0_rgb")
    pos = ZarrV2Array(root / "data/robot0_eef_pos").all()
    rp = ZarrV2Array(root / "data/robot0_eef_roll_pitch").all()
    grip = ZarrV2Array(root / "data/robot0_gripper_width").all()
    ends = ZarrV2Array(root / "meta/episode_ends").all().astype(int)
    states = np.concatenate((pos, rp, grip), axis=1).astype(np.float32)
    deploy.validate_model_artifacts(deploy.MODEL_PATH)
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = torch.device(a.device)
    policy = SmolVLAPolicy.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    rows = []
    start = 0
    for ep, end_value in enumerate(ends[: a.episodes], start=1):
        end = int(end_value)
        start_frame = start + int((end - start - 1) * a.start_fraction)
        n = min(a.horizon, (end - start_frame - 1) // a.frame_stride)
        recursive = states[start_frame].copy()
        teacher_errors = []
        recursive_errors = []
        for k in range(n):
            frame = start_frame + k * a.frame_stride
            rgb = cv2.resize(images.frame(frame), (256, 256), interpolation=cv2.INTER_AREA)
            policy.reset()
            with torch.inference_mode():
                raw = np.asarray(predict_action(
                    {"observation.images.camera1": rgb, "observation.state": recursive},
                    policy, device, pre, post, False,
                    task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
                ).squeeze(0).detach().cpu(), dtype=np.float64)
            recorded_next = states[min(frame + a.frame_stride, end - 1)].astype(np.float64)
            recursive = raw.astype(np.float32)
            recursive[5] = np.clip(recursive[5], 0.0, 1.0)
            recursive_errors.append({
                "xyz_m": float(np.linalg.norm(recursive[:3] - recorded_next[:3])),
                "rp_rad": float(np.linalg.norm((recursive[3:5] - recorded_next[3:5] + np.pi) % (2*np.pi) - np.pi)),
                "gripper": float(abs(recursive[5] - recorded_next[5])),
            })
            if a.teacher_forced:
                # Teacher-forced reference: same image, but true recorded state each step.
                policy.reset()
                with torch.inference_mode():
                    teacher_raw = np.asarray(predict_action(
                        {"observation.images.camera1": rgb, "observation.state": states[frame]},
                        policy, device, pre, post, False,
                        task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
                    ).squeeze(0).detach().cpu(), dtype=np.float64)
                teacher_errors.append({
                    "xyz_m": float(np.linalg.norm(teacher_raw[:3] - recorded_next[:3])),
                    "rp_rad": float(np.linalg.norm((teacher_raw[3:5] - recorded_next[3:5] + np.pi) % (2*np.pi) - np.pi)),
                    "gripper": float(abs(np.clip(teacher_raw[5], 0, 1) - recorded_next[5])),
                })
            if (k + 1) % 1 == 0:
                a.report.parent.mkdir(parents=True, exist_ok=True)
                a.report.write_text(json.dumps({"status": "running", "episodes_completed": ep - 1,
                    "episode": ep, "step": k + 1, "horizon": n, "frame_stride": a.frame_stride,
                    "recursive_partial": recursive_errors, "teacher_forced_partial": teacher_errors}, indent=2) + "\n")
        rows.append({"episode": ep, "horizon": n, "frame_stride": a.frame_stride,
                     "recursive": recursive_errors, "teacher_forced": teacher_errors})
        start = end

    def stats(key: str, mode: str) -> dict[str, float]:
        vals = [x[key] for row in rows for x in row[mode]]
        arr = np.asarray(vals) * (1000 if key == "xyz_m" else 180/np.pi if key == "rp_rad" else 1)
        return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max()), "final_mean": float(np.mean([row[mode][-1][key] * (1000 if key == 'xyz_m' else 180/np.pi if key == 'rp_rad' else 1) for row in rows]))}

    report = {"mode": "recursive_state_rollout_vs_teacher_forced", "episodes": len(rows),
              "horizon": a.horizon, "frame_stride": a.frame_stride,
              "decision_period_s_at_60fps": a.frame_stride / 60.0,
              "recursive": {k: stats(k, "recursive") for k in ("xyz_m", "rp_rad", "gripper")},
              "teacher_forced": ({k: stats(k, "teacher_forced") for k in ("xyz_m", "rp_rad", "gripper")} if a.teacher_forced else None),
              "rows": rows}
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("mode", "episodes", "horizon", "recursive", "teacher_forced")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
