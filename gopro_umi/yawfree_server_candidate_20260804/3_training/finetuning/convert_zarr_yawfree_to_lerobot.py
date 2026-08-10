#!/usr/bin/env python3
"""Convert yaw-free UMI Zarr to a separate 6D LeRobot dataset."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_PATH = Path("/home/kimminje/lerobot/src")
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))
from lerobot.datasets.lerobot_dataset import LeRobotDataset


AXES = ["x", "y", "z", "roll", "pitch", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=PROJECT_ROOT / "2_dataset" / "replay_buffer_yawfree.zarr")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "2_dataset" / "lerobot_dataset_yawfree")
    parser.add_argument("--repo-id", default="gopro_umi/smolvla_yawfree_dataset")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = zarr.open(str(args.zarr), mode="r")
    expected_attrs = {
        "pose_source": "trajectory_tcp_robot.csv",
        "pose_frame": "robot_base_tcp_roll_pitch_pose",
        "task_dof": "xyz_roll_pitch",
        "coordinate_transforms_applied_by_builder": "none",
    }
    for key, expected in expected_attrs.items():
        if root.attrs.get(key) != expected:
            raise ValueError(f"yaw-free Zarr attr {key!r} must be {expected!r}")
    images = root["data/camera0_rgb"]
    pos = np.asarray(root["data/robot0_eef_pos"][:], dtype=np.float32)
    roll_pitch = np.asarray(root["data/robot0_eef_roll_pitch"][:], dtype=np.float32)
    gripper = np.asarray(root["data/robot0_gripper_width"][:], dtype=np.float32)
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    frames = images.shape[0]
    episodes = len(ends)
    if frames <= 0 or images.shape[1:] != (224, 224, 3) or pos.shape != (frames, 3):
        raise ValueError("yaw-free image/position schema mismatch")
    if roll_pitch.shape != (frames, 2) or gripper.shape != (frames, 1):
        raise ValueError("yaw-free roll/pitch or gripper schema mismatch")
    if ends.ndim != 1 or episodes == 0 or int(ends[-1]) != frames or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("invalid yaw-free episode boundaries")
    states = np.concatenate((pos, roll_pitch, gripper), axis=1).astype(np.float32)
    if states.shape != (frames, 6) or not np.isfinite(states).all():
        raise ValueError("6D state contains NaN/inf or has wrong shape")
    if float(gripper.min()) < -1e-6 or float(gripper.max()) > 1.0 + 1e-6:
        raise ValueError("gripper must stay normalized in [0,1]")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}; use --overwrite to replace only yaw-free LeRobot data")
        if args.output.is_symlink():
            raise ValueError("refusing to remove a symlink")
        shutil.rmtree(args.output)
    features = {
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": {"axes": AXES}},
        "action": {"dtype": "float32", "shape": (6,), "names": {"axes": AXES}},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=60, features=features, robot_type="umi_tcp",
        root=args.output, use_videos=False,
    )
    from PIL import Image
    start = 0
    for episode_index, end in enumerate(ends):
        end = int(end)
        for frame_index in range(start, end):
            action = states[frame_index + 1] if frame_index + 1 < end else states[frame_index]
            dataset.add_frame({
                "observation.images.top": Image.fromarray(images[frame_index]),
                "observation.state": states[frame_index],
                "action": action,
                "task": "pick and place the target object",
            })
        dataset.save_episode()
        print(f"converted episode {episode_index + 1}/{episodes}", flush=True)
        start = end
    print(f"Wrote 6D LeRobot dataset: {args.output} ({episodes} episodes, {frames} frames)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
