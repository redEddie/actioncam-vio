#!/usr/bin/env python3
"""Derive a yaw-free 6D UMI Zarr from the protected 7D replay buffer.

No SLAM, video decoding, camera/TCP transform, or tag transform is performed.
The source rotation vector is converted through a rotation matrix to intrinsic
Euler ZYX ``[yaw, pitch, roll]``; only ``roll, pitch`` are retained.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def discover_project_root() -> Path:
    """Support both an activated script and an isolated candidate copy."""
    script_root = Path(__file__).resolve().parent.parent
    for root in (script_root, script_root.parent):
        if all((root / name).is_dir() for name in ("2_dataset", "3_training", "4_deploy")):
            return root
    raise RuntimeError("cannot locate project root containing 2_dataset/3_training/4_deploy")


PROJECT_ROOT = discover_project_root()
EXPECTED_FRAMES, EXPECTED_EPISODES = 55_751, 56


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=PROJECT_ROOT / "2_dataset" / "replay_buffer.zarr",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "2_dataset" / "replay_buffer_yawfree.zarr",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_source(root: zarr.Group) -> tuple[zarr.Array, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_attrs = {
        "pose_source": "trajectory_tcp_robot.csv",
        "pose_frame": "robot_base_tcp_link_pose",
        "coordinate_transforms_applied_by_builder": "none",
    }
    for key, expected in required_attrs.items():
        if root.attrs.get(key) != expected:
            raise ValueError(f"source Zarr attr {key!r} must be {expected!r}")
    images = root["data/camera0_rgb"]
    pos = np.asarray(root["data/robot0_eef_pos"][:], dtype=np.float32)
    rotvec = np.asarray(root["data/robot0_eef_rot_axis_angle"][:], dtype=np.float64)
    gripper = np.asarray(root["data/robot0_gripper_width"][:], dtype=np.float32)
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if images.shape != (EXPECTED_FRAMES, 224, 224, 3):
        raise ValueError(f"unexpected image shape: {images.shape}")
    if pos.shape != (EXPECTED_FRAMES, 3) or rotvec.shape != (EXPECTED_FRAMES, 3):
        raise ValueError("unexpected source pose shapes")
    if gripper.shape != (EXPECTED_FRAMES, 1) or ends.shape != (EXPECTED_EPISODES,):
        raise ValueError("unexpected source gripper/episode schema")
    if int(ends[-1]) != EXPECTED_FRAMES or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("source episode ends are invalid")
    if not np.isfinite(pos).all() or not np.isfinite(rotvec).all() or not np.isfinite(gripper).all():
        raise ValueError("source contains NaN/inf")
    if float(gripper.min()) < -1e-6 or float(gripper.max()) > 1.0 + 1e-6:
        raise ValueError("source gripper must already be normalized to [0,1]")
    # Explicitly go through rotation matrices: never remove rotvec_z directly.
    euler_zyx = Rotation.from_matrix(Rotation.from_rotvec(rotvec).as_matrix()).as_euler("ZYX")
    roll_pitch = np.column_stack((euler_zyx[:, 2], euler_zyx[:, 1])).astype(np.float32)
    if not np.isfinite(roll_pitch).all():
        raise ValueError("Euler conversion produced NaN/inf")
    return images, pos, roll_pitch, gripper, ends


def remove_output(output: Path) -> None:
    target = output.resolve()
    if target == Path(target.anchor) or target.suffix != ".zarr" or not target.is_dir() or target.is_symlink():
        raise ValueError(f"refusing unsafe output deletion: {output}")
    shutil.rmtree(target)


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    source = zarr.open(str(args.source), mode="r")
    images, pos, roll_pitch, gripper, ends = validate_source(source)
    print(
        f"Validated source: {EXPECTED_EPISODES} episodes, {EXPECTED_FRAMES} frames; "
        "rotvec -> matrix -> Euler ZYX -> [roll, pitch]."
    )
    print(
        "roll range=" f"[{roll_pitch[:, 0].min():.6f}, {roll_pitch[:, 0].max():.6f}] rad, "
        "pitch range=" f"[{roll_pitch[:, 1].min():.6f}, {roll_pitch[:, 1].max():.6f}] rad"
    )
    if args.dry_run:
        return
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {args.output}; use --overwrite to replace only this yaw-free output")
        remove_output(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(args.output))
    root = zarr.group(store=store)
    data, meta = root.create_group("data"), root.create_group("meta")
    image_chunks = images.chunks or (100, 224, 224, 3)
    image_out = data.create_dataset(
        "camera0_rgb", shape=images.shape, dtype=images.dtype, chunks=image_chunks,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    for start in range(0, EXPECTED_FRAMES, image_chunks[0]):
        end = min(start + image_chunks[0], EXPECTED_FRAMES)
        image_out[start:end] = images[start:end]
        if start % 5_000 == 0:
            print(f"  copied RGB frames {start}:{end}", flush=True)
    data.create_dataset("robot0_eef_pos", data=pos, chunks=(100, 3))
    data.create_dataset("robot0_eef_roll_pitch", data=roll_pitch, chunks=(100, 2))
    data.create_dataset("robot0_gripper_width", data=gripper, chunks=(100, 1))
    meta.create_dataset("episode_ends", data=ends)
    root.attrs.update({
        "pose_source": "trajectory_tcp_robot.csv",
        "pose_frame": "robot_base_tcp_roll_pitch_pose",
        "task_dof": "xyz_roll_pitch",
        "coordinate_transforms_applied_by_builder": "none",
        "rotation_conversion": "rotvec -> rotation_matrix -> Euler intrinsic ZYX [yaw,pitch,roll]; stored [roll,pitch] radians",
        "yaw_removed": True,
        "source_zarr": str(args.source.resolve()),
    })
    print(f"Wrote yaw-free Zarr: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
