#!/usr/bin/env python3
"""Build an IK/training Zarr dataset from validated robot-base TCP trajectories.

The pose source is deliberately *only*
``world/trajectory_tcp_robot.csv``.  It already represents
``robot_base -> tcp_link`` and must not be subjected to a camera offset, a
tag-to-robot transform, or the URDF ``root -> tcp_link`` transform again.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


POSE_COLUMNS = ("timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "ok")
OUTPUT_FRAME = "robot_base_tcp_link_pose"


@dataclass(frozen=True)
class EpisodeInput:
    name: str
    video: Path
    pose: Path
    gripper: Path
    validation: Path


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actioncam-root", type=Path, default=project_root / "1_capture" / "actioncam-vio",
                    help="directory containing Episode/ and Episode_result/")
    ap.add_argument("-o", "--output", type=Path,
                    default=project_root / "artifacts" / "zarr" / "replay_buffer.zarr")
    ap.add_argument("--episodes", nargs="+", metavar="EPISODE",
                    help="optional subset, e.g. episode_1 episode_2")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output Zarr store")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate sources and report them without decoding video or writing Zarr")
    return ap.parse_args()


def _load_episode_inputs(root: Path, requested: list[str] | None) -> list[EpisodeInput]:
    result_dir, video_dir = root / "Episode_result", root / "Episode"
    mapping_path = result_dir / "episode_mapping_log.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"missing episode mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text())
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"invalid or empty episode mapping: {mapping_path}")
    names = requested if requested is not None else sorted(mapping, key=lambda s: int(s.rsplit("_", 1)[-1]))
    unknown = sorted(set(names) - set(mapping))
    if unknown:
        raise ValueError(f"episodes absent from mapping: {', '.join(unknown)}")
    return [EpisodeInput(
        name=name,
        video=video_dir / mapping[name],
        pose=result_dir / name / "world" / "trajectory_tcp_robot.csv",
        gripper=result_dir / name / "gripper" / "gripper_width.csv",
        validation=result_dir / name / "world" / "tcp_robot_validation.json",
    ) for name in names]


def _validate_episode(item: EpisodeInput) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [str(p) for p in (item.video, item.pose, item.gripper, item.validation) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{item.name}: missing required input(s): {', '.join(missing)}")
    report = json.loads(item.validation.read_text())
    if not report.get("passed") or report.get("frames", {}).get("output") != OUTPUT_FRAME:
        raise ValueError(f"{item.name}: TCP provenance is not a passed {OUTPUT_FRAME} transform")

    pose, grip = pd.read_csv(item.pose), pd.read_csv(item.gripper)
    missing_pose = sorted(set(POSE_COLUMNS) - set(pose.columns))
    if missing_pose:
        raise ValueError(f"{item.name}: robot TCP CSV missing columns: {missing_pose}")
    if "t_s" not in grip or "width_norm" not in grip:
        raise ValueError(f"{item.name}: gripper CSV requires t_s and width_norm")
    pose = pose.loc[pose["ok"].astype(bool)].copy()
    pose = pose.sort_values("timestamp").drop_duplicates("timestamp")
    # Width extraction can contain an isolated failed image measurement.  It is
    # not a pose-frame conversion and can safely be omitted before interpolation.
    invalid_grip = ~np.isfinite(grip[["t_s", "width_norm"]].to_numpy(float)).all(axis=1)
    if invalid_grip.any():
        print(f"  {item.name}: dropping {int(invalid_grip.sum())} invalid gripper sample(s)")
    grip = grip.loc[~invalid_grip].sort_values("t_s").drop_duplicates("t_s")
    if len(pose) < 2 or len(grip) < 2:
        raise ValueError(f"{item.name}: need at least two valid TCP and gripper samples")
    numeric = pose[list(POSE_COLUMNS[:-1])].to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{item.name}: NaN/inf in valid TCP samples")
    quat = pose[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
    if np.any(np.linalg.norm(quat, axis=1) < 1e-8):
        raise ValueError(f"{item.name}: zero TCP quaternion")
    return pose, grip


def _sample_episode(item: EpisodeInput, pose: pd.DataFrame, grip: pd.DataFrame):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to build frames") from exc
    t_pose = pose["timestamp"].to_numpy(float)
    pos = pose[["x", "y", "z"]].to_numpy(float)
    quat = pose[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
    pos_interp = interp1d(t_pose, pos, axis=0, bounds_error=False)
    rot_interp = Slerp(t_pose, Rotation.from_quat(quat))
    grip_interp = interp1d(grip["t_s"].to_numpy(float), grip["width_norm"].to_numpy(float), bounds_error=False)

    cap = cv2.VideoCapture(str(item.video))
    if not cap.isOpened():
        raise RuntimeError(f"{item.name}: cannot open video: {item.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError(f"{item.name}: invalid video FPS")
    images, positions, rotations, widths = [], [], [], []
    idx = 0
    start, end = max(t_pose[0], float(grip["t_s"].iloc[0])), min(t_pose[-1], float(grip["t_s"].iloc[-1]))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if start <= t <= end:
            frame = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
            images.append(frame)
            positions.append(pos_interp(t))
            rotations.append(rot_interp(t).as_rotvec())
            widths.append(grip_interp(t))
        idx += 1
    cap.release()
    if not images:
        raise ValueError(f"{item.name}: no video frames overlap TCP and gripper timestamps")
    return (np.asarray(images, dtype=np.uint8), np.asarray(positions, dtype=np.float32),
            np.asarray(rotations, dtype=np.float32), np.asarray(widths, dtype=np.float32).reshape(-1, 1))


def _remove_existing_zarr(output: Path) -> None:
    """Remove only an explicitly named Zarr directory after ``--overwrite``."""
    target = output.resolve()
    if target == Path(target.anchor) or target.suffix != ".zarr" or not target.is_dir() or target.is_symlink():
        raise ValueError(f"refusing to remove non-directory or unsafe Zarr output: {output}")
    shutil.rmtree(target)


def main() -> None:
    args = _parse_args()
    items = _load_episode_inputs(args.actioncam_root.resolve(), args.episodes)
    validated = [(item, *_validate_episode(item)) for item in items]
    print(f"Validated {len(validated)} episode(s): source is {OUTPUT_FRAME}; no coordinate transform is applied.")
    if args.dry_run:
        for item, pose, grip in validated:
            print(f"  {item.name}: TCP={len(pose)}, gripper={len(grip)}, video={item.video.name}")
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output} (use --overwrite to replace it)")
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr>=2.16 is required to write the dataset") from exc
    if not hasattr(zarr, "DirectoryStore"):
        raise RuntimeError("this builder requires Zarr v2 (DirectoryStore); install zarr<3")

    all_img, all_pos, all_rot, all_grip, ends = [], [], [], [], []
    total = 0
    for item, pose, grip in validated:
        images, positions, rotations, widths = _sample_episode(item, pose, grip)
        all_img.append(images); all_pos.append(positions); all_rot.append(rotations); all_grip.append(widths)
        total += len(images); ends.append(total)
        print(f"  {item.name}: {len(images)} aligned frames")
    if args.output.exists():
        _remove_existing_zarr(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.group(store=zarr.DirectoryStore(str(args.output)))
    data, meta = root.create_group("data"), root.create_group("meta")
    codec = zarr.Blosc(cname="zstd", clevel=3)
    data.create_dataset("camera0_rgb", data=np.concatenate(all_img), chunks=(100, 224, 224, 3), compressor=codec)
    data.create_dataset("robot0_eef_pos", data=np.concatenate(all_pos), chunks=(100, 3))
    data.create_dataset("robot0_eef_rot_axis_angle", data=np.concatenate(all_rot), chunks=(100, 3))
    data.create_dataset("robot0_gripper_width", data=np.concatenate(all_grip), chunks=(100, 1))
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    root.attrs.update({"pose_source": "trajectory_tcp_robot.csv", "pose_frame": OUTPUT_FRAME,
                       "coordinate_transforms_applied_by_builder": "none"})
    print(f"Wrote {total} frames from {len(items)} episodes to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
