#!/usr/bin/env python3
"""Read the live encoders and compare FK TCP pose with training start poses.

This program is deliberately read-only: it opens the calibrated SO-100 serial
bus directly, reads ``Present_Position``, computes ``base_link -> tcp_link``
with the yaw-free candidate FK, and closes the bus without touching torque or
Goal_Position.  It does *not* open a camera, run SmolVLA, run IK, or write any
motor register.

By default it compares the current pose with the first valid row of every
episode's ``trajectory_tcp_robot.csv``.  That answers "which episode should be
the start?" from the pose data instead of arbitrarily choosing episode 1.
Use ``--episode N`` only when a particular recorded scene is intentionally
being recreated.  Use ``--all-frames`` to inspect proximity to the entire
training distribution; that is useful diagnostics, but is not a start-pose
selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy


TRAJECTORY_ROOT = (
    deploy.PROJECT_ROOT
    / "1_data_pipeline/actioncam-vio/Episode_result"
)


@dataclass(frozen=True)
class ReferencePose:
    episode: int
    frame: int
    timestamp_s: float
    xyz_m: np.ndarray
    roll_pitch_rad: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--episode", type=int,
        help="compare only this episode; default compares starts of every episode",
    )
    parser.add_argument(
        "--all-frames", action="store_true",
        help="compare every valid training frame instead of episode starts",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, help="optional JSON output path")
    return parser.parse_args()


def episode_number(path: Path) -> int:
    # .../episode_N/world/trajectory_tcp_robot.csv
    name = path.parent.parent.name
    if not name.startswith("episode_"):
        raise ValueError(f"unexpected trajectory path: {path}")
    return int(name.removeprefix("episode_"))


def load_references(args: argparse.Namespace) -> list[ReferencePose]:
    paths = sorted(TRAJECTORY_ROOT.glob("episode_*/world/trajectory_tcp_robot.csv"), key=episode_number)
    if args.episode is not None:
        paths = [path for path in paths if episode_number(path) == args.episode]
    if not paths:
        raise FileNotFoundError(f"no trajectory CSV found for selection under {TRAJECTORY_ROOT}")

    result: list[ReferencePose] = []
    for path in paths:
        episode = episode_number(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            found_valid = False
            for frame, row in enumerate(reader):
                if str(row.get("ok", "1")).strip().lower() not in ("1", "true", "1.0"):
                    continue
                quaternion = np.array(
                    [float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])],
                    dtype=np.float64,
                )
                if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                    continue
                # SciPy returns intrinsic ZYX as [yaw, pitch, roll].  Yaw is
                # intentionally excluded from the yaw-free comparison.
                euler = Rotation.from_quat(quaternion).as_euler("ZYX", degrees=False)
                result.append(
                    ReferencePose(
                        episode=episode,
                        frame=frame,
                        timestamp_s=float(row["timestamp"]),
                        xyz_m=np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64),
                        roll_pitch_rad=euler[[2, 1]],
                    )
                )
                found_valid = True
                if not args.all_frames:
                    break
            if not found_valid:
                print(f"WARNING: episode {episode} has no valid CSV row", file=sys.stderr)
    if not result:
        raise RuntimeError("no valid reference poses loaded")
    return result


def read_live_fk(port: str) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Return joint degrees, TCP xyz, and yaw-free [roll,pitch]. No bus write."""
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    solver = deploy.load_calibrated_solver(limit_margin_ratio=0.95)
    follower = SO100Follower(SO100FollowerConfig(port=port, use_degrees=True))
    # Do not call follower.connect(), because it configures motor registers.
    follower.bus.connect(handshake=True)
    try:
        observed = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        arm_names = [name for name in follower.bus.motors if name != "gripper"]
        q_rad = np.deg2rad(np.array([observed[name] for name in arm_names], dtype=np.float64))
        pose = solver.forward_kinematics(q_rad)
        euler = Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX", degrees=False)
        return observed, pose[:3, 3].copy(), euler[[2, 1]].copy()
    finally:
        # This only closes the serial connection; it does not change torque.
        follower.bus.disconnect(disable_torque=False)


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def classify(position_mm: float, roll_pitch_deg: float) -> str:
    if position_mm <= 10.0 and roll_pitch_deg <= 5.0:
        return "CLOSE: suitable pose candidate; confirm camera scene next"
    if position_mm <= 20.0 and roll_pitch_deg <= 10.0:
        return "BORDERLINE: alignment recommended before inference"
    return "FAR: do not treat as a matching training start"


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.episode is not None and args.episode <= 0:
        raise ValueError("--episode must be positive")

    references = load_references(args)
    joints_deg, current_xyz, current_rp = read_live_fk(args.port)
    ranked: list[dict[str, object]] = []
    for reference in references:
        delta_xyz = reference.xyz_m - current_xyz
        delta_rp = wrap_radians(reference.roll_pitch_rad - current_rp)
        position_mm = float(np.linalg.norm(delta_xyz) * 1000.0)
        roll_pitch_deg = float(np.linalg.norm(np.rad2deg(delta_rp)))
        # A transparent ranking only; it is not a safety or model-quality metric.
        score = position_mm / 10.0 + roll_pitch_deg / 5.0
        ranked.append({
            "episode": reference.episode,
            "frame": reference.frame,
            "timestamp_s": reference.timestamp_s,
            "reference_xyz_m": reference.xyz_m.tolist(),
            "reference_roll_pitch_deg": np.rad2deg(reference.roll_pitch_rad).tolist(),
            "target_minus_current_xyz_mm": (delta_xyz * 1000.0).tolist(),
            "target_minus_current_roll_pitch_deg": np.rad2deg(delta_rp).tolist(),
            "position_distance_mm": position_mm,
            "roll_pitch_distance_deg": roll_pitch_deg,
            "ranking_score": score,
            "assessment": classify(position_mm, roll_pitch_deg),
        })
    ranked.sort(key=lambda item: float(item["ranking_score"]))

    report = {
        "read_only": True,
        "motor_command": "NONE",
        "comparison": "robot_base TCP XYZ + Euler roll/pitch; yaw intentionally ignored",
        "reference_mode": "all_valid_frames" if args.all_frames else "first_valid_frame_per_episode",
        "references_considered": len(references),
        "current_joint_deg": joints_deg,
        "current_xyz_m": current_xyz.tolist(),
        "current_roll_pitch_deg": np.rad2deg(current_rp).tolist(),
        "nearest": ranked[:args.top_k],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"REPORT_WRITTEN: {args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
