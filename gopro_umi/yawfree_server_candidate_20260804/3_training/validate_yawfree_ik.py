#!/usr/bin/env python3
"""Gate yaw-free XYZ/roll/pitch data with the deployment DLS IK solver.

The source is the protected 7D Zarr.  Solver control keeps its global
rotation-vector X/Y error, while acceptance is evaluated independently in
Euler intrinsic ZYX coordinates and ignores yaw completely.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
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
CANDIDATE_IK_PATH = Path(__file__).resolve().parent.parent / "4_deploy" / "ik_solver_v7.py"
_ik_spec = importlib.util.spec_from_file_location("yawfree_candidate_ik", CANDIDATE_IK_PATH)
if _ik_spec is None or _ik_spec.loader is None:
    raise RuntimeError(f"cannot import candidate IK: {CANDIDATE_IK_PATH}")
_ik_module = importlib.util.module_from_spec(_ik_spec)
_ik_spec.loader.exec_module(_ik_module)
DLSInverseKinematicsV7 = _ik_module.DLSInverseKinematicsV7


EXPECTED_FRAMES = 55_751
EXPECTED_EPISODES = 56
POSITION_GATE_M = 0.020
EULER_ROLL_PITCH_GATE_RAD = 0.10
ANGLE_AXES = "ZYX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr",
        type=Path,
        default=PROJECT_ROOT / "2_dataset" / "replay_buffer.zarr",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "4_deploy" / "URDF" / "so_arm_with_gopro_final.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "3_training" / "yawfree_ik_validation.json",
    )
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-frames", type=int, help="smoke test only; cannot produce a passing gate")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--pitch-singularity-deg",
        type=float,
        default=85.0,
        help="flag source pitch at or above this absolute angle",
    )
    return parser.parse_args()


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def source_data_sha256(
    positions: np.ndarray,
    rotvecs: np.ndarray,
    gripper: np.ndarray,
    episode_ends: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("robot0_eef_pos", positions),
        ("robot0_eef_rot_axis_angle", rotvecs),
        ("robot0_gripper_width", gripper),
        ("episode_ends", episode_ends),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def validate_source(root: zarr.hierarchy.Group) -> tuple[np.ndarray, ...]:
    if root.attrs.get("pose_source") != "trajectory_tcp_robot.csv":
        raise ValueError("source Zarr pose_source must be trajectory_tcp_robot.csv")
    if root.attrs.get("pose_frame") != "robot_base_tcp_link_pose":
        raise ValueError("source Zarr pose_frame must be robot_base_tcp_link_pose")
    if root.attrs.get("coordinate_transforms_applied_by_builder") != "none":
        raise ValueError("source Zarr must not contain a builder coordinate transform")
    images = root["data/camera0_rgb"]
    positions = np.asarray(root["data/robot0_eef_pos"][:], dtype=np.float64)
    rotvecs = np.asarray(root["data/robot0_eef_rot_axis_angle"][:], dtype=np.float64)
    gripper = np.asarray(root["data/robot0_gripper_width"][:], dtype=np.float64)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    n = images.shape[0]
    if images.shape != (EXPECTED_FRAMES, 224, 224, 3):
        raise ValueError(f"expected RGB shape ({EXPECTED_FRAMES},224,224,3), got {images.shape}")
    if positions.shape != (n, 3) or rotvecs.shape != (n, 3) or gripper.shape != (n, 1):
        raise ValueError("source pose/gripper array shape mismatch")
    if episode_ends.shape != (EXPECTED_EPISODES,) or int(episode_ends[-1]) != n:
        raise ValueError("source must contain exactly 56 non-empty episodes and 55,751 frames")
    if np.any(np.diff(np.concatenate([[0], episode_ends])) <= 0):
        raise ValueError("source contains an empty or unordered episode")
    if not all(np.isfinite(x).all() for x in (positions, rotvecs, gripper)):
        raise ValueError("source contains NaN/inf")
    if float(gripper.min()) < -1e-6 or float(gripper.max()) > 1.0 + 1e-6:
        raise ValueError("source gripper must already be normalized to [0,1]")
    return positions, rotvecs, gripper, episode_ends


def main() -> int:
    args = parse_args()
    if not args.zarr.is_dir():
        raise FileNotFoundError(args.zarr)
    if not args.urdf.is_file():
        raise FileNotFoundError(args.urdf)
    root = zarr.open(str(args.zarr), mode="r")
    positions, rotvecs, gripper, episode_ends = validate_source(root)
    source_euler = Rotation.from_rotvec(rotvecs).as_euler(ANGLE_AXES, degrees=False)
    source_pitch = source_euler[:, 1]
    source_roll = source_euler[:, 2]
    singularity_mask = np.abs(np.rad2deg(source_pitch)) >= args.pitch_singularity_deg
    source_hash = source_data_sha256(positions, rotvecs, gripper, episode_ends)

    solver = DLSInverseKinematicsV7(str(args.urdf))
    joint_min = np.full(solver.n_joints, np.inf, dtype=np.float64)
    joint_max = np.full(solver.n_joints, -np.inf, dtype=np.float64)
    position_residuals: list[float] = []
    euler_residuals: list[float] = []
    strict_frames = 0
    gate_frames = 0
    clamp_events = 0
    physical_attempt_events = 0
    boundary_frames = 0
    physical_violation_frames = 0
    failure_indices: list[int] = []
    episodes: list[dict[str, object]] = []
    processed = 0
    started = time.monotonic()
    stop_at = len(positions) if args.max_frames is None else min(args.max_frames, len(positions))
    previous_end = 0

    for episode_index, episode_end_raw in enumerate(episode_ends):
        episode_end = min(int(episode_end_raw), stop_at)
        if previous_end >= stop_at:
            break
        q = np.zeros(solver.n_joints, dtype=np.float64)
        ep_pos: list[float] = []
        ep_rot: list[float] = []
        ep_failures: list[int] = []
        ep_clamps = 0
        ep_boundaries = 0
        ep_strict = 0
        for frame_index in range(previous_end, episode_end):
            current_pose = solver.forward_kinematics(q)
            current_yaw = Rotation.from_matrix(current_pose[:3, :3]).as_euler(ANGLE_AXES)[0]
            target_rotation = Rotation.from_euler(
                ANGLE_AXES,
                [current_yaw, source_pitch[frame_index], source_roll[frame_index]],
            ).as_matrix()
            q = solver.solve(
                positions[frame_index], target_rotation, q, max_iter=args.max_iter
            )
            solved_pose = solver.forward_kinematics(q)
            position_error = float(np.linalg.norm(positions[frame_index] - solved_pose[:3, 3]))
            solved_euler = Rotation.from_matrix(solved_pose[:3, :3]).as_euler(ANGLE_AXES)
            roll_error = float(wrap_angle(solved_euler[2] - source_roll[frame_index]))
            pitch_error = float(wrap_angle(solved_euler[1] - source_pitch[frame_index]))
            euler_error = float(np.hypot(roll_error, pitch_error))
            position_residuals.append(position_error)
            euler_residuals.append(euler_error)
            ep_pos.append(position_error)
            ep_rot.append(euler_error)
            strict_frames += int(solver.last_converged)
            ep_strict += int(solver.last_converged)
            clamp_events += solver.last_clamp_events
            ep_clamps += solver.last_clamp_events
            physical_attempt_events += solver.last_physical_limit_attempt_events
            joint_min = np.minimum(joint_min, q)
            joint_max = np.maximum(joint_max, q)

            at_boundary = False
            physical_violation = False
            for joint_index, name in enumerate(solver.joint_names):
                safe_lo, safe_hi = solver.safe_limits[name]
                physical_lo, physical_hi = solver.PHYSICAL_JOINT_LIMITS_RAD[name]
                at_boundary |= bool(
                    np.isclose(q[joint_index], safe_lo, atol=1e-6, rtol=0.0)
                    or np.isclose(q[joint_index], safe_hi, atol=1e-6, rtol=0.0)
                )
                physical_violation |= bool(
                    q[joint_index] < physical_lo - 1e-9
                    or q[joint_index] > physical_hi + 1e-9
                )
            boundary_frames += int(at_boundary)
            ep_boundaries += int(at_boundary)
            physical_violation_frames += int(physical_violation)
            frame_gate = position_error <= POSITION_GATE_M and euler_error <= EULER_ROLL_PITCH_GATE_RAD
            gate_frames += int(frame_gate)
            if not frame_gate:
                failure_indices.append(frame_index)
                ep_failures.append(frame_index)
            processed += 1
            if args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"frames={processed}/{stop_at} episode={episode_index + 1} "
                    f"pos={position_error:.5f}m rp={euler_error:.5f}rad "
                    f"clamps={clamp_events} elapsed={elapsed:.1f}s",
                    flush=True,
                )
        episodes.append(
            {
                "episode_index": episode_index,
                "start": previous_end,
                "end": episode_end,
                "frames": episode_end - previous_end,
                "strict_solver_converged_frames": ep_strict,
                "gate_passed_frames": (episode_end - previous_end) - len(ep_failures),
                "position_residual_m": stats(ep_pos),
                "euler_roll_pitch_residual_rad": stats(ep_rot),
                "clamp_events": ep_clamps,
                "safe_limit_boundary_frames": ep_boundaries,
                "failed_frame_indices": ep_failures,
            }
        )
        previous_end = int(episode_end_raw)

    full_scope = processed == EXPECTED_FRAMES and len(episodes) == EXPECTED_EPISODES
    pitch_singularity_frames = int(singularity_mask[:processed].sum())
    passed = bool(
        full_scope
        and gate_frames == EXPECTED_FRAMES
        and pitch_singularity_frames == 0
        and clamp_events == 0
        and boundary_frames == 0
        and physical_attempt_events == 0
        and physical_violation_frames == 0
        and max(position_residuals) <= POSITION_GATE_M
        and max(euler_residuals) <= EULER_ROLL_PITCH_GATE_RAD
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "validation_scope": "full" if full_scope else "smoke",
        "source_zarr": str(args.zarr.resolve()),
        "source_data_sha256": source_hash,
        "urdf": str(args.urdf.resolve()),
        "frames": processed,
        "episodes": len(episodes),
        "expected_frames": EXPECTED_FRAMES,
        "expected_episodes": EXPECTED_EPISODES,
        "rotation_convention": "scipy intrinsic uppercase ZYX; values [yaw,pitch,roll] radians",
        "yaw_policy": "target yaw comes from current sequential IK FK and is excluded from validation",
        "solver_control_error": "global rotation-vector X/Y",
        "validation_rotation_error": "wrapped Euler ZYX roll/pitch norm; yaw ignored",
        "strict_solver_converged_frames": strict_frames,
        "gate_passed_frames": gate_frames,
        "strict_solver_nonconverged_frames": processed - strict_frames,
        "gate_failed_frames": processed - gate_frames,
        "position_gate_m": POSITION_GATE_M,
        "euler_roll_pitch_gate_rad": EULER_ROLL_PITCH_GATE_RAD,
        "position_residual_m": stats(position_residuals),
        "euler_roll_pitch_residual_rad": stats(euler_residuals),
        "max_position_residual_m": float(max(position_residuals)),
        "max_euler_roll_pitch_residual_rad": float(max(euler_residuals)),
        "clamp_events": clamp_events,
        "physical_limit_attempt_events": physical_attempt_events,
        "safe_limit_boundary_frames": boundary_frames,
        "joint_limit_violations": physical_violation_frames,
        "pitch_singularity_threshold_deg": args.pitch_singularity_deg,
        "pitch_singularity_frames": pitch_singularity_frames,
        "source_pitch_deg": {
            "min": float(np.rad2deg(source_pitch[:processed]).min()),
            "max": float(np.rad2deg(source_pitch[:processed]).max()),
        },
        "joint_usage_rad": {
            name: {
                "min": float(joint_min[index]),
                "max": float(joint_max[index]),
                "safe_limit": [float(x) for x in solver.safe_limits[name]],
                "physical_limit": [float(x) for x in solver.PHYSICAL_JOINT_LIMITS_RAD[name]],
            }
            for index, name in enumerate(solver.joint_names)
        },
        "failed_frame_indices": failure_indices,
        "episode_summaries": episodes,
        "elapsed_seconds": float(time.monotonic() - started),
        "gate_requirements": {
            "full_55751_frames_56_episodes": full_scope,
            "all_frames_within_residual_thresholds": gate_frames == EXPECTED_FRAMES,
            "zero_pitch_singularity_frames": pitch_singularity_frames == 0,
            "zero_clamp_events": clamp_events == 0,
            "zero_safe_limit_boundary_frames": boundary_frames == 0,
            "zero_physical_limit_attempt_events": physical_attempt_events == 0,
            "zero_joint_limit_violations": physical_violation_frames == 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}: passed={passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
