#!/usr/bin/env python3
"""Replay exact yaw-free training inputs through SmolVLA without hardware.

The displayed image and 6D state are read directly from the canonical Zarr
files, not reconstructed from a camera or virtual joint pose.  Each overlay
shows the model input, raw model output, clipped deployment target, recorded
next-frame action, and an IK diagnostic.  No robot driver is imported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from numcodecs import get_codec
from scipy.spatial.transform import Rotation
import torch


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
sys.path.insert(0, str(DEPLOY_DIR))
import deploy_smolvla_yawfree as deploy  # noqa: E402


class ZarrV2Array:
    """Small read-only Zarr v2 reader that avoids requiring zarr<3 locally."""

    def __init__(self, path: Path):
        self.path = path
        self.meta = json.loads((path / ".zarray").read_text())
        self.shape = tuple(self.meta["shape"])
        self.chunks = tuple(self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.codec = get_codec(self.meta["compressor"])
        self._cached_chunk_index: int | None = None
        self._cached_chunk: np.ndarray | None = None

    def _chunk(self, index: int) -> np.ndarray:
        chunk_index = index // self.chunks[0]
        if self._cached_chunk_index != chunk_index:
            key = ".".join([str(chunk_index)] + ["0"] * (len(self.shape) - 1))
            raw = self.codec.decode((self.path / key).read_bytes())
            self._cached_chunk = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks)
            self._cached_chunk_index = chunk_index
        assert self._cached_chunk is not None
        return self._cached_chunk

    def frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        return self._chunk(index)[index % self.chunks[0]].copy()

    def all(self) -> np.ndarray:
        output = np.empty(self.shape, dtype=self.dtype)
        for index in range(self.shape[0]):
            output[index] = self.frame(index)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=1, help="one-based episode number [1,56]")
    parser.add_argument("--start-frame", type=int, default=0, help="zero-based frame offset within the selected episode")
    parser.add_argument("--stride", type=int, default=30, help="source-frame interval per inference")
    parser.add_argument("--max-frames", type=int, default=10, help="number of model inferences")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("/tmp/yawfree_shadow_episode_1.mp4"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/yawfree_shadow_episode_1.json"))
    return parser.parse_args()


def put_lines(image: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> np.ndarray:
    canvas = cv2.resize(image, (896, 896), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((360, 896, 3), dtype=np.uint8)
    for row, line in enumerate(lines):
        cv2.putText(panel, line, (12, 28 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack((canvas, panel))


def main() -> int:
    args = parse_args()
    if args.stride <= 0 or args.max_frames <= 0 or args.start_frame < 0:
        raise ValueError("--stride and --max-frames must be positive; --start-frame must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source = PROJECT_ROOT / "2_dataset/replay_buffer.zarr"
    yawfree = PROJECT_ROOT / "2_dataset/replay_buffer_yawfree.zarr"
    images = ZarrV2Array(yawfree / "data/camera0_rgb")
    positions = ZarrV2Array(yawfree / "data/robot0_eef_pos").all()
    roll_pitch = ZarrV2Array(yawfree / "data/robot0_eef_roll_pitch").all()
    gripper = ZarrV2Array(yawfree / "data/robot0_gripper_width").all()
    ends = ZarrV2Array(yawfree / "meta/episode_ends").all().astype(int)
    rotvec = ZarrV2Array(source / "data/robot0_eef_rot_axis_angle").all()
    if not 1 <= args.episode <= len(ends):
        raise ValueError(f"--episode must be in [1,{len(ends)}]")
    start = 0 if args.episode == 1 else int(ends[args.episode - 2])
    end = int(ends[args.episode - 1])
    if start + args.start_frame >= end:
        raise ValueError(f"--start-frame must be below this episode length ({end - start})")
    start += args.start_frame

    config = deploy.validate_model_artifacts(deploy.MODEL_PATH)
    device = torch.device(args.device)
    ik = deploy.load_calibrated_solver(limit_margin_ratio=0.90)
    policy_cls = __import__("lerobot.policies.smolvla.modeling_smolvla", fromlist=["SmolVLAPolicy"]).SmolVLAPolicy
    processors = __import__("lerobot.policies.factory", fromlist=["make_pre_post_processors"])
    control = __import__("lerobot.common.control_utils", fromlist=["predict_action"])
    policy = policy_cls.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    pre, post = processors.make_pre_post_processors(
        policy.config,
        pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (896, 1256))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open output video: {args.output}")
    limits = deploy.SafetyLimits()
    q = np.zeros(5, dtype=np.float64)
    rows: list[dict[str, object]] = []
    try:
        for ordinal, frame_index in enumerate(range(start, end, args.stride)):
            if ordinal >= args.max_frames:
                break
            state = np.concatenate((positions[frame_index], roll_pitch[frame_index], gripper[frame_index])).astype(np.float32)
            recorded = np.concatenate((
                positions[min(frame_index + 1, end - 1)],
                roll_pitch[min(frame_index + 1, end - 1)],
                gripper[min(frame_index + 1, end - 1)],
            )).astype(np.float32)
            rgb = images.frame(frame_index)
            model_rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            torch.manual_seed(20260804 + ordinal)
            policy.reset()
            raw = np.asarray(
                control.predict_action(
                    {"observation.images.camera1": model_rgb, "observation.state": state},
                    policy, device, pre, post, False,
                    task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
                ).squeeze(0).cpu(),
                dtype=np.float64,
            )
            target, clipped = deploy.clip_target(raw, state, limits)
            current_yaw = Rotation.from_rotvec(rotvec[frame_index]).as_euler("ZYX")[0]
            target_rotation = Rotation.from_euler("ZYX", [current_yaw, target[4], target[3]]).as_matrix()
            q = ik.solve(target[:3], target_rotation, q)
            solved = ik.forward_kinematics(q)
            solved_euler = Rotation.from_matrix(solved[:3, :3]).as_euler("ZYX")
            rp_delta = (solved_euler[[2, 1]] - target[[3, 4]] + np.pi) % (2 * np.pi) - np.pi
            pos_error = float(np.linalg.norm(target[:3] - solved[:3, 3]))
            rp_error = float(np.linalg.norm(rp_delta))
            boundary = deploy.at_safe_limit(ik, q)
            gate = bool(
                pos_error <= limits.position_gate_m
                and rp_error <= limits.euler_roll_pitch_gate_rad
                and ik.last_clamp_events == 0
                and ik.last_physical_limit_attempt_events == 0
                and not boundary
            )
            row = {
                "frame": frame_index,
                "input_state": state.tolist(),
                "recorded_next_action": recorded.tolist(),
                "raw_model_action": raw.tolist(),
                "clipped_target": target.tolist(),
                "ik_gate": gate,
                "position_residual_m": pos_error,
                "euler_roll_pitch_residual_rad": rp_error,
                "clamp_events": ik.last_clamp_events,
                "physical_limit_attempt_events": ik.last_physical_limit_attempt_events,
                "safe_limit_boundary": boundary,
            }
            rows.append(row)
            color = (0, 220, 0) if gate else (0, 0, 255)
            lines = [
                f"EP {args.episode}  source frame {frame_index}  IK {'PASS' if gate else 'BLOCK'}",
                "INPUT  [x y z roll pitch grip]: " + " ".join(f"{x:+.4f}" for x in state),
                "RECORDED NEXT:                 " + " ".join(f"{x:+.4f}" for x in recorded),
                "MODEL RAW:                     " + " ".join(f"{x:+.4f}" for x in raw),
                "CLIPPED TARGET:                " + " ".join(f"{x:+.4f}" for x in target),
                f"clip xyz={clipped['position_step_clipped']} rp={clipped['roll_pitch_step_clipped']} grip={clipped['gripper_clipped']}",
                f"IK residual position={pos_error*1000:.3f} mm  roll/pitch={rp_error:.6f} rad",
                f"strict={ik.last_converged} clamps={ik.last_clamp_events} physical_attempts={ik.last_physical_limit_attempt_events} boundary={boundary}",
                "Replay only: exact Zarr image/state input; no camera, robot, or motor API.",
            ]
            writer.write(put_lines(rgb, lines, color))
            print(f"frame={frame_index} gate={gate} pos_mm={pos_error*1000:.3f} rp={rp_error:.6f}", flush=True)
    finally:
        writer.release()
    report = {
        "mode": "exact_zarr_input_replay_no_hardware",
        "episode": args.episode,
        "episode_range": [start, end],
        "model_config_state_action": [
            config["input_features"]["observation.state"]["shape"],
            config["output_features"]["action"]["shape"],
        ],
        "frames": rows,
        "gate_passed_frames": sum(bool(row["ik_gate"]) for row in rows),
        "total_frames": len(rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote video: {args.output}\nwrote report: {args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
