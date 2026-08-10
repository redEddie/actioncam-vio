#!/usr/bin/env python3
"""Yaw-free 6D SmolVLA deployment *dry-run* gate.

This is deliberately separate from ``deploy_smolvla.py`` (the protected 7D
deployment).  It never imports a robot driver and therefore cannot send a
motor command.  Its purpose is to verify the exact 6D deployment chain:

    current TCP FK -> 6D state -> SmolVLA -> yaw-preserving target rotation
    -> diagnostic 5D IK -> Euler roll/pitch safety gate

Use this file to establish a validated safe workspace before adding any
hardware-control integration in a separate, reviewed step.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
MODEL_PATH = PROJECT_ROOT / "yawfree_server_candidate_new"
URDF_PATH = DEPLOY_DIR / "URDF/so_arm_with_gopro_final.urdf"
CANDIDATE_IK_PATH = PROJECT_ROOT / "yawfree_server_candidate_20260804/4_deploy/ik_solver_v7.py"
ARM_CALIBRATION_PATH = PROJECT_ROOT / "smolvla_cache/lerobot/calibration/robots/so_follower/None.json"
JOINT_ZERO_PATH = DEPLOY_DIR / "yawfree_user_defined_joint_zero.json"
DEFAULT_IMAGE_PATH = DEPLOY_DIR / "camera_test_0.jpg"
EXPECTED_STEP = 20_000
AXES = ("x", "y", "z", "roll", "pitch", "gripper")
# User-selected autonomous endpoints. Dataset convention remains width
# 1=open and 0=closed; only this final motor conversion uses raw units.
GRIPPER_SAFE_OPEN_RAW = 1500.0
GRIPPER_SAFE_CLOSE_RAW = 600.0
TASK_DESCRIPTION = "pick and place the target object"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# Hugging Face reads these during import in some versions. They must be set
# before importing torch/LeRobot/transformers, otherwise a dry-run may make a
# network request even though every required asset is local.
os.environ["HF_HOME"] = str(PROJECT_ROOT / "smolvla_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SafetyLimits:
    """Conservative per-decision limits; all units are SI/radians."""

    max_position_step_m: float = 0.020
    max_roll_pitch_step_rad: float = 0.10
    max_gripper_step: float = 0.05
    position_gate_m: float = 0.020
    euler_roll_pitch_gate_rad: float = 0.10


def gripper_width_from_raw(
    raw_position: float,
    open_raw: float = GRIPPER_SAFE_OPEN_RAW,
    close_raw: float = GRIPPER_SAFE_CLOSE_RAW,
) -> float:
    """Convert the verified physical gripper endpoints to model width.

    The dataset convention is width 1=open and 0=closed.  The autonomous safe
    endpoints are 600=open and 3000=closed. These raw endpoints are deliberately
    separate from the policy output: the policy already produces normalized
    width and must not be normalized again.
    """
    return float(np.clip((close_raw - raw_position) / (close_raw - open_raw), 0.0, 1.0))


def gripper_raw_from_width(width: float) -> int:
    """Convert model width (1=open, 0=closed) into a safe raw motor goal."""
    if not np.isfinite(width):
        raise ValueError("gripper width must be finite")
    bounded = float(np.clip(width, 0.0, 1.0))
    return int(round(GRIPPER_SAFE_CLOSE_RAW - bounded * (GRIPPER_SAFE_CLOSE_RAW - GRIPPER_SAFE_OPEN_RAW)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument(
        "--current-joints-deg",
        default="0,0,0,0,0",
        help="five arm joints in IK joint order; only FK is used",
    )
    parser.add_argument(
        "--current-gripper",
        type=float,
        default=0.2832,
        help="already-normalized visual gripper width in [0,1] (not motor raw units)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-position-step-m", type=float, default=SafetyLimits.max_position_step_m)
    parser.add_argument("--max-roll-pitch-step-rad", type=float, default=SafetyLimits.max_roll_pitch_step_rad)
    parser.add_argument("--max-gripper-step", type=float, default=SafetyLimits.max_gripper_step)
    parser.add_argument("--report", type=Path, help="optional JSON report path; omitted means no file write")
    return parser.parse_args()


def parse_joint_degrees(value: str) -> np.ndarray:
    joints = np.fromstring(value, dtype=np.float64, sep=",")
    if joints.shape != (5,) or not np.isfinite(joints).all():
        raise ValueError("--current-joints-deg must be five finite comma-separated values")
    return joints


def load_user_joint_zero() -> dict[str, dict[str, float]]:
    """Load the operator-defined encoder-to-URDF zero without touching hardware."""
    if not JOINT_ZERO_PATH.is_file():
        raise FileNotFoundError(f"missing user joint-zero record: {JOINT_ZERO_PATH}")
    data = json.loads(JOINT_ZERO_PATH.read_text())
    joints = data.get("joints", {})
    if set(joints) != set(ARM_JOINTS):
        raise RuntimeError(f"invalid joint-zero record: {JOINT_ZERO_PATH}")
    result: dict[str, dict[str, float]] = {}
    for name in ARM_JOINTS:
        item = joints[name]
        sign = float(item["sign"])
        zero = float(item["encoder_zero_deg"])
        if sign not in (-1.0, 1.0) or not np.isfinite(zero):
            raise RuntimeError(f"invalid joint-zero entry for {name}")
        result[name] = {"sign": sign, "encoder_zero_deg": zero}
    return result


def encoder_degrees_to_urdf_degrees(encoder_deg: np.ndarray, joint_names=ARM_JOINTS) -> np.ndarray:
    mapping = load_user_joint_zero()
    values = np.asarray(encoder_deg, dtype=np.float64)
    if values.shape != (len(joint_names),):
        raise ValueError("joint vector has an unexpected shape")
    return np.asarray([mapping[name]["sign"] * (values[i] - mapping[name]["encoder_zero_deg"])
                       for i, name in enumerate(joint_names)], dtype=np.float64)


def urdf_degrees_to_encoder_degrees(urdf_deg: np.ndarray, joint_names=ARM_JOINTS) -> np.ndarray:
    mapping = load_user_joint_zero()
    values = np.asarray(urdf_deg, dtype=np.float64)
    if values.shape != (len(joint_names),):
        raise ValueError("joint vector has an unexpected shape")
    return np.asarray([values[i] * mapping[name]["sign"] + mapping[name]["encoder_zero_deg"]
                       for i, name in enumerate(joint_names)], dtype=np.float64)


def load_candidate_solver() -> type:
    """Load the diagnostic candidate without replacing the active IK module."""
    if not CANDIDATE_IK_PATH.is_file():
        raise FileNotFoundError(f"missing reviewed candidate IK: {CANDIDATE_IK_PATH}")
    spec = importlib.util.spec_from_file_location("yawfree_candidate_ik", CANDIDATE_IK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate IK: {CANDIDATE_IK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DLSInverseKinematicsV7


def load_calibrated_solver(limit_margin_ratio: float = 0.90) -> Any:
    """Create the yaw-free IK solver with measured arm travel ranges.

    LeRobot's degree normalization maps each measured raw min/max pair to a
    symmetric degree interval.  We treat the measured travel of pan, lift,
    elbow and wrist-flex as the usable physical interval.  Wrist-roll is not
    swept by ``lerobot-calibrate`` (it is assigned 0..4095 automatically), so
    its URDF limit remains authoritative.  This keeps yaw-free deployment
    consistent with the user's chosen calibration-based limit assumption
    without pretending the wrist-roll default is a measured mechanical limit.
    """
    if not 0.0 < limit_margin_ratio <= 1.0:
        raise ValueError("limit_margin_ratio must be in (0, 1]")
    if not ARM_CALIBRATION_PATH.is_file():
        raise FileNotFoundError(f"missing arm calibration: {ARM_CALIBRATION_PATH}")
    calibration = json.loads(ARM_CALIBRATION_PATH.read_text())
    zero_mapping = load_user_joint_zero()
    solver = load_candidate_solver()(str(URDF_PATH), limit_margin_ratio=limit_margin_ratio)
    limits = dict(solver.PHYSICAL_JOINT_LIMITS_RAD)
    measured_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
    for name in measured_names:
        entry = calibration.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"calibration has no {name} entry")
        raw_min, raw_max = float(entry["range_min"]), float(entry["range_max"])
        if not raw_max > raw_min:
            raise RuntimeError(f"invalid calibrated range for {name}: {raw_min}..{raw_max}")
        half_range_deg = (raw_max - raw_min) * 360.0 / (2.0 * 4095.0)
        sign = zero_mapping[name]["sign"]
        zero = zero_mapping[name]["encoder_zero_deg"]
        mapped = sorted((sign * (-half_range_deg - zero), sign * (half_range_deg - zero)))
        limits[name] = tuple(float(np.deg2rad(x)) for x in mapped)
    solver.PHYSICAL_JOINT_LIMITS_RAD = limits
    solver.safe_limits = {}
    for name in solver.joint_names:
        lo, hi = limits[name]
        center = (lo + hi) / 2.0
        half = (hi - lo) * limit_margin_ratio / 2.0
        solver.safe_limits[name] = (center - half, center + half)
    solver.limit_source = {
        "shoulder_pan": "active_calibration",
        "shoulder_lift": "active_calibration",
        "elbow_flex": "active_calibration",
        "wrist_flex": "active_calibration",
        "wrist_roll": "URDF_not_measured_by_lerobot_calibration",
    }
    return solver


def validate_model_artifacts(model_path: Path) -> dict[str, Any]:
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete yaw-free checkpoint: missing {missing}")

    config = json.loads((model_path / "config.json").read_text())
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if state_shape != [6] or action_shape != [6]:
        raise RuntimeError(f"refusing non-6D checkpoint: state={state_shape}, action={action_shape}")
    if config.get("chunk_size") != 50 or config.get("n_action_steps") != 50:
        raise RuntimeError("unexpected action-chunk configuration")

    # ``model_path`` is ``.../020000/pretrained_model``; the checkpoint's
    # training-state directory is its sibling under ``.../020000``.
    step_path = model_path.parent / "training_state/training_step.json"
    if step_path.is_file():
        step_data = json.loads(step_path.read_text())
        step = step_data if isinstance(step_data, int) else step_data.get("step")
        if step != EXPECTED_STEP:
            print(f"WARNING: checkpoint at step {step}; expected {EXPECTED_STEP}")
    else:
        print(f"INFO: no training_step proof at {step_path}, skipping step check")

    pre = json.loads((model_path / "policy_preprocessor.json").read_text())
    post = json.loads((model_path / "policy_postprocessor.json").read_text())
    serialized = json.dumps({"pre": pre, "post": post})
    if '"shape": [6]' not in serialized:
        raise RuntimeError("processor metadata does not contain required 6D features")
    return config


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rgb_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read RGB input image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Training samples were square-resized. The policy subsequently pads to its
    # configured 512x512 internal visual input.
    return cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)


def clip_target(
    action: np.ndarray, current_state: np.ndarray, limits: SafetyLimits
) -> tuple[np.ndarray, dict[str, bool]]:
    """Clip only per-decision motion; action remains absolute TCP semantics."""
    target = np.asarray(action, dtype=np.float64).copy()
    flags: dict[str, bool] = {}
    delta_xyz = target[:3] - current_state[:3]
    xyz_norm = float(np.linalg.norm(delta_xyz))
    clipped_xyz = (
        delta_xyz
        if xyz_norm <= limits.max_position_step_m
        else delta_xyz * (limits.max_position_step_m / xyz_norm)
    )
    target[:3] = current_state[:3] + clipped_xyz
    flags["position_step_clipped"] = xyz_norm > limits.max_position_step_m

    delta_rp = target[3:5] - current_state[3:5]
    rp_norm = float(np.linalg.norm(delta_rp))
    clipped_rp = (
        delta_rp
        if rp_norm <= limits.max_roll_pitch_step_rad
        else delta_rp * (limits.max_roll_pitch_step_rad / rp_norm)
    )
    target[3:5] = current_state[3:5] + clipped_rp
    flags["roll_pitch_step_clipped"] = rp_norm > limits.max_roll_pitch_step_rad

    raw_gripper = float(target[5])
    target[5] = np.clip(target[5], 0.0, 1.0)
    delta_gripper = float(target[5] - current_state[5])
    target[5] = current_state[5] + np.clip(
        delta_gripper, -limits.max_gripper_step, limits.max_gripper_step
    )
    flags["gripper_clipped"] = raw_gripper != float(target[5])
    return target, flags


def at_safe_limit(solver: Any, joints_rad: np.ndarray) -> bool:
    return any(
        np.isclose(joints_rad[index], solver.safe_limits[name][0], atol=1e-6, rtol=0.0)
        or np.isclose(joints_rad[index], solver.safe_limits[name][1], atol=1e-6, rtol=0.0)
        for index, name in enumerate(solver.joint_names)
    )


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    # This script has no follower import and no send_action call by design.
    if not URDF_PATH.is_file():
        raise FileNotFoundError(URDF_PATH)
    config = validate_model_artifacts(MODEL_PATH)
    device = get_device(args.device)
    limits = SafetyLimits(
        max_position_step_m=args.max_position_step_m,
        max_roll_pitch_step_rad=args.max_roll_pitch_step_rad,
        max_gripper_step=args.max_gripper_step,
    )
    if min(asdict(limits).values()) <= 0.0:
        raise ValueError("all safety limits must be positive")
    if not 0.0 <= args.current_gripper <= 1.0:
        raise ValueError("--current-gripper must already be normalized to [0,1]")

    solver = load_calibrated_solver(limit_margin_ratio=0.90)
    current_q = np.deg2rad(encoder_degrees_to_urdf_degrees(parse_joint_degrees(args.current_joints_deg)))
    outside_safe = [
        name
        for index, name in enumerate(solver.joint_names)
        if current_q[index] < solver.safe_limits[name][0]
        or current_q[index] > solver.safe_limits[name][1]
    ]
    if outside_safe:
        raise RuntimeError(
            "dry-run refuses a current joint state outside the 90% safe limits: "
            + ", ".join(outside_safe)
        )
    current_pose = solver.forward_kinematics(current_q)
    current_euler = Rotation.from_matrix(current_pose[:3, :3]).as_euler("ZYX", degrees=False)
    current_state = np.array(
        [
            current_pose[0, 3],
            current_pose[1, 3],
            current_pose[2, 3],
            current_euler[2],  # roll
            current_euler[1],  # pitch
            args.current_gripper,
        ],
        dtype=np.float32,
    )

    lerobot_src = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if lerobot_src.is_dir() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Select exactly one closed-loop decision. Never consume a 50-action queue
    # during dry-run or future motor integration without a deliberate review.
    policy.reset()
    action_tensor = predict_action(
        observation={
            "observation.images.camera1": load_rgb_image(args.image),
            "observation.state": current_state,
        },
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=device.type == "cuda",
        task=TASK_DESCRIPTION,
        robot_type="so_follower",
    )
    action = np.asarray(action_tensor.squeeze(0).detach().cpu(), dtype=np.float64)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise RuntimeError(f"invalid yaw-free action: shape={action.shape}, values={action}")

    clipped_action, clip_flags = clip_target(action, current_state, limits)
    # Intrinsic uppercase ZYX: yaw comes only from the current FK pose.
    target_rotation = Rotation.from_euler(
        "ZYX",
        [current_euler[0], clipped_action[4], clipped_action[3]],
        degrees=False,
    ).as_matrix()
    solved_q = solver.solve(clipped_action[:3], target_rotation, current_q)
    solved_pose = solver.forward_kinematics(solved_q)
    solved_euler = Rotation.from_matrix(solved_pose[:3, :3]).as_euler("ZYX", degrees=False)
    rp_error_components = (solved_euler[[2, 1]] - clipped_action[[3, 4]] + np.pi) % (2 * np.pi) - np.pi
    position_error = float(np.linalg.norm(clipped_action[:3] - solved_pose[:3, 3]))
    roll_pitch_error = float(np.linalg.norm(rp_error_components))
    boundary = at_safe_limit(solver, solved_q)
    gate_passed = bool(
        position_error <= limits.position_gate_m
        and roll_pitch_error <= limits.euler_roll_pitch_gate_rad
        and solver.last_clamp_events == 0
        and solver.last_physical_limit_attempt_events == 0
        and not boundary
    )

    return {
        "mode": "dry_run_only_no_motor_api_imported",
        "model_path": str(MODEL_PATH),
        "model_state_action_shape": [
            config["input_features"]["observation.state"]["shape"],
            config["output_features"]["action"]["shape"],
        ],
        "axes": list(AXES),
        "device": str(device),
        "image": str(args.image.resolve()),
        "current_state": current_state.tolist(),
        "raw_model_action": action.tolist(),
        "clipped_target_action": clipped_action.tolist(),
        "clip_flags": clip_flags,
        "target_yaw_rad_from_current_fk": float(current_euler[0]),
        "position_residual_m": position_error,
        "euler_roll_pitch_residual_rad": roll_pitch_error,
        "strict_solver_converged_diagnostic": bool(solver.last_converged),
        "clamp_events": int(solver.last_clamp_events),
        "physical_limit_attempt_events": int(solver.last_physical_limit_attempt_events),
        "safe_limit_boundary": boundary,
        "gate_passed": gate_passed,
        "solved_joints_deg": np.rad2deg(solved_q).tolist(),
        "safety_limits": asdict(limits),
        "motor_command": "BLOCKED_BY_DESIGN",
    }


def main() -> int:
    args = parse_args()
    report = run_dry_run(args)
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report: {args.report}")
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
