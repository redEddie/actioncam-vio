#!/usr/bin/env python3
"""Read-only live shadow test for the yaw-free 6D UMI policy.

This is the first hardware-facing 6D test. It opens the camera and reads the
present joint positions, but deliberately has no motor command path:
``SO100Follower.connect()``, ``configure()``, ``send_action()``, torque writes,
and every bus write API are intentionally absent.  The serial bus is opened
directly in read-only mode and closed without disabling/enabling torque.

For every frame it evaluates:
camera + FK(current joints) -> 6D policy -> yaw-preserving target -> 5D IK
and displays the resulting safety decision. Press q or ESC to exit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frames", type=int, default=20, help="0 means run until q/ESC")
    parser.add_argument("--gripper-width", type=float,
                        help="optional diagnostic override; normally inferred from verified raw endpoints")
    parser.add_argument("--allow-outside-safe-for-diagnostics", action="store_true",
                        help="continue read-only inference from an unsafe start, but force every result to BLOCK")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def overlay(frame_bgr: np.ndarray, lines: list[str], ok: bool) -> np.ndarray:
    canvas = cv2.resize(frame_bgr, (960, 540), interpolation=cv2.INTER_AREA)
    panel = np.zeros((250, 960, 3), dtype=np.uint8)
    colour = (0, 210, 0) if ok else (0, 0, 235)
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (12, 28 + index * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.57, colour, 1, cv2.LINE_AA)
    return np.vstack((canvas, panel))


def main() -> int:
    args = parse_args()
    if args.frames < 0 or (args.gripper_width is not None and not 0.0 <= args.gripper_width <= 1.0):
        raise ValueError("--frames must be >= 0 and --gripper-width must be in [0,1]")

    # Configure import paths before importing LeRobot.  We intentionally use
    # SO100Follower only as a container for its calibrated read-only bus.
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = deploy.validate_model_artifacts(deploy.MODEL_PATH)
    device = deploy.get_device(args.device)
    solver = deploy.load_calibrated_solver(limit_margin_ratio=0.90)
    policy = SmolVLAPolicy.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    # Do NOT call follower.connect(): it invokes configure(), which writes motor settings.
    follower.bus.connect(handshake=True)
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        follower.bus.disconnect(disable_torque=False)
        raise RuntimeError(f"cannot open camera index {args.camera_index}")
    # This capture board needs the same MJPG/1080p setup and warm-up already
    # used by the project's proven camera probe.  These are camera-only
    # settings; no robot bus setting is changed.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(5):
        cap.grab()

    arm_names = [name for name in follower.bus.motors if name != "gripper"]
    limits = deploy.SafetyLimits()
    print("LIVE SHADOW: read-only. No motor command API is present. Press q/ESC to exit.")
    try:
        count = 0
        while args.frames == 0 or count < args.frames:
            # This is a read request only. Normalized arm values are degrees.
            observed = follower.bus.sync_read("Present_Position")
            raw_observed = follower.bus.sync_read("Present_Position", normalize=False)
            q = np.deg2rad(np.asarray([float(observed[name]) for name in arm_names], dtype=np.float64))
            current_pose = solver.forward_kinematics(q)
            current_euler = Rotation.from_matrix(current_pose[:3, :3]).as_euler("ZYX", degrees=False)
            inferred_width = deploy.gripper_width_from_raw(float(raw_observed["gripper"]))
            gripper_width = args.gripper_width if args.gripper_width is not None else inferred_width
            state = np.asarray([
                current_pose[0, 3], current_pose[1, 3], current_pose[2, 3],
                current_euler[2], current_euler[1], gripper_width,
            ], dtype=np.float32)
            outside = [
                (name, float(np.rad2deg(q[i])),
                 float(np.rad2deg(solver.safe_limits[name][0])),
                 float(np.rad2deg(solver.safe_limits[name][1])))
                for i, name in enumerate(solver.joint_names)
                if q[i] < solver.safe_limits[name][0] or q[i] > solver.safe_limits[name][1]
            ]
            start_outside_safe = bool(outside)
            if outside:
                print("CURRENT JOINTS DEG:", {name: round(float(observed[name]), 3) for name in arm_names})
                print("OUTSIDE 90% IK SAFE LIMITS (joint, current_deg, min_deg, max_deg):", outside)
                if not args.allow_outside_safe_for_diagnostics:
                    raise RuntimeError("current joints are outside the IK 90% safe limits; refusing inference")
                print("DIAGNOSTIC ONLY: continuing without motor control; final gate will be forced BLOCK")

            ok_frame, frame_bgr = cap.read()
            if not ok_frame:
                raise RuntimeError(
                    f"camera index {args.camera_index} opened but returned no frame; try another camera index"
                )

            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_input = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_AREA)
            policy.reset()  # one decision only; never consume a 50-action queue.
            output = predict_action(
                {"observation.images.camera1": image_input, "observation.state": state},
                policy, device, preprocessor, postprocessor, device.type == "cuda",
                task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
            )
            raw = np.asarray(output.squeeze(0).detach().cpu(), dtype=np.float64)
            if raw.shape != (6,) or not np.isfinite(raw).all():
                raise RuntimeError(f"invalid 6D policy output: {raw}")
            target, clipped = deploy.clip_target(raw, state, limits)
            target_rotation = Rotation.from_euler(
                "ZYX", [current_euler[0], target[4], target[3]], degrees=False
            ).as_matrix()
            solved_q = solver.solve(target[:3], target_rotation, q)
            solved_pose = solver.forward_kinematics(solved_q)
            solved_euler = Rotation.from_matrix(solved_pose[:3, :3]).as_euler("ZYX", degrees=False)
            rp = (solved_euler[[2, 1]] - target[[3, 4]] + np.pi) % (2 * np.pi) - np.pi
            pos_error = float(np.linalg.norm(target[:3] - solved_pose[:3, 3]))
            rp_error = float(np.linalg.norm(rp))
            boundary = deploy.at_safe_limit(solver, solved_q)
            passed = bool(pos_error <= limits.position_gate_m and rp_error <= limits.euler_roll_pitch_gate_rad
                          and solver.last_clamp_events == 0 and solver.last_physical_limit_attempt_events == 0
                          and not boundary and not start_outside_safe)
            lines = [
                f"LIVE SHADOW | {'GATE PASS' if passed else 'GATE BLOCK'} | motor command: IMPOSSIBLE BY DESIGN",
                "state xyz=" + " ".join(f"{v:+.3f}" for v in state[:3]) + f"  roll/pitch={state[3]:+.3f}/{state[4]:+.3f}",
                "model xyz=" + " ".join(f"{v:+.3f}" for v in raw[:3]) + f"  roll/pitch={raw[3]:+.3f}/{raw[4]:+.3f}",
                f"clipped xyz={clipped['position_step_clipped']} rp={clipped['roll_pitch_step_clipped']} grip={clipped['gripper_clipped']}",
                f"IK residual: position={pos_error*1000:.3f} mm, roll/pitch={rp_error:.6f} rad",
                f"clamp={solver.last_clamp_events} physical_attempt={solver.last_physical_limit_attempt_events} boundary={boundary}",
                f"start_outside_safe={start_outside_safe}",
                f"gripper raw={float(raw_observed['gripper']):.0f}; inferred width={inferred_width:.3f}; model gripper={raw[5]:.3f}",
            ]
            print(" | ".join(lines[:1] + lines[4:6]))
            cv2.imshow("Yaw-free UMI live shadow (read-only)", overlay(frame_bgr, lines, passed))
            count += 1
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Explicit false: do not issue the library's torque-disable write on exit.
        follower.bus.disconnect(disable_torque=False)
    print(f"completed {count} read-only live-shadow decisions using 6D config {config['output_features']['action']['shape']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
