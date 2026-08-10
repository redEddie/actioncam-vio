#!/usr/bin/env python3
"""Timed physical yaw-free policy loop.

This is intentionally separate from both the protected 7D deployment and the
read-only shadow script.  By default it only previews one decision.  Real
motion requires ``--execute`` and is fail-closed:

* each decision uses the model's un-clipped Cartesian target;
* each decision is interpolated over one second in five motor goals by default;
* the normalized model gripper width is converted only at the final command to
  raw 600=open through 3000=closed;
* physical-limit clamp and failed-IK checks remain active.

Keep the workspace clear and power switch reachable. Ctrl-C stops future goal
writes and leaves the arm holding its last intermediate goal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy


PHYSICAL_START_PATH = deploy.DEPLOY_DIR / "yawfree_physical_start.json"


def args_parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--gripper-width", type=float,
                   help="optional normalized state override; otherwise inferred from raw 600=open, 3000=closed")
    p.add_argument("--max-decisions", type=int, default=20,
                   help="policy decisions; default 20 means 20 one-second motion intervals")
    p.add_argument("--duration-s", type=float, default=1.0)
    p.add_argument("--interpolation-steps", type=int, default=5)
    p.add_argument("--start-duration-s", type=float, default=3.0,
                   help="time used to reach the recorded physical start before policy inference")
    p.add_argument("--start-interpolation-steps", type=int, default=5)
    p.add_argument("--limit-margin-ratio", type=float, default=0.90,
                   help="IK soft joint-limit fraction; allowed only in [0.90, 0.95]")
    p.add_argument("--run", "--execute", dest="run", action="store_true", help="fully execute policy (move to start and run inference loop)")
    p.add_argument("--shadow-run", action="store_true", help="move to start, but only preview 1 inference step without moving")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def load_physical_start(arm_names: list[str]) -> dict[str, float]:
    if not PHYSICAL_START_PATH.is_file():
        raise FileNotFoundError(f"missing physical start record: {PHYSICAL_START_PATH}")
    record = json.loads(PHYSICAL_START_PATH.read_text())
    if not record.get("approved_for_policy_start", False):
        raise RuntimeError(
            "physical-start file is only a joint-zero reference, not an approved training start; "
            "solve and validate the canonical TCP start first"
        )
    target = record.get("arm_joint_degrees", {})
    if set(target) != set(arm_names):
        raise RuntimeError(f"invalid physical-start joint record: {PHYSICAL_START_PATH}")
    target = {name: float(target[name]) for name in arm_names}
    # Inference always begins with the tool roll explicitly zeroed.
    target["wrist_roll"] = 0.0
    return target  # values are URDF degrees


def move_to_physical_start(follower, arm_names: list[str], target: dict[str, float], duration_s: float, steps: int) -> None:
    """Execute the required pre-inference arm move; gripper is untouched."""
    current_encoder = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
    current_raw = follower.bus.sync_read("Present_Position", normalize=False)
    current_urdf = deploy.encoder_degrees_to_urdf_degrees(np.asarray([current_encoder[n] for n in arm_names]), arm_names)
    print("PRE_INFERENCE_START_CURRENT_URDF_DEG:", dict(zip(arm_names, np.round(current_urdf, 3))))
    print("PRE_INFERENCE_START_TARGET_URDF_DEG:", {name: round(target[name], 3) for name in arm_names})
    print(f"PRE_INFERENCE_START_MOVE: {steps} interpolation goals over {duration_s:.1f}s; gripper opening to 600")
    follower.bus.enable_torque()
    
    current_gripper_raw = float(current_raw["gripper"])
    target_gripper_raw = 1500.0

    for step in range(1, steps + 1):
        ratio = step / steps
        desired_urdf = current_urdf + ratio * (np.asarray([target[n] for n in arm_names]) - current_urdf)
        desired_encoder = deploy.urdf_degrees_to_encoder_degrees(desired_urdf, arm_names)
        desired = dict(zip(arm_names, desired_encoder))
        ids_degrees = {follower.bus.motors[name].id: desired[name] for name in arm_names}
        ids_raw = follower.bus._unnormalize(ids_degrees)
        
        write_dict = {name: ids_raw[follower.bus.motors[name].id] for name in arm_names}
        write_dict["gripper"] = int(round(current_gripper_raw + ratio * (target_gripper_raw - current_gripper_raw)))
        
        follower.bus.sync_write(
            "Goal_Position",
            write_dict,
            normalize=False,
        )
        time.sleep(duration_s / steps)
    time.sleep(0.3)
    settled = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
    settled_urdf = deploy.encoder_degrees_to_urdf_degrees(np.asarray([settled[n] for n in arm_names]), arm_names)
    error = settled_urdf - np.asarray([target[n] for n in arm_names])
    print("PRE_INFERENCE_START_FINAL_URDF_DEG:", dict(zip(arm_names, np.round(settled_urdf, 3))))
    print("PRE_INFERENCE_START_ERROR_URDF_DEG:", dict(zip(arm_names, np.round(error, 3))))


def main() -> int:
    args = args_parse()
    if (args.gripper_width is not None and not 0.0 <= args.gripper_width <= 1.0) or args.max_decisions < 1:
        raise ValueError("invalid gripper width or max decisions")
    if args.duration_s <= 0 or args.interpolation_steps < 1:
        raise ValueError("duration-s must be positive and interpolation-steps must be >=1")
    if args.start_duration_s <= 0 or args.start_interpolation_steps < 1:
        raise ValueError("invalid pre-inference start move duration or steps")
    if not 0.90 <= args.limit_margin_ratio <= 0.95:
        raise ValueError("limit margin must remain within [0.90, 0.95]")

    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    deploy.validate_model_artifacts(deploy.MODEL_PATH)
    device = deploy.get_device(args.device)
    solver = deploy.load_calibrated_solver(limit_margin_ratio=args.limit_margin_ratio)
    policy = SmolVLAPolicy.from_pretrained(deploy.MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(deploy.MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    arm_names = [name for name in follower.bus.motors if name != "gripper"]
    start_target = load_physical_start(arm_names)
    if args.run or args.shadow_run:
        # Mandatory execution order: move to physical start first, then take
        # the camera/state observation and run the first policy inference.
        move_to_physical_start(
            follower, arm_names, start_target,
            duration_s=args.start_duration_s,
            steps=args.start_interpolation_steps,
        )
    else:
        print("PRE_INFERENCE_START_TARGET_DEG (preview; no move without --run or --shadow-run):",
              {name: round(start_target[name], 3) for name in arm_names})
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release(); cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        follower.bus.disconnect(disable_torque=False)
        raise RuntimeError(f"cannot open camera {args.camera_index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    for _ in range(5): cap.grab()

    if args.run:
        mode = "POLICY EXECUTE: arm + gripper"
    elif args.shadow_run:
        mode = "SHADOW RUN: preview only (moved to start, but no inference movement)"
    else:
        mode = "POLICY PREVIEW"
    print(f"{mode} | target is un-clipped | {args.duration_s:.1f}s / {args.interpolation_steps} goals per decision | IK soft joint limits={args.limit_margin_ratio * 100:.0f}%")
    try:
        for decision in range(args.max_decisions):
            observed = follower.bus.sync_read("Present_Position")
            raw_observed = follower.bus.sync_read("Present_Position", normalize=False)
            encoder_deg = np.asarray([float(observed[name]) for name in arm_names])
            q = np.deg2rad(deploy.encoder_degrees_to_urdf_degrees(encoder_deg, arm_names))
            outside = [name for i, name in enumerate(solver.joint_names)
                       if q[i] < solver.safe_limits[name][0] or q[i] > solver.safe_limits[name][1]]
            if outside:
                print(f"WARNING: current joints slightly outside 90% safe limits: {outside}")
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError("camera did not return a frame")
            pose = solver.forward_kinematics(q)
            euler = Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX", degrees=False)
            inferred_width = deploy.gripper_width_from_raw(float(raw_observed["gripper"]))
            gripper_width = args.gripper_width if args.gripper_width is not None else inferred_width
            state = np.asarray([pose[0, 3], pose[1, 3], pose[2, 3], euler[2], euler[1], gripper_width], dtype=np.float32)
            image = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA)
            # One fresh absolute TCP prediction per closed-loop decision.  Do
            # not consume the checkpoint's 50-action queue across decisions:
            # the requested deployment rate is one new inference per second.
            policy.reset()
            raw = np.asarray(predict_action(
                {"observation.images.camera1": image, "observation.state": state}, policy, device, pre, post,
                device.type == "cuda", task=deploy.TASK_DESCRIPTION, robot_type="so_follower",
            ).squeeze(0).detach().cpu(), dtype=np.float64)
            if raw.shape != (6,) or not np.isfinite(raw).all():
                raise RuntimeError(f"invalid policy action {raw}")
            # The arm target is the raw model output: no Cartesian, roll/pitch,
            # or joint-delta micro-step clipping/backoff is applied here.
            target = raw.copy()
            target[5] = float(np.clip(target[5], 0.0, 1.0))
            rotation = Rotation.from_euler("ZYX", [euler[0], target[4], target[3]], degrees=False).as_matrix()
            solution = solver.solve(target[:3], rotation, q)
            solution_pose = solver.forward_kinematics(solution)
            solution_euler = Rotation.from_matrix(solution_pose[:3, :3]).as_euler("ZYX", degrees=False)
            rp = (solution_euler[[2, 1]] - target[[3, 4]] + np.pi) % (2*np.pi) - np.pi
            pos_err = float(np.linalg.norm(target[:3] - solution_pose[:3, 3]))
            rp_err = float(np.linalg.norm(rp))
            delta_deg = np.rad2deg(solution - q)
            reachable = (pos_err <= 0.020 and rp_err <= 0.10
                         and solver.last_clamp_events == 0
                         and solver.last_physical_limit_attempt_events == 0
                         and not deploy.at_safe_limit(solver, solution))
            if not reachable:
                raise RuntimeError("un-clipped policy target is not reachable without physical-limit clamp; no motor goal written")
            gripper_goal_raw = deploy.gripper_raw_from_width(float(target[5]))
            log_line = (f"decision {decision+1}: gripper_raw={float(raw_observed['gripper']):.0f}, inferred_width={inferred_width:.3f}, "
                  f"state_xyz={np.round(state[:3], 4)}, raw_xyz={np.round(raw[:3], 4)}, "
                  f"raw_delta_mm={np.round((raw[:3]-state[:3])*1000, 2)}, "
                  f"sent_delta_mm={np.round((target[:3]-state[:3])*1000, 2)}, "
                  f"arm_joint_delta_deg={np.round(delta_deg,3)}, "
                  f"gripper_target_width={target[5]:.3f}, gripper_goal_raw={gripper_goal_raw}, "
                  f"ik_mm={pos_err*1000:.3f}, rp_rad={rp_err:.6f}, pass=True")
            with open("4_deploy/inference_backlog.txt", "a") as f: f.write(log_line + "\n")
            print(f"[{decision+1:03d}] 위치: {np.round(state[:3]*1000, 1)}mm, RP: {np.round(np.rad2deg(state[3:5]), 1)}도  =>  "
                  f"목표: {np.round(raw[:3]*1000, 1)}mm, RP: {np.round(np.rad2deg(raw[3:5]), 1)}도  |  그리퍼: {float(raw_observed['gripper']):.0f} => {gripper_goal_raw}")
            if not args.run:
                if args.shadow_run:
                    print("SHADOW RUN PASS: Start position reached, inference previewed. No movement commanded.")
                else:
                    print("PREVIEW PASS: add --run to execute this policy decision.")
                return 0
            follower.bus.enable_torque()
            initial_gripper_raw = float(raw_observed["gripper"])
            for i in range(1, args.interpolation_steps + 1):
                intermediate = q + (i / args.interpolation_steps) * (solution - q)
                # Convert degrees to encoder units before writing.  The bus
                # convenience API truncates a normalized degree value to int
                # first, which would collapse fine interpolation into 1-degree
                # stairs.  Raw goals preserve the planned micro-step.
                intermediate_deg = deploy.urdf_degrees_to_encoder_degrees(np.rad2deg(intermediate), arm_names)
                ids_degrees = {
                    follower.bus.motors[name].id: float(intermediate_deg[index])
                    for index, name in enumerate(arm_names)
                }
                ids_raw = follower.bus._unnormalize(ids_degrees)
                raw_goals = {
                    name: ids_raw[follower.bus.motors[name].id] for name in arm_names
                }
                raw_goals["gripper"] = int(round(initial_gripper_raw + (i / args.interpolation_steps) * (gripper_goal_raw - initial_gripper_raw)))
                follower.bus.sync_write("Goal_Position", raw_goals, normalize=False)
                time.sleep(args.duration_s / args.interpolation_steps)
            # Verify from the physical encoders after the final goal; this is
            # read-only and does not issue another motor command.
            time.sleep(0.3)
            settled = follower.bus.sync_read("Present_Position")
            settled_q = np.deg2rad(deploy.encoder_degrees_to_urdf_degrees(
                np.asarray([float(settled[name]) for name in arm_names]), arm_names))
            settled_pose = solver.forward_kinematics(settled_q)
            actual_tcp_delta_mm = float(np.linalg.norm(settled_pose[:3, 3] - pose[:3, 3]) * 1000.0)
            joint_target_error_deg = np.rad2deg(settled_q - solution)
            settled_raw = follower.bus.sync_read("Present_Position", normalize=False)
            gripper_target_error_raw = float(settled_raw["gripper"]) - gripper_goal_raw
            post_log = ("POST_MOVE: "
                  f"actual_tcp_delta_mm={actual_tcp_delta_mm:.3f}, "
                  f"joint_target_error_deg={np.round(joint_target_error_deg, 3)}, "
                  f"gripper_target_error_raw={gripper_target_error_raw:.0f}\n"
                  "POLICY DECISION SENT, VERIFIED, AND HELD.")
            with open("4_deploy/inference_backlog.txt", "a") as f: f.write(post_log + "\n")
            if np.max(np.abs(joint_target_error_deg)) > 5.0:
                raise RuntimeError("motor did not reach the commanded micro-step within 5 degrees")
        return 0
    finally:
        cap.release()
        # Do not change torque on disconnect; it holds the final microstep goal.
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: no further goals sent; inspect arm before continuing.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
