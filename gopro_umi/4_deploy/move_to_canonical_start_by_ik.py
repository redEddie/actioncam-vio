#!/usr/bin/env python3
"""Solve and move to the fixed episode-12 canonical TCP start pose.

Target: episode 12 frame 0 XYZ, roll and pitch.  The live FK yaw is preserved,
as required by the yaw-free task.  The program has no camera or policy path.
It uses the URDF's 100% physical joint bounds (not the 90/95% software
margins), prints the solved arm angles first, and writes motor goals only with
``--execute``.  The gripper is never commanded.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy
import live_start_pose_alignment as alignment


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def solve_from_observed(solver, observed: dict[str, float], target_xyz: np.ndarray, target_rp: np.ndarray):
    current_q = np.deg2rad(np.array([observed[name] for name in solver.joint_names], dtype=np.float64))
    current_pose = solver.forward_kinematics(current_q)
    current_euler = Rotation.from_matrix(current_pose[:3, :3]).as_euler("ZYX", degrees=False)
    target_rotation = Rotation.from_euler(
        "ZYX", [current_euler[0], target_rp[1], target_rp[0]], degrees=False
    ).as_matrix()
    solved_q = solver.solve(target_xyz, target_rotation, current_q)
    solved_pose = solver.forward_kinematics(solved_q)
    solved_euler = Rotation.from_matrix(solved_pose[:3, :3]).as_euler("ZYX", degrees=False)
    pos_mm = float(np.linalg.norm(target_xyz - solved_pose[:3, 3]) * 1000.0)
    rp_error_deg = np.rad2deg(wrap_radians(solved_euler[[2, 1]] - target_rp))
    return current_q, current_pose, current_euler, solved_q, pos_mm, rp_error_deg


def main() -> int:
    args = parse_args()
    if args.duration_s < 4.0 or args.steps < 20:
        raise ValueError("duration must be >=4 seconds and steps >=20")
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    target_xyz, target_rp, _ = alignment.read_reference()
    # Use only actual URDF physical bounds; no reduced software margin.
    solver = deploy.load_calibrated_solver(limit_margin_ratio=1.0)
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        observed = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        current_q, current_pose, current_euler, solved_q, pos_mm, rp_error_deg = solve_from_observed(
            solver, observed, target_xyz, target_rp
        )
        impossible = solver.last_clamp_events > 0 or solver.last_physical_limit_attempt_events > 0
        print("TARGET: episode 12 frame 0 XYZ+roll/pitch; current yaw preserved")
        print("CURRENT_ARM_JOINT_DEG:", {name: round(observed[name], 3) for name in solver.joint_names})
        print("CURRENT_TCP_XYZ_M:", np.round(current_pose[:3, 3], 6))
        print("CURRENT_EULER_ZYX_DEG:", np.round(np.rad2deg(current_euler), 3))
        print("SOLVED_ARM_JOINT_DEG:", {name: round(float(np.rad2deg(solved_q[i])), 3) for i, name in enumerate(solver.joint_names)})
        print("JOINT_DELTA_DEG:", {name: round(float(np.rad2deg(solved_q[i] - current_q[i])), 3) for i, name in enumerate(solver.joint_names)})
        print(f"IK_CHECK: pos_residual={pos_mm:.3f} mm, euler_rp_residual_deg={np.round(rp_error_deg, 3)}, "
              f"clamp={solver.last_clamp_events}, physical_attempt={solver.last_physical_limit_attempt_events}")
        if impossible:
            raise RuntimeError("target requires a physical-limit clamp; no motor goal was written")
        if pos_mm > 1.0 or float(np.linalg.norm(rp_error_deg)) > 1.0:
            raise RuntimeError("IK cannot reproduce target XYZ+roll/pitch within 1 mm / 1 degree; no motor goal was written")
        if not args.execute:
            print("PREVIEW PASS. Add --execute to send the solved arm trajectory; gripper remains untouched.")
            return 0

        initial_raw = follower.bus.sync_read("Present_Position", normalize=False)
        # Hold all motors at their measured current values before beginning;
        # subsequent writes contain only the five arm motors, never gripper.
        follower.bus.sync_write("Goal_Position", initial_raw, normalize=False)
        follower.bus.enable_torque()
        interval = args.duration_s / args.steps
        for step in range(1, args.steps + 1):
            q_step = current_q + (step / args.steps) * (solved_q - current_q)
            degrees_by_id = {
                follower.bus.motors[name].id: float(np.rad2deg(q_step[index]))
                for index, name in enumerate(solver.joint_names)
            }
            raw_by_id = follower.bus._unnormalize(degrees_by_id)
            follower.bus.sync_write(
                "Goal_Position",
                {name: raw_by_id[follower.bus.motors[name].id] for name in solver.joint_names},
                normalize=False,
            )
            time.sleep(interval)
            if step == 1 or step % 10 == 0 or step == args.steps:
                actual = follower.bus.sync_read("Present_Position")
                desired_deg = np.rad2deg(q_step)
                tracking = {name: float(actual[name]) - desired_deg[index] for index, name in enumerate(solver.joint_names)}
                print({"step": step, "tracking_error_deg": {name: round(value, 3) for name, value in tracking.items()}})

        time.sleep(0.5)
        settled = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        _, settled_pose, settled_euler, _, _, _ = solve_from_observed(
            solver, settled, target_xyz, target_rp
        )
        actual_xyz_error_mm = (target_xyz - settled_pose[:3, 3]) * 1000.0
        actual_rp_error_deg = np.rad2deg(wrap_radians(settled_euler[[2, 1]] - target_rp))
        print("POST_MOVE_ACTUAL_TCP:", {
            "xyz_error_mm": np.round(actual_xyz_error_mm, 3).tolist(),
            "xyz_norm_mm": round(float(np.linalg.norm(actual_xyz_error_mm)), 3),
            "roll_pitch_error_deg": np.round(actual_rp_error_deg, 3).tolist(),
            "roll_pitch_norm_deg": round(float(np.linalg.norm(actual_rp_error_deg)), 3),
        })
        return 0
    finally:
        if follower.bus.is_connected and args.execute:
            now = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", now, normalize=False)
            print("FINAL_HOLD_CURRENT_RAW:", now)
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: no further arm goals will be sent.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
