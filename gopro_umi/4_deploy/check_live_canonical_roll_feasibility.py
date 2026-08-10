#!/usr/bin/env python3
"""Read-only IK feasibility test for the fixed canonical start pose.

It answers one question only: from the present encoder configuration, can the
URDF's five arm joints hold the *episode 12/frame 0* XYZ while changing to its
roll/pitch?  The target keeps the current FK yaw, exactly as yaw-free deploy
does.  The solver is run with physical (100%) limits, not the 90/95% software
margin, so this is a kinematic/physical-bound test rather than a safe-start
gate.  No motor register is written and no camera or policy is loaded.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy
import live_start_pose_alignment as alignment


def wrap_radians(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    # limit_margin_ratio=1.0 means only the URDF physical joint bounds remain.
    solver = deploy.load_calibrated_solver(limit_margin_ratio=1.0)
    target_xyz, target_rp, _ = alignment.read_reference()
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        observed = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
    finally:
        follower.bus.disconnect(disable_torque=False)

    current_q = np.deg2rad(np.array([observed[name] for name in solver.joint_names], dtype=np.float64))
    current_pose = solver.forward_kinematics(current_q)
    current_euler = Rotation.from_matrix(current_pose[:3, :3]).as_euler("ZYX", degrees=False)
    # yaw-free target: preserve live yaw; impose only recorded roll/pitch.
    target_rotation = Rotation.from_euler(
        "ZYX", [current_euler[0], target_rp[1], target_rp[0]], degrees=False
    ).as_matrix()
    solution_q = solver.solve(target_xyz, target_rotation, current_q)
    solution_pose = solver.forward_kinematics(solution_q)
    solution_euler = Rotation.from_matrix(solution_pose[:3, :3]).as_euler("ZYX", degrees=False)
    position_error_mm = float(np.linalg.norm(target_xyz - solution_pose[:3, 3]) * 1000.0)
    rp_error_deg = np.rad2deg(wrap_radians(solution_euler[[2, 1]] - target_rp))
    rp_error_norm_deg = float(np.linalg.norm(rp_error_deg))
    # This is deliberately much tighter than the green start-alignment box.
    physically_reachable = bool(
        position_error_mm <= 1.0
        and rp_error_norm_deg <= 1.0
        and solver.last_physical_limit_attempt_events == 0
        and solver.last_clamp_events == 0
    )
    report = {
        "read_only": True,
        "motor_command": "NONE",
        "target_reference": "episode_12 frame_0; XYZ+roll/pitch, current yaw preserved",
        "solver_limit_mode": "100 percent physical URDF limits; no 90/95 software margin",
        "current_joint_deg": {name: observed[name] for name in solver.joint_names},
        "current_xyz_m": current_pose[:3, 3].tolist(),
        "current_euler_zyx_deg": np.rad2deg(current_euler).tolist(),
        "target_xyz_m": target_xyz.tolist(),
        "target_roll_pitch_deg": np.rad2deg(target_rp).tolist(),
        "solved_joint_deg": np.rad2deg(solution_q).tolist(),
        "solved_xyz_m": solution_pose[:3, 3].tolist(),
        "solved_euler_zyx_deg": np.rad2deg(solution_euler).tolist(),
        "position_residual_mm": position_error_mm,
        "euler_roll_pitch_residual_deg": rp_error_deg.tolist(),
        "euler_roll_pitch_residual_norm_deg": rp_error_norm_deg,
        "strict_solver_converged_diagnostic": bool(solver.last_converged),
        "solver_rotation_vector_xy_residual_rad": float(solver.last_orientation_error),
        "clamp_events": int(solver.last_clamp_events),
        "physical_limit_attempt_events": int(solver.last_physical_limit_attempt_events),
        "physically_reachable_at_this_yaw": physically_reachable,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if physically_reachable else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
