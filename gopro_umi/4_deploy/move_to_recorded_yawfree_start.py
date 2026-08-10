#!/usr/bin/env python3
"""Move the five arm joints to a previously measured canonical yaw-free start.

The target is created only by ``live_start_pose_alignment.py`` after its
encoder-FK comparison passed.  This program does not use a camera or policy.
It interpolates raw motor encoder goals, then recomputes FK from the actual
encoders and succeeds only when the saved TCP XYZ and roll/pitch gates pass.
The gripper is deliberately held at its present value, not replayed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy


RECORD_PATH = deploy.DEPLOY_DIR / "yawfree_canonical_start_record.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--execute", action="store_true", help="required before any torque/goal write")
    return parser.parse_args()


def load_record() -> dict:
    if not RECORD_PATH.is_file():
        raise FileNotFoundError(
            f"no recorded canonical start: {RECORD_PATH}; run live_start_pose_alignment.py until green first"
        )
    record = json.loads(RECORD_PATH.read_text())
    if record.get("format") != "yawfree_canonical_start_v1":
        raise RuntimeError("unrecognized canonical-start record format")
    measured = record.get("measured_at_record", {})
    required = {"arm_joint_order", "arm_joint_deg", "fk_xyz_m", "fk_roll_pitch_deg"}
    if not required <= set(measured):
        raise RuntimeError("incomplete canonical-start record")
    return record


def wrap_radians(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi


def check_fk(record: dict, solver, observed: dict[str, float]) -> tuple[bool, dict]:
    order = list(solver.joint_names)
    q = np.deg2rad(np.array([float(observed[name]) for name in order], dtype=np.float64))
    pose = solver.forward_kinematics(q)
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX", degrees=False)
    current_xyz = pose[:3, 3]
    current_rp_deg = np.rad2deg(euler[[2, 1]])
    reference = record["reference"]
    tolerance = record["tolerance"]
    target_xyz = np.asarray(reference["target_xyz_m"], dtype=np.float64)
    target_rp_deg = np.asarray(reference["target_roll_pitch_deg"], dtype=np.float64)
    xyz_error_mm = (target_xyz - current_xyz) * 1000.0
    rp_error_deg = np.rad2deg(wrap_radians(np.deg2rad(target_rp_deg - current_rp_deg)))
    component_ok = (
        np.all(np.abs(xyz_error_mm) <= float(tolerance["xyz_axis_mm"]))
        and np.all(np.abs(rp_error_deg) <= float(tolerance["roll_pitch_axis_deg"]))
    )
    vector_ok = (
        float(np.linalg.norm(xyz_error_mm)) <= float(tolerance["xyz_vector_mm"])
        and float(np.linalg.norm(rp_error_deg)) <= float(tolerance["roll_pitch_vector_deg"])
    )
    details = {
        "actual_xyz_m": current_xyz.tolist(),
        "actual_roll_pitch_deg": current_rp_deg.tolist(),
        "target_minus_actual_xyz_mm": xyz_error_mm.tolist(),
        "target_minus_actual_roll_pitch_deg": rp_error_deg.tolist(),
        "xyz_vector_error_mm": float(np.linalg.norm(xyz_error_mm)),
        "roll_pitch_vector_error_deg": float(np.linalg.norm(rp_error_deg)),
        "component_gate": bool(component_ok),
        "vector_gate": bool(vector_ok),
    }
    return bool(component_ok and vector_ok), details


def main() -> int:
    args = parse_args()
    if args.duration_s < 10.0 or args.steps < 50:
        raise ValueError("recorded-start move requires duration >=10s and steps >=50")
    record = load_record()
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    solver = deploy.load_calibrated_solver(limit_margin_ratio=0.95)
    target_order = record["measured_at_record"]["arm_joint_order"]
    if target_order != list(solver.joint_names):
        raise RuntimeError(f"joint order mismatch: record={target_order}, solver={solver.joint_names}")
    target_deg = {name: float(record["measured_at_record"]["arm_joint_deg"][name]) for name in solver.joint_names}
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        print("RECORDED_TARGET_DEG:", {name: round(target_deg[name], 3) for name in solver.joint_names})
        print("CURRENT_DEG:", {name: round(current[name], 3) for name in follower.bus.motors})
        print("UNCHANGED: gripper")
        print(f"PLAN: {args.steps} raw-encoder interpolated arm goals over {args.duration_s:.1f}s")
        if not args.execute:
            print("DRY PLAN ONLY. Add --execute to move to the recorded encoder pose.")
            return 0

        initial_raw = follower.bus.sync_read("Present_Position", normalize=False)
        # Clear stale arm goals before enabling torque.  The gripper remains
        # held at the value it already has and is never commanded afterwards.
        follower.bus.sync_write("Goal_Position", initial_raw, normalize=False)
        follower.bus.enable_torque()
        interval = args.duration_s / args.steps
        for step in range(1, args.steps + 1):
            fraction = step / args.steps
            desired = {
                name: current[name] + fraction * (target_deg[name] - current[name])
                for name in solver.joint_names
            }
            ids_deg = {follower.bus.motors[name].id: desired[name] for name in solver.joint_names}
            ids_raw = follower.bus._unnormalize(ids_deg)
            follower.bus.sync_write(
                "Goal_Position",
                {name: ids_raw[follower.bus.motors[name].id] for name in solver.joint_names},
                normalize=False,
            )
            time.sleep(interval)
            if step == 1 or step % 10 == 0 or step == args.steps:
                actual = follower.bus.sync_read("Present_Position")
                tracking = {name: float(actual[name]) - desired[name] for name in solver.joint_names}
                print({"step": step, "tracking_error_deg": {name: round(value, 3) for name, value in tracking.items()}})

        time.sleep(0.5)
        settled = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        passed, details = check_fk(record, solver, settled)
        print("POST_MOVE_FK:", json.dumps(details, ensure_ascii=False))
        if not passed:
            raise RuntimeError("recorded-start FK gate failed; inference must not start from this measured pose")
        print("RECORDED_START_REACHED_AND_VERIFIED")
        return 0
    finally:
        if follower.bus.is_connected and args.execute:
            # Stop pursuing the last interpolation point if a failure occurs.
            now = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", now, normalize=False)
            print("FINAL_HOLD_CURRENT_RAW:", now)
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: current pose is held before disconnect.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
