#!/usr/bin/env python3
"""One-time, supervised move from the known folded start pose to a safe pose.

This script performs real motor writes only when ``--execute`` is passed. It
does not load a policy, camera, or IK target. It preserves shoulder_pan,
wrist_roll, and gripper; it changes only shoulder_lift, elbow_flex, and
wrist_flex by slow linear interpolation.

Emergency stop: Ctrl-C stops issuing new goals and leaves torque enabled at
the last commanded intermediate position. Keep the power switch reachable and
the motion volume clear throughout this supervised move.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

TARGET_DEG = {
    "shoulder_lift": -80.0,
    "elbow_flex": 75.0,
    "wrist_flex": 35.0,
}
# Refuse to move unless the arm is still in the explicitly inspected folded
# start pose. This avoids applying a stale command after somebody moved it.
EXPECTED_START_DEG = {
    "shoulder_pan": (3.165, 8.0),
    "shoulder_lift": (-108.264, 8.0),
    "elbow_flex": (101.582, 8.0),
    "wrist_flex": (51.297, 8.0),
    "wrist_roll": (-1.802, 8.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--execute", action="store_true", help="required to enable torque and command motion")
    return parser.parse_args()


def read_degrees(bus) -> dict[str, float]:
    values = bus.sync_read("Present_Position")
    return {name: float(value) for name, value in values.items()}


def main() -> int:
    args = parse_args()
    if args.duration_s < 4.0 or args.steps < 20:
        raise ValueError("refusing fast motion: --duration-s must be >=4 and --steps >=20")

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    # Avoid follower.connect(): it would configure motor registers. We use the
    # calibrated bus directly and make only the explicit torque/goal writes below.
    follower.bus.connect(handshake=True)
    try:
        current = read_degrees(follower.bus)
        print("CURRENT DEG:", {name: round(current[name], 3) for name in follower.bus.motors})
        mismatches = [
            (name, current[name], expected, tolerance)
            for name, (expected, tolerance) in EXPECTED_START_DEG.items()
            if abs(current[name] - expected) > tolerance
        ]
        if mismatches:
            raise RuntimeError("refusing stale/unknown start pose: " + repr(mismatches))

        print("TARGET DEG (only these three joints change):", TARGET_DEG)
        print("UNCHANGED: shoulder_pan, wrist_roll, gripper")
        print(f"PLAN: {args.steps} interpolated commands over {args.duration_s:.1f}s")
        if not args.execute:
            print("DRY PLAN ONLY. Re-run with --execute after clearing the workspace and keeping power reachable.")
            return 0

        # This is the first real write. Enabling torque intentionally precedes
        # goal writes; no configuration, PID, mode, or gripper write is made.
        follower.bus.enable_torque()
        arm_names = list(TARGET_DEG)
        initial = {name: current[name] for name in arm_names}
        interval = args.duration_s / args.steps
        for step in range(1, args.steps + 1):
            fraction = step / args.steps
            goal = {
                name: initial[name] + fraction * (TARGET_DEG[name] - initial[name])
                for name in arm_names
            }
            follower.bus.sync_write("Goal_Position", goal)
            if step == 1 or step % 5 == 0 or step == args.steps:
                print(f"step {step:02d}/{args.steps}: " + ", ".join(f"{name}={value:+.2f}" for name, value in goal.items()))
            time.sleep(interval)

        settled = read_degrees(follower.bus)
        errors = {name: settled[name] - target for name, target in TARGET_DEG.items()}
        print("FINAL DEG:", {name: round(settled[name], 3) for name in follower.bus.motors})
        print("TARGET ERROR DEG:", {name: round(error, 3) for name, error in errors.items()})
        if any(abs(error) > 5.0 for error in errors.values()):
            raise RuntimeError("target did not settle within 5 degrees; do not proceed to live shadow")
        print("SAFE-START MOVE COMPLETED. Torque remains enabled at the final hold pose.")
        return 0
    finally:
        # False is intentional: do not change torque state on disconnect.
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: no further goals sent; inspect the robot before any next command.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
