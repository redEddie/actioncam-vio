#!/usr/bin/env python3
"""Move only the five arm joints to the recorded yaw-free physical start.

The gripper is never commanded.  A motor write requires ``--execute``; without
it this prints the exact start target only.  The target is the physical start
record, with wrist_roll deliberately set to 0 degrees.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
RECORD_PATH = DEPLOY_DIR / "yawfree_physical_start.json"
ARM_JOINTS = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.duration_s <= 0 or args.steps <= 0:
        raise ValueError("duration-s and steps must be positive")
    record = json.loads(RECORD_PATH.read_text())
    target = record.get("arm_joint_degrees", {})
    if set(target) != set(ARM_JOINTS):
        raise RuntimeError(f"invalid start record: {RECORD_PATH}")
    target = {name: float(target[name]) for name in ARM_JOINTS}
    # This is intentional even if an old record is edited by hand.
    target["wrist_roll"] = 0.0

    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        print("CURRENT_DEG:", {name: round(current[name], 3) for name in ARM_JOINTS})
        print("PHYSICAL_START_TARGET_DEG:", {name: round(target[name], 3) for name in ARM_JOINTS})
        print("UNCHANGED: gripper")
        if not args.execute:
            print("DRY PLAN ONLY. Add --execute to send arm goals.")
            return 0
        interval = args.duration_s / args.steps
        for step in range(1, args.steps + 1):
            ratio = step / args.steps
            desired = {name: current[name] + ratio * (target[name] - current[name]) for name in ARM_JOINTS}
            follower.bus.sync_write("Goal_Position", desired)
            time.sleep(interval)
        actual = {name: float(value) for name, value in follower.bus.sync_read("Present_Position").items()}
        error = {name: actual[name] - target[name] for name in ARM_JOINTS}
        print("FINAL_DEG:", {name: round(actual[name], 3) for name in ARM_JOINTS})
        print("TARGET_ERROR_DEG:", {name: round(error[name], 3) for name in ARM_JOINTS})
        print("PHYSICAL_START_COMMAND_COMPLETE")
        return 0
    finally:
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
