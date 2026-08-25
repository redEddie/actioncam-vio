#!/usr/bin/env python3
"""Print only the five current arm-joint angles in degrees.

This is read-only: it sends no goal position, torque, calibration, gripper,
camera, policy, or file-write command.  ``--id`` is accepted as a human label
so the same invocation can be used with the SO101 follower setup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="so101_follower_arm", help="read-only arm label")
    args = parser.parse_args()

    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        degrees = follower.bus.sync_read("Present_Position")
        print(f"READ_ONLY: id={args.id}, port={args.port}, unit=deg")
        for name in ARM_JOINTS:
            print(f"{name}: {float(degrees[name]):+.3f}")
    finally:
        # Do not disable or enable torque on disconnect: preserve motor state.
        follower.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
