#!/usr/bin/env python3
"""Set every SO-100 motor's stored goal to its current raw encoder position.

Use this to stop a servo from continuing to pursue a previous goal while
keeping torque on, so the arm does not suddenly sag. This performs no policy,
camera, or configuration operation. Real writes require --execute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-torque", action="store_true",
                        help="after writing current goals, enable torque to hold a manually placed pose")
    args = parser.parse_args()
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = follower.bus.sync_read("Present_Position", normalize=False)
        previous_goal = follower.bus.sync_read("Goal_Position", normalize=False)
        print("PRESENT_RAW:", current)
        print("PREVIOUS_GOAL_RAW:", previous_goal)
        if not args.execute:
            print("DRY RUN ONLY. Add --execute to replace all goals with current positions.")
            return 0
        follower.bus.sync_write("Goal_Position", current, normalize=False)
        # If torque had been disabled for a manual mechanical check, goals are
        # already set to the measured current pose before torque is restored.
        if args.enable_torque:
            follower.bus.enable_torque()
        confirmed = follower.bus.sync_read("Goal_Position", normalize=False)
        print("CONFIRMED_HOLD_GOAL_RAW:", confirmed)
        if any(abs(int(confirmed[name]) - int(current[name])) > 1 for name in current):
            raise RuntimeError("goal read-back differs from current position")
        print("HOLD CURRENT CONFIRMED" + (": torque enabled at current pose." if args.enable_torque else ": torque state was not changed."))
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
