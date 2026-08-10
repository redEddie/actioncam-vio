#!/usr/bin/env python3
"""Explicit emergency torque-off for the connected SO-100 arm.

This is intentionally a one-purpose command. It does not configure motors,
load a policy, open a camera, or write a goal position. With ``--execute`` it
only sends Torque_Enable=0 to every motor and then reads the register back.
The arm can drop under gravity once torque is off: keep hands and objects clear.
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
    parser.add_argument("--execute", action="store_true", help="required to write Torque_Enable=0")
    args = parser.parse_args()
    if not args.execute:
        print("DRY RUN ONLY. Add --execute to disable torque on all six motors.")
        return 0
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        follower.bus.disable_torque()
        state = follower.bus.sync_read("Torque_Enable", normalize=False)
        print("TORQUE_ENABLE_AFTER_COMMAND:", state)
        if any(int(value) != 0 for value in state.values()):
            raise RuntimeError("one or more motors still report torque enabled")
        print("TORQUE OFF CONFIRMED")
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
