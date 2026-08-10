#!/usr/bin/env python3
"""Conservative elbow/wrist holding-gain test; hardware writes need --execute."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

TUNED = ("elbow_flex", "wrist_flex")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--p", type=int, default=24, help="elbow/wrist position P gain; allowed 16..64")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not 16 <= args.p <= 64:
        raise ValueError("--p must stay within the conservative 16..64 range")
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = follower.bus.sync_read("Present_Position", normalize=False)
        old_p = follower.bus.sync_read("P_Coefficient", normalize=False)
        print("CURRENT_RAW:", current)
        print("P_BEFORE:", old_p)
        print(f"PLAN: set all goals to CURRENT_RAW; set only {TUNED} P gain to {args.p}; no torque/current limit change")
        if not args.execute:
            print("DRY RUN ONLY. Add --execute to apply this conservative holding-gain test.")
            return 0
        # First eliminate all old position errors so changing P cannot cause a
        # sudden pull toward an earlier model or recentering goal.
        follower.bus.sync_write("Goal_Position", current, normalize=False)
        for name in TUNED:
            follower.bus.write("P_Coefficient", name, args.p, normalize=False)
        confirmed_p = follower.bus.sync_read("P_Coefficient", normalize=False)
        goal = follower.bus.sync_read("Goal_Position", normalize=False)
        if any(int(goal[name]) != int(current[name]) for name in current):
            raise RuntimeError("failed to hold all motors at their current positions")
        if any(int(confirmed_p[name]) != args.p for name in TUNED):
            raise RuntimeError("P coefficient read-back failed")
        time.sleep(3.0)
        settled = follower.bus.sync_read("Present_Position", normalize=False)
        drift = {name: int(settled[name]) - int(current[name]) for name in current}
        load = follower.bus.sync_read("Present_Load", normalize=False)
        print("P_AFTER:", confirmed_p)
        print("HOLD_DRIFT_RAW_OVER_3S:", drift)
        print("LOAD_AFTER:", load)
        print("HOLD-GAIN TEST COMPLETE: no autonomous or gripper command was sent.")
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
