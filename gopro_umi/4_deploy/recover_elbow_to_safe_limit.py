#!/usr/bin/env python3
"""Supervised elbow-only recovery from the 95% soft-limit neighbourhood.

No policy, camera, IK target, gripper, or non-elbow joint command is used.
It moves elbow_flex only toward 85 degrees and finally holds the measured
encoder pose whether it succeeds or fails.
"""
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

TARGET_DEG = 85.0  # inside the original 90% elbow soft limit (~87.15 deg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--duration-s", type=float, default=16.0)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.duration_s < 8 or args.steps < 40:
        raise ValueError("refusing fast recovery: duration >=8s and steps >=40")

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        start = float(follower.bus.sync_read("Present_Position")["elbow_flex"])
        if not 87.0 <= start <= 93.0:
            raise RuntimeError(f"refusing recovery outside 87..93deg elbow band: {start:.3f}")
        print(f"ELBOW RECOVERY: START={start:.3f}deg TARGET={TARGET_DEG:.3f}deg")
        print("ONLY elbow_flex changes; all other joints and gripper hold current.")
        if not args.execute:
            print("DRY PLAN ONLY. Add --execute for supervised recovery.")
            return 0

        present_raw = follower.bus.sync_read("Present_Position", normalize=False)
        follower.bus.sync_write("Goal_Position", present_raw, normalize=False)
        follower.bus.enable_torque()
        elbow_id = follower.bus.motors["elbow_flex"].id
        for step in range(1, args.steps + 1):
            desired = start + (TARGET_DEG - start) * step / args.steps
            raw_goal = follower.bus._unnormalize({elbow_id: desired})[elbow_id]
            follower.bus.write("Goal_Position", "elbow_flex", raw_goal, normalize=False)
            time.sleep(args.duration_s / args.steps)
            actual = float(follower.bus.sync_read("Present_Position")["elbow_flex"])
            if step == 1 or step % 10 == 0 or step == args.steps:
                print({"step": step, "goal_deg": round(desired, 3), "actual_deg": round(actual, 3),
                       "tracking_error_deg": round(actual - desired, 3)})
        final = float(follower.bus.sync_read("Present_Position")["elbow_flex"])
        print(f"FINAL_ELBOW_DEG={final:.3f}")
        if final > 87.0:
            raise RuntimeError("recovery did not return elbow inside original 90% soft limit")
        print("ELBOW_RECOVERY_VERIFIED")
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
        print("INTERRUPTED: current encoder pose will be held.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
