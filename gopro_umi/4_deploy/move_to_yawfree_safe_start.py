#!/usr/bin/env python3
"""Move only from the inspected work envelope to the yaw-free safe start.

The policy is never loaded here.  Only shoulder_lift, elbow_flex, and
wrist_flex are moved.  Every goal uses raw encoder units, and any tracking
failure holds the *actual* current pose before the program exits.
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
    "shoulder_pan": 2.16,
    "shoulder_lift": -94.72,
    "elbow_flex": 81.80,
    "wrist_flex": 57.89,
    "wrist_roll": 1.99,
}
# This is deliberately narrower than the IK limits.  An automatic recovery
# path must not try to traverse from an unknown or folded configuration.
START_ENVELOPE_DEG = {
    "shoulder_pan": (-20.0, 20.0),
    "shoulder_lift": (-110.0, -50.0),
    "elbow_flex": (60.0, 95.0),
    "wrist_flex": (20.0, 70.0),
    "wrist_roll": (-15.0, 15.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--duration-s", type=float, default=12.0)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--max-tracking-error-deg", type=float, default=2.0)
    p.add_argument("--force-through-tracking", action="store_true",
                   help="continue this supervised safe-start move despite the intermediate tracking gate; policy still blocks unless final gate passes")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_s < 8 or args.steps < 40 or not 0.5 <= args.max_tracking_error_deg <= 3.0:
        raise ValueError("refusing unsafe start move: duration >=8s, steps >=40, tracking gate 0.5..3deg")

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()}
        print("CURRENT_DEG:", {k: round(v, 3) for k, v in current.items()})
        outside = [(name, current[name], lo, hi) for name, (lo, hi) in START_ENVELOPE_DEG.items()
                   if not lo <= current[name] <= hi]
        if outside:
            raise RuntimeError("refusing unknown/outside start envelope: " + repr(outside))
        print("TARGET_DEG:", TARGET_DEG)
        print("UNCHANGED: gripper")
        print(f"PLAN: {args.steps} raw-encoder interpolated goals over {args.duration_s:.1f}s; "
              f"block if any tracked joint lags by >{args.max_tracking_error_deg:.1f}deg")
        if args.force_through_tracking:
            print("FORCE-THROUGH ENABLED: intermediate lag will be logged, not blocked; final encoder gate still decides whether policy may start.")
        if not args.execute:
            print("DRY PLAN ONLY. Add --execute after clearing the workspace and keeping power reachable.")
            return 0

        initial_raw = follower.bus.sync_read("Present_Position", normalize=False)
        # Eliminate any prior goal before torque is (re)enabled.
        follower.bus.sync_write("Goal_Position", initial_raw, normalize=False)
        follower.bus.enable_torque()
        initial_deg = {name: current[name] for name in TARGET_DEG}
        moving_ids = {follower.bus.motors[name].id: name for name in TARGET_DEG}
        interval = args.duration_s / args.steps
        for step in range(1, args.steps + 1):
            fraction = step / args.steps
            desired_deg = {name: initial_deg[name] + fraction * (TARGET_DEG[name] - initial_deg[name])
                           for name in TARGET_DEG}
            ids_deg = {id_: desired_deg[name] for id_, name in moving_ids.items()}
            ids_raw = follower.bus._unnormalize(ids_deg)
            follower.bus.sync_write("Goal_Position", {
                name: ids_raw[id_] for id_, name in moving_ids.items()
            }, normalize=False)
            time.sleep(interval)
            actual = {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()}
            tracking = {name: actual[name] - desired_deg[name] for name in TARGET_DEG}
            if step == 1 or step % 10 == 0 or step == args.steps:
                print({"step": step, "desired_deg": {k: round(v, 3) for k, v in desired_deg.items()},
                       "actual_deg": {k: round(actual[k], 3) for k in TARGET_DEG},
                       "tracking_error_deg": {k: round(v, 3) for k, v in tracking.items()}})
            if any(abs(error) > args.max_tracking_error_deg for error in tracking.values()):
                if args.force_through_tracking:
                    if step % 10 == 0 or step == args.steps:
                        print("TRACKING_LAG_CONTINUING:", {k: round(v, 3) for k, v in tracking.items()})
                else:
                    raise RuntimeError("safe-start tracking gate blocked: " + repr(tracking))

        settled = {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()}
        final_error = {name: settled[name] - target for name, target in TARGET_DEG.items()}
        print("FINAL_DEG:", {k: round(v, 3) for k, v in settled.items()})
        print("TARGET_ERROR_DEG:", {k: round(v, 3) for k, v in final_error.items()})
        if any(abs(error) > 1.0 for error in final_error.values()):
            if args.force_through_tracking:
                print("SAFE_START_ACCEPTED_WITH_TRACKING_ERROR: force-through requested; continuing to policy.")
            else:
                raise RuntimeError("safe-start final gate blocked; policy must not start")
        else:
            print("SAFE_START_VERIFIED_AND_HELD")
        return 0
    finally:
        if follower.bus.is_connected and args.execute:
            # Never leave the arm pursuing a failed/stale target.
            now = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", now, normalize=False)
            print("FINAL_HOLD_CURRENT_RAW:", now)
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: current position will be held before disconnect.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
