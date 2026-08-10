#!/usr/bin/env python3
"""Measure one slow elbow move without policy, camera, or IK; writes need --execute."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--delta-deg", type=float, default=-3.0)
    p.add_argument("--duration-s", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--settle-s", type=float, default=0.0,
                   help="after the ramp, hold the final elbow goal for this many seconds before measuring")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (not (-5.0 <= args.delta_deg <= 5.0) or args.duration_s < 4
            or args.steps < 20 or not 0.0 <= args.settle_s <= 12.0):
        raise ValueError("refusing: delta must be within +/-5deg, duration >=4s, steps >=20, settle 0..12s")
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        initial = follower.bus.sync_read("Present_Position")
        elbow_start = float(initial["elbow_flex"])
        elbow_target = elbow_start + args.delta_deg
        # Keep a 2-degree buffer inside the established 90% IK elbow range
        # (about +/-87.1 degrees).  The current safe hardware pose can be
        # slightly above 80 degrees, so +/-80 was needlessly restrictive.
        if not -85.0 <= elbow_target <= 85.0:
            raise RuntimeError(f"refusing elbow target {elbow_target:.2f}deg outside characterization band")
        print(f"ELBOW START={elbow_start:.3f}deg TARGET={elbow_target:.3f}deg DELTA={args.delta_deg:.3f}deg")
        print("OTHER JOINTS AND GRIPPER: hold current; no policy/camera/IK is used")
        if not args.execute:
            print("DRY RUN ONLY. Add --execute for one supervised slow elbow characterization.")
            return 0

        # Eliminate older goals before the one-joint test.
        initial_raw = follower.bus.sync_read("Present_Position", normalize=False)
        follower.bus.sync_write("Goal_Position", initial_raw, normalize=False)
        follower.bus.enable_torque()
        interval = args.duration_s / args.steps
        rows = []
        elbow_id = follower.bus.motors["elbow_flex"].id
        for step in range(1, args.steps + 1):
            goal = elbow_start + args.delta_deg * (step / args.steps)
            # MotorsBus.write() casts its input to int *before* converting a
            # degree value to encoder units.  Convert first so a 0.1-degree
            # ramp is not silently collapsed into one-degree stairs.
            goal_raw = follower.bus._unnormalize({elbow_id: goal})[elbow_id]
            follower.bus.write("Goal_Position", "elbow_flex", goal_raw, normalize=False)
            time.sleep(interval)
            present = follower.bus.sync_read("Present_Position")
            raw = follower.bus.sync_read("Present_Position", normalize=False)
            goal_register = follower.bus.sync_read("Goal_Position", normalize=False)
            load = follower.bus.sync_read("Present_Load", normalize=False)
            voltage = follower.bus.sync_read("Present_Voltage", normalize=False)
            current = follower.bus.sync_read("Present_Current", normalize=False)
            row = {
                "step": step, "goal_deg": round(goal, 3), "present_deg": round(float(present["elbow_flex"]), 3),
                "error_deg": round(float(present["elbow_flex"]) - goal, 3), "raw": int(raw["elbow_flex"]),
                "goal_raw": int(goal_register["elbow_flex"]),
                "load_raw": int(load["elbow_flex"]), "voltage_raw_0p1V": int(voltage["elbow_flex"]),
                "current_raw": int(current["elbow_flex"]),
            }
            rows.append(row)
            if step == 1 or step % 5 == 0 or step == args.steps:
                print(row)
        final = rows[-1]
        print("FINAL_TRACKING:", final)
        if args.settle_s:
            # Keep the final target unchanged.  This distinguishes an actuator
            # bandwidth lag from inability to hold the commanded load.
            print(f"SETTLING_AT_FINAL_TARGET_FOR_S={args.settle_s:.1f}")
            time.sleep(args.settle_s)
            present = follower.bus.sync_read("Present_Position")
            raw = follower.bus.sync_read("Present_Position", normalize=False)
            goal_register = follower.bus.sync_read("Goal_Position", normalize=False)
            load = follower.bus.sync_read("Present_Load", normalize=False)
            voltage = follower.bus.sync_read("Present_Voltage", normalize=False)
            current = follower.bus.sync_read("Present_Current", normalize=False)
            settled = {
                "goal_deg": round(elbow_target, 3),
                "present_deg": round(float(present["elbow_flex"]), 3),
                "error_deg": round(float(present["elbow_flex"]) - elbow_target, 3),
                "raw": int(raw["elbow_flex"]),
                "goal_raw": int(goal_register["elbow_flex"]),
                "load_raw": int(load["elbow_flex"]),
                "voltage_raw_0p1V": int(voltage["elbow_flex"]),
                "current_raw": int(current["elbow_flex"]),
            }
            print("SETTLED_TRACKING:", settled)
        return 0
    finally:
        # Stop pursuing the tested target before returning control to the user.
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
        print("INTERRUPTED: current goals will be held before disconnect.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
