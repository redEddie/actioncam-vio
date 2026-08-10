#!/usr/bin/env python3
"""Supervised recentering from the inspected safe pose; real writes need --execute."""
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

# Read-only inspection on 2026-08-04 established this pose after the blocked
# microstep. Refuse any materially different pose to avoid stale commands.
EXPECTED_CURRENT = {
    "shoulder_pan": (2.901, 3.0), "shoulder_lift": (-79.077, 3.0),
    "elbow_flex": (85.582, 3.0), "wrist_flex": (38.286, 3.0),
    "wrist_roll": (-1.890, 3.0),
}
TARGET = {"shoulder_lift": -80.0, "elbow_flex": 75.0, "wrist_flex": 35.0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--duration-s", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.duration_s < 4 or args.steps < 20:
        raise ValueError("refusing fast recentering")
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        current = {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()}
        print("CURRENT DEG:", {k: round(current[k], 3) for k in follower.bus.motors})
        stale = [(name, current[name], expected, tolerance) for name, (expected, tolerance) in EXPECTED_CURRENT.items()
                 if abs(current[name] - expected) > tolerance]
        if stale:
            raise RuntimeError("refusing unknown current pose: " + repr(stale))
        print("TARGET DEG:", TARGET)
        print("UNCHANGED: shoulder_pan, wrist_roll, gripper")
        if not args.execute:
            print("DRY PLAN ONLY. Add --execute only after the workspace is clear.")
            return 0
        follower.bus.enable_torque()
        initial = {name: current[name] for name in TARGET}
        for step in range(1, args.steps + 1):
            f = step / args.steps
            follower.bus.sync_write("Goal_Position", {
                name: initial[name] + f * (TARGET[name] - initial[name]) for name in TARGET
            })
            time.sleep(args.duration_s / args.steps)
        time.sleep(0.5)
        settled = {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()}
        errors = {name: settled[name] - target for name, target in TARGET.items()}
        print("FINAL DEG:", {k: round(settled[k], 3) for k in follower.bus.motors})
        print("TARGET ERROR DEG:", {k: round(v, 3) for k, v in errors.items()})
        if any(abs(value) > 2.0 for value in errors.values()):
            raise RuntimeError("recenter did not reach all targets within 2 degrees; do not run policy movement")
        print("RECENTER VERIFIED AND HELD")
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: no further goals sent.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
