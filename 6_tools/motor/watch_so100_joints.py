#!/usr/bin/env python3
"""Read and print SO-100 joint angles continuously; never writes motor state."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()
    if args.hz <= 0 or args.hz > 20:
        raise ValueError("--hz must be in (0,20]")
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    print("READ-ONLY JOINT MONITOR. Ctrl-C stops monitoring; it does not change torque.")
    try:
        while True:
            values = follower.bus.sync_read("Present_Position")
            print("  ".join(f"{name}={float(values[name]):+7.2f}deg" for name in follower.bus.motors), flush=True)
            time.sleep(1.0 / args.hz)
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
