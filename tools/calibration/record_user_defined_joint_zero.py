#!/usr/bin/env python3
"""Record the current physical arm pose as the project's user-defined 0 degrees.

This script never sends a motor goal and never changes torque, calibration, or
any motor register.  It only reads the five arm encoders.  Run it after placing
the arm by hand at the joint-zero pose you have chosen.

The recorded file is deliberately separate from LeRobot calibration.  A later
FK/IK conversion must use:

    urdf_deg = sign * (encoder_deg - encoder_zero_deg)

for each arm joint.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
DEFAULT_OUTPUT = DEPLOY_DIR / "yawfree_user_defined_joint_zero.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually write the JSON record; otherwise only print the proposed record",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing record (requires --execute)",
    )
    args = parser.parse_args()

    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    try:
        normalized = follower.bus.sync_read("Present_Position")
        raw = follower.bus.sync_read("Present_Position", normalize=False)
    finally:
        # Reading must not alter the robot's torque state.
        follower.bus.disconnect(disable_torque=False)

    record = {
        "schema_version": 1,
        "purpose": "user-defined encoder zero for URDF angle conversion",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "port_at_recording": args.port,
        "formula": "urdf_deg = sign * (encoder_deg - encoder_zero_deg)",
        "joints": {
            name: {
                "sign": 1,
                "encoder_zero_deg": round(float(normalized[name]), 6),
                "encoder_zero_raw": int(round(float(raw[name]))),
            }
            for name in ARM_JOINTS
        },
        "notes": [
            "Recorded with no motor command, torque change, or calibration write.",
            "sign=+1 is an initial assumption; validate each joint direction before using IK.",
            "This file does not modify LeRobot calibration or the URDF.",
            "The gripper is intentionally excluded from arm joint-zero calibration.",
        ],
    }

    print("USER-DEFINED JOINT ZERO (current pose will mean URDF 0 deg)")
    for name in ARM_JOINTS:
        item = record["joints"][name]
        print(f"{name}: encoder_zero_deg={item['encoder_zero_deg']:+.3f}, raw={item['encoder_zero_raw']}")
    print(f"OUTPUT: {args.output}")

    if not args.execute:
        print("PREVIEW ONLY. Keep the arm at the intended zero pose, then add --execute to record it.")
        return 0
    if args.output.exists() and not args.overwrite:
        print("ERROR: output already exists; inspect it or add --overwrite intentionally.", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("RECORDED: no motor command, torque change, or calibration write was performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
