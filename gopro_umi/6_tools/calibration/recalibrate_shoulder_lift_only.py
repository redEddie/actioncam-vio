#!/usr/bin/env python3
"""Interactively recalibrate only shoulder_lift; preserve every other joint.

Unlike ``lerobot-calibrate`` this program never records ranges for pan, elbow,
wrist, wrist-roll, or gripper.  It updates only shoulder_lift's homing offset
and measured raw min/max in the active project calibration JSON and on that
single motor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JOINT = "shoulder_lift"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE = PROJECT_ROOT / "4_deploy/config/so_follower_calibration.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    active = json.loads(ACTIVE.read_text())
    print("CURRENT_SHOULDER_LIFT_CALIBRATION:", active[JOINT])
    print("PRESERVED_JOINTS:", [name for name in active if name != JOINT])
    if not args.execute:
        print("DRY RUN ONLY. Add --execute for the interactive shoulder_lift-only sweep.")
        return 0

    lerobot_src = PROJECT_ROOT / "third_party" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
    from lerobot.motors.feetech.feetech import OperatingMode

    follower = SO100Follower(
        SO100FollowerConfig(
            port=args.port, use_degrees=True, id=ACTIVE.stem, calibration_dir=ACTIVE.parent
        )
    )
    follower.bus.connect(handshake=True)
    try:
        torque_before = float(follower.bus.read("Torque_Enable", JOINT, normalize=False))
        follower.bus.disable_torque(JOINT)
        follower.bus.write("Operating_Mode", JOINT, OperatingMode.POSITION.value)
        input("Move shoulder_lift to the middle of its real mechanical range, then press ENTER...")
        homing = follower.bus.set_half_turn_homings([JOINT])[JOINT]
        print("Move ONLY shoulder_lift slowly through its full real mechanical range. Recording positions. Press ENTER to stop...")
        mins, maxes = follower.bus.record_ranges_of_motion([JOINT])
        calibration = follower.bus.calibration[JOINT]
        calibration.id = int(active[JOINT]["id"])
        calibration.drive_mode = int(active[JOINT]["drive_mode"])
        calibration.homing_offset = int(homing)
        calibration.range_min = int(mins[JOINT])
        calibration.range_max = int(maxes[JOINT])
        follower.bus.write_calibration({JOINT: calibration}, cache=False)
        if torque_before != 0:
            follower.bus.enable_torque(JOINT)
        active[JOINT] = {
            "id": calibration.id,
            "drive_mode": calibration.drive_mode,
            "homing_offset": calibration.homing_offset,
            "range_min": calibration.range_min,
            "range_max": calibration.range_max,
        }
        ACTIVE.write_text(json.dumps(active, indent=4) + "\n")
        print("NEW_SHOULDER_LIFT_CALIBRATION:", active[JOINT])
        print("SHOULDER_LIFT_ONLY_CALIBRATION_SAVED")
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: shoulder_lift calibration was not completed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
