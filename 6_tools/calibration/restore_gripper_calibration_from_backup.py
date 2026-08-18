#!/usr/bin/env python3
"""Restore only the gripper calibration overwritten by an incomplete sweep.

The arm-joint calibration produced by the most recent LeRobot sweep is kept.
Only ``gripper`` is restored from the pre-calibration backup into both the
project calibration JSON and gripper motor registers, with the raw range taken
from the canonical motor mapping. Default mode prints the planned values;
``--execute`` is required for the JSON and motor writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE = PROJECT_ROOT / "4_deploy/config/so_follower_calibration.json"
MOTOR_MAPPING = PROJECT_ROOT / "4_deploy/config/motor_mapping.json"
BACKUP = PROJECT_ROOT / "etc/smolvla_cache/lerobot/calibration/robots/so_follower/None.before_encoder_recalibration_20260805.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    active = json.loads(ACTIVE.read_text())
    backup = json.loads(BACKUP.read_text())
    canonical_gripper = json.loads(MOTOR_MAPPING.read_text())["gripper"]
    endpoint_values = [int(canonical_gripper["open_raw"]), int(canonical_gripper["closed_raw"])]
    if "gripper" not in active or "gripper" not in backup:
        raise RuntimeError("active or backup calibration has no gripper entry")
    old = active["gripper"]
    restore = dict(backup["gripper"])
    # Preserve the known-good pre-calibration zero, but intentionally narrow
    # the usable gripper range as requested.
    restore["range_min"] = min(endpoint_values)
    restore["range_max"] = max(endpoint_values)
    print("CURRENT_BAD_GRIPPER_CALIBRATION:", old)
    print("RESTORE_GRIPPER_CALIBRATION:", restore)
    print("ARM_JOINTS_UNCHANGED: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll")
    if not args.execute:
        print("DRY RUN ONLY. Add --execute to restore only the gripper JSON and motor registers.")
        return 0

    lerobot_src = PROJECT_ROOT / "etc" / "third_party" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(
        SO100FollowerConfig(
            port=args.port, use_degrees=True, id=ACTIVE.stem, calibration_dir=ACTIVE.parent
        )
    )
    follower.bus.connect(handshake=True)
    try:
        # The bus has already parsed the active JSON into MotorCalibration
        # objects. Replace only that one object field-by-field and write only
        # that motor's calibration registers.
        calibration = follower.bus.calibration["gripper"]
        calibration.id = int(restore["id"])
        calibration.drive_mode = int(restore["drive_mode"])
        calibration.homing_offset = int(restore["homing_offset"])
        calibration.range_min = int(restore["range_min"])
        calibration.range_max = int(restore["range_max"])
        torque_before = float(follower.bus.read("Torque_Enable", "gripper", normalize=False))
        follower.bus.disable_torque("gripper")
        follower.bus.write_calibration({"gripper": calibration}, cache=False)
        if torque_before != 0:
            follower.bus.enable_torque("gripper")
        active["gripper"] = restore
        ACTIVE.write_text(json.dumps(active, indent=4) + "\n")
        print("GRIPPER_CALIBRATION_RESTORED; arm calibration retained.")
        return 0
    finally:
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
