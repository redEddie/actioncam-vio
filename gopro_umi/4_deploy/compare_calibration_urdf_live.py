#!/usr/bin/env python3
"""Compare active LeRobot calibration ranges, URDF limits, and live encoders.

Read-only: no calibration write, torque change, goal write, IK solve, camera,
or policy.  Arm calibration ranges are converted using LeRobot's exact
``MotorNormMode.DEGREES`` formula.  This exposes disagreements such as a
calibrated range wider than the URDF joint limit.
"""
from __future__ import annotations

import argparse
import json
import sys

import deploy_smolvla_yawfree as deploy


CALIBRATION_PATH = deploy.PROJECT_ROOT / "smolvla_cache/lerobot/calibration/robots/so_follower/None.json"
ENCODER_RESOLUTION_MINUS_ONE = 4095.0
ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def calibrated_degrees(raw: float, minimum: float, maximum: float) -> float:
    return (raw - (minimum + maximum) / 2.0) * 360.0 / ENCODER_RESOLUTION_MINUS_ONE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--no-live", action="store_true", help="compare JSON and URDF only")
    args = parser.parse_args()
    calibration = json.loads(CALIBRATION_PATH.read_text())
    solver = deploy.load_candidate_solver()(str(deploy.URDF_PATH), limit_margin_ratio=1.0)
    live_raw: dict[str, float] | None = None
    if not args.no_live:
        lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
        if str(lerobot_src) not in sys.path:
            sys.path.insert(0, str(lerobot_src))
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

        follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
        follower.bus.connect(handshake=True)
        try:
            live_raw = {name: float(value) for name, value in follower.bus.sync_read("Present_Position", normalize=False).items()}
        finally:
            follower.bus.disconnect(disable_torque=False)

    print("CALIBRATION_JSON:", CALIBRATION_PATH)
    print("Arm degrees use LeRobot: (raw - (range_min + range_max)/2) * 360 / 4095")
    print(f"{'joint':16s} {'calib raw':>16s} {'calibrated deg':>24s} {'URDF physical deg':>24s} {'live raw':>10s} {'live deg':>11s}")
    for name in ARM_NAMES:
        item = calibration[name]
        raw_lo, raw_hi = float(item["range_min"]), float(item["range_max"])
        cal_lo, cal_hi = calibrated_degrees(raw_lo, raw_lo, raw_hi), calibrated_degrees(raw_hi, raw_lo, raw_hi)
        urdf_lo, urdf_hi = (float(value) * 180.0 / 3.141592653589793 for value in solver.PHYSICAL_JOINT_LIMITS_RAD[name])
        if live_raw is None:
            live_text = ("-", "-")
        else:
            raw = live_raw[name]
            live_text = (f"{raw:.0f}", f"{calibrated_degrees(raw, raw_lo, raw_hi):+.2f}")
        print(f"{name:16s} {raw_lo:6.0f}..{raw_hi:<6.0f} {cal_lo:+10.2f}..{cal_hi:+10.2f} {urdf_lo:+10.2f}..{urdf_hi:+10.2f} {live_text[0]:>10s} {live_text[1]:>11s}")

    grip = calibration["gripper"]
    print("\nGRIPPER (not an URDF arm-joint limit):")
    print(f"  software calibration raw range = {grip['range_min']}..{grip['range_max']}; homing_offset={grip['homing_offset']}")
    print("  Note: its JSON is 600..3000 as requested; actual motor-register restoration remains pending until serial connection succeeds.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
