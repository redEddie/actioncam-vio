#!/usr/bin/env python3
"""Read-only SO-100 motor status inspection; performs no motor writes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig


def read_register(bus, name: str) -> dict[str, float] | dict[str, str]:
    try:
        return {key: float(value) for key, value in bus.sync_read(name, normalize=False).items()}
    except Exception as error:
        return {"unavailable": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    # Deliberately bypass follower.connect(), which would configure registers.
    follower.bus.connect(handshake=True)
    try:
        report = {
            "read_only": True,
            "present_position_normalized": {k: float(v) for k, v in follower.bus.sync_read("Present_Position").items()},
            "present_position_raw": read_register(follower.bus, "Present_Position"),
            "goal_position_raw": read_register(follower.bus, "Goal_Position"),
            "torque_enable": read_register(follower.bus, "Torque_Enable"),
            "p_coefficient": read_register(follower.bus, "P_Coefficient"),
            "i_coefficient": read_register(follower.bus, "I_Coefficient"),
            "d_coefficient": read_register(follower.bus, "D_Coefficient"),
            "max_torque_limit": read_register(follower.bus, "Max_Torque_Limit"),
            "torque_limit": read_register(follower.bus, "Torque_Limit"),
            "minimum_startup_force": read_register(follower.bus, "Minimum_Startup_Force"),
            "cw_dead_zone": read_register(follower.bus, "CW_Dead_Zone"),
            "ccw_dead_zone": read_register(follower.bus, "CCW_Dead_Zone"),
            "protection_current": read_register(follower.bus, "Protection_Current"),
            "acceleration": read_register(follower.bus, "Acceleration"),
            "operating_mode": read_register(follower.bus, "Operating_Mode"),
            "goal_velocity": read_register(follower.bus, "Goal_Velocity"),
            "moving_velocity": read_register(follower.bus, "Moving_Velocity"),
            "maximum_velocity_limit": read_register(follower.bus, "Maximum_Velocity_Limit"),
            "maximum_acceleration": read_register(follower.bus, "Maximum_Acceleration"),
            "acceleration_multiplier": read_register(follower.bus, "Acceleration_Multiplier "),
            "present_load": read_register(follower.bus, "Present_Load"),
            "present_voltage": read_register(follower.bus, "Present_Voltage"),
            "present_current": read_register(follower.bus, "Present_Current"),
            "present_temperature": read_register(follower.bus, "Present_Temperature"),
            "moving": read_register(follower.bus, "Moving"),
            "status": read_register(follower.bus, "Status"),
            "present_speed": read_register(follower.bus, "Present_Speed"),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    finally:
        # Explicitly no torque change on exit.
        follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
