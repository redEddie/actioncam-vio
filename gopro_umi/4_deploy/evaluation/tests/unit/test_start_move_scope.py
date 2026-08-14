from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from config.runtime_config import load_runtime_config
from control.joint_mapping import JointMapping


TOOL_PATH = ROOT / "6_tools" / "motor" / "move_to_physical_start.py"
SPEC = importlib.util.spec_from_file_location("move_to_physical_start_tool", TOOL_PATH)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


class FakeBus:
    def __init__(self, arm_degrees):
        self.arm_degrees = dict(zip(tool.ARM_JOINTS, arm_degrees, strict=True))
        self.gripper_raw = 1777
        self.writes = []

    def sync_read(self, register, normalize=True):
        assert register == "Present_Position"
        if normalize:
            return {**self.arm_degrees, "gripper": 50.0}
        return {**{name: 2000.0 for name in tool.ARM_JOINTS}, "gripper": self.gripper_raw}

    def sync_write(self, register, values, normalize=True):
        assert register == "Goal_Position" and normalize is True
        assert set(values) == set(tool.ARM_JOINTS)
        self.writes.append((register, dict(values), normalize))
        self.arm_degrees.update(values)


def test_canonical_start_move_writes_only_five_arm_goals_and_preserves_gripper():
    config = load_runtime_config(DEPLOY / "config" / "deployment.yaml")
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    preflight = tool.target_preflight(config, mapping)
    assert preflight["result"] == "PASS"
    assert all(item["inside"] for item in preflight["motor_calibration_ranges"].values())

    fake_time = [0.0]

    def clock():
        return fake_time[0]

    def sleep(duration):
        fake_time[0] += duration

    bus = FakeBus(config.start_raw_deg + np.array([5.0, -6.0, 4.0, -3.0, 2.0]))
    report = tool.execute_controlled_move(
        bus,
        mapping,
        config.start_raw_deg,
        duration_s=6.0,
        steps=30,
        settle_s=0.5,
        clock=clock,
        sleeper=sleep,
    )
    assert len(bus.writes) == 30
    assert all(set(values) == set(tool.ARM_JOINTS) for _, values, _ in bus.writes)
    assert bus.gripper_raw == 1777
    assert report["move"]["goal_position_arm_physical_writes"] == 30
    assert report["move"]["gripper_goal_writes"] == 0
    assert report["move"]["pid_writes"] == 0
    assert report["move"]["torque_writes"] == 0
    assert report["move"]["other_register_writes"] == 0
    assert np.array_equal(report["post_move"]["arm_raw_deg"], config.start_raw_deg)

