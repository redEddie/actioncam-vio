from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from config.runtime_config import load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ALL_MOTORS, ARM_JOINTS, MotorIO
from control.safety import require_joint_limits
from ik.ik_solver_v7 import DLSInverseKinematicsV7


class FakeBus:
    def __init__(self):
        self.pid = {register: {} for register in ("P_Coefficient", "I_Coefficient", "D_Coefficient")}
        self.goal_writes = []

    def connect(self, handshake=True):
        return None

    def write(self, register, name, value, normalize=False):
        self.pid[register][name] = int(value)

    def sync_read(self, register, motors=None, normalize=True):
        if register in self.pid:
            return self.pid[register].copy()
        raise AssertionError("real/present-position read is forbidden in this motorless test")

    def sync_write(self, register, values, normalize=True):
        self.goal_writes.append((register, dict(values), normalize))

    def disconnect(self, disable_torque):
        return None


def test_pose_dependent_gravity_bp_delta_to_fake_motor_boundary():
    config = load_runtime_config(DEPLOY / "config" / "deployment.yaml")
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    gravity = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    ik = DLSInverseKinematicsV7(str(DEPLOY / "ik" / "urdf" / "so_arm_with_gopro_final.urdf"))
    bus = FakeBus()
    motor = MotorIO(
        config.deployment["motor"]["port"],
        config.motor_calibration,
        config.deployment["motor"]["pid"],
        bus=bus,
    )
    motor.connect(permit_motion=True)

    q_ref = config.start_urdf_deg.copy()
    # Exercise the existing RAW↔URDF boundary before gravity sees actual URDF degrees.
    q_actual = mapping.raw_degrees_to_urdf_degrees(
        mapping.urdf_degrees_to_raw_degrees(q_ref)
    )
    hold = controller.compute_hold(q_ref, q_actual)
    assert np.array_equal(
        hold,
        q_ref + config.static_support_bias_deg + config.gravity_reference_bias_deg,
    )

    def write_fake(q_command, width=None):
        if width is None:
            width = mapping.dataset_open * 0.5
        require_joint_limits(
            np.deg2rad(q_command), ik.joint_names, ik.PHYSICAL_JOINT_LIMITS_RAD
        )
        goals = mapping.arm_degrees_to_raw_ticks(
            mapping.urdf_degrees_to_raw_degrees(q_command)
        )
        goals["gripper"] = mapping.gripper_raw_from_width(width)
        assert set(goals) == set(ALL_MOTORS)
        return motor.write_goal_ticks(goals)

    write_fake(hold)
    origin = controller.begin_trajectory(q_actual, q_ref, hold)
    tick0 = controller.compute_trajectory(q_ref, q_actual)
    assert np.array_equal(tick0, hold)
    write_fake(tick0)

    q_actual_1 = q_ref + np.array([0.0, 3.0, -2.0, 1.0, 0.0])
    q_nom_1 = q_ref + np.array([0.0, 1.0, -0.5, 0.25, 0.0])
    trajectory_1 = controller.compute_trajectory(q_nom_1, q_actual_1)
    assert np.any(np.abs(controller.last_delta_support_deg[[1, 2, 3]]) > 0.0)
    write_fake(trajectory_1)

    # A simulated new chunk/overlap changes the nominal target, never the origin.
    q_nom_2 = q_ref + np.array([0.0, 1.5, -0.75, 0.4, 0.0])
    trajectory_2 = controller.compute_trajectory(q_nom_2, q_actual_1)
    assert controller.origin is origin
    write_fake(trajectory_2)

    # A true return to HOLD and subsequent onset explicitly creates a new origin.
    controller.reset()
    new_hold = controller.compute_hold(q_nom_2, q_actual_1)
    new_origin = controller.begin_trajectory(q_actual_1, q_nom_2, new_hold)
    assert new_origin is not origin
    assert np.array_equal(controller.compute_trajectory(q_nom_2, q_actual_1), new_hold)

    assert len(bus.goal_writes) == 4
    assert all(register == "Goal_Position" and normalize is False for register, _, normalize in bus.goal_writes)
    assert all(set(values) == set(ALL_MOTORS) for _, values, _ in bus.goal_writes)
    assert all(values["gripper"] == 1800 for _, values, _ in bus.goal_writes)
    assert tuple(ARM_JOINTS) == config.joint_order
