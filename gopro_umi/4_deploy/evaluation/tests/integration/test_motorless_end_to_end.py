from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from config.runtime_config import load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ALL_MOTORS, ARM_JOINTS, MotorIO
from control.safety import require_ik_success, require_joint_limits
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.camera import FrameSnapshot
from inference.observation import build_observation, state_from_motor_sample
from inference.remote_smolvla import RemoteRequest, RemoteSmolVLAClient
from trajectory.chunk_scheduler import ChunkScheduler


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.started = False

    def start(self):
        self.started = True

    def request(self, payload, timeout_s):
        assert self.started and payload["id"] == 7
        return {"id": payload["id"], "action": self.response.tolist()}

    def close(self):
        self.started = False


class FakeBus:
    def __init__(self, positions):
        self.positions = dict(positions)
        self.pid = {"P_Coefficient": {}, "I_Coefficient": {}, "D_Coefficient": {}}
        self.goal_writes = []

    def connect(self, handshake=True):
        pass

    def write(self, register, name, value, normalize=False):
        self.pid[register][name] = int(value)

    def sync_read(self, register, motors=None, normalize=True):
        if register in self.pid:
            return self.pid[register].copy()
        if register == "Present_Position":
            return self.positions.copy()
        raise KeyError(register)

    def sync_write(self, register, values, normalize=True):
        self.goal_writes.append((register, dict(values), normalize))

    def disconnect(self, disable_torque):
        pass


def test_full_fake_pipeline_to_single_motor_boundary():
    config = load_runtime_config(DEPLOY / "config" / "deployment.yaml")
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    ik = DLSInverseKinematicsV7(str(DEPLOY / "ik" / "urdf" / "so_arm_with_gopro_final.urdf"))

    raw_initial = config.start_raw_deg.copy()
    positions = dict(zip(ARM_JOINTS, raw_initial, strict=True))
    positions["gripper"] = 1050.0
    actual_obs = state_from_motor_sample(positions, 10.0, mapping, ik)
    frame = FrameSnapshot(np.zeros((256, 256, 3), dtype=np.uint8), 9.999, 11, (1080, 1920, 3))
    observation = build_observation(7, frame, actual_obs)

    raw_response = np.zeros((1, 15, 6), dtype=np.float64)
    raw_response[0, :, 0] = 0.0001
    raw_response[0, :, 5] = 0.01
    client = RemoteSmolVLAClient(FakeTransport(raw_response), request_timeout_s=1.0)
    client.start()
    client.submit(RemoteRequest.from_observation(observation))
    response = None
    deadline = time.monotonic() + 2.0
    while response is None and time.monotonic() < deadline:
        response = client.poll()
        time.sleep(0.001)
    client.close()
    assert response is not None and response.error is None and response.request_id == 7
    actions = validate_postprocessed_action_chunk(response.action)
    assert actions.shape == (15, 6) and np.array_equal(actions, raw_response[0])

    scheduler = ChunkScheduler(config.deployment["scheduler"]["request_stride_s"], config.deployment["scheduler"]["transition_overlap_s"])
    context = scheduler.register_request(observation)
    arrival_positions = positions.copy()
    arrival_positions["shoulder_pan"] += 0.02
    actual_arrival = state_from_motor_sample(arrival_positions, 10.15, mapping, ik)
    integrated = scheduler.handle_response(7, actions, 10.15, actual_arrival.state)
    assert integrated["status"] == "INTEGRATED"
    assert np.array_equal(context.s_obs, observation.s_obs)
    assert np.array_equal(integrated["reanchor"]["actual_arrival_state"], actual_arrival.state)

    sample = scheduler.sample(10.15)
    assert sample["status"] == "VALID"
    target = sample["action"]
    rotation = Rotation.from_euler("ZYX", [actual_arrival.yaw_radians, target[4], target[3]]).as_matrix()
    seed = np.deg2rad(actual_arrival.q_urdf_degrees)
    q_nom = ik.solve(target[:3], rotation, seed)
    require_ik_success(q_nom, ik.last_converged)
    require_joint_limits(q_nom, ik.joint_names, ik.PHYSICAL_JOINT_LIMITS_RAD)
    require_joint_limits(q_nom, ik.joint_names, ik.safe_limits)

    gravity = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    q_actual_deg = actual_arrival.q_urdf_degrees
    hold = controller.compute_hold(q_actual_deg, q_actual_deg)
    q_nom_deg = np.rad2deg(q_nom)
    controller.begin_trajectory(q_actual_deg, q_nom_deg, hold)
    q_command = controller.compute_trajectory(q_nom_deg, q_actual_deg)
    assert np.array_equal(q_command, hold)

    raw_degrees = mapping.urdf_degrees_to_raw_degrees(q_command)
    goals = mapping.arm_degrees_to_raw_ticks(raw_degrees)
    goals["gripper"] = mapping.gripper_raw_from_width(float(target[5]))
    fake_bus = FakeBus(positions)
    motor = MotorIO(
        config.deployment["motor"]["port"],
        config.motor_calibration,
        config.deployment["motor"]["pid"],
        bus=fake_bus,
    )
    motor.connect(permit_motion=True)
    motor.write_goal_ticks(goals)
    assert len(fake_bus.goal_writes) == 1
    assert fake_bus.goal_writes[0][0] == "Goal_Position" and fake_bus.goal_writes[0][2] is False
