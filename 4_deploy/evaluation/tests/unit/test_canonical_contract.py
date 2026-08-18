from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from config.runtime_config import ConfigError, load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ALL_MOTORS, ARM_JOINTS, MotorIO
from control.safety import (
    SafetyError,
    TargetStatus,
    require_ik_success,
    require_joint_limits,
    require_raw_goal_ticks,
    target_status,
)
from inference.action_postprocess import validate_postprocessed_action_chunk
from inference.camera import CameraError, FrameSnapshot, LatestFrameCamera, prepare_transport_rgb
from inference.observation import ActualStateSnapshot, ObservationSnapshot
import inference.remote_smolvla as remote_module
from inference.remote_smolvla import SubprocessSSHTransport
from trajectory.chunk_scheduler import ChunkScheduler
from trajectory.incremental import reconstruct_incremental_states
from trajectory.interpolate import generate_30hz_schedule, sample_trajectory_at_time
from trajectory.overlap import overlap_ensemble
from trajectory.reanchor import reanchor_trajectory


CONFIG_PATH = DEPLOY / "config" / "deployment.yaml"


def config_and_mapping():
    config = load_runtime_config(CONFIG_PATH)
    return config, JointMapping(config.motor_mapping_path, config.motor_calibration)


def observation(request_id: int, t_obs: float, state=None) -> ObservationSnapshot:
    if state is None:
        state = np.array([0.2, 0.0, 0.2, 0.0, 0.0, 0.5])
    return ObservationSnapshot(
        request_id=request_id,
        t_obs=t_obs,
        t_frame=t_obs - 0.001,
        t_q_obs=t_obs - 0.0005,
        frame_sequence=request_id + 1,
        frame_metadata={"fake": True},
        image_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        raw_degrees=np.zeros(5),
        q_obs_degrees=np.zeros(5),
        s_obs=state,
        yaw_radians=0.2,
    )


def test_runtime_config_and_mapping_roundtrip():
    config, mapping = config_and_mapping()
    assert config.k_ext == 0.5
    assert np.allclose(mapping.urdf_degrees_to_raw_degrees(np.zeros(5)), config.encoder_zero_deg)
    assert np.allclose(mapping.urdf_degrees_to_raw_degrees(config.start_urdf_deg), config.start_raw_deg)
    assert np.allclose(mapping.raw_degrees_to_urdf_degrees(config.start_raw_deg), config.start_urdf_deg)
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = rng.uniform(-80.0, 80.0, size=5)
        assert np.allclose(mapping.raw_degrees_to_urdf_degrees(mapping.urdf_degrees_to_raw_degrees(q)), q)


def test_gripper_endpoint_contract_clamp_monotonicity_and_roundtrip():
    config, mapping = config_and_mapping()
    gripper = config.motor_mapping["gripper"]
    assert gripper["open_raw"] == 600
    assert gripper["closed_raw"] == 3000
    ds_open = float(gripper.get("dataset_open", 0.441305))
    ds_closed = float(gripper.get("dataset_closed", 0.0))
    assert mapping.gripper_raw_from_width(ds_open) == 600
    assert mapping.gripper_raw_from_width(ds_closed) == 3000
    assert mapping.gripper_raw_from_width(ds_open * 0.5) == 1800
    assert mapping.gripper_raw_from_width(-1.0) == 3000
    assert mapping.gripper_raw_from_width(2.0) == 600
    samples = np.linspace(ds_closed, ds_open, 101)
    raw = np.array([mapping.gripper_raw_from_width(value) for value in samples])
    assert np.all(np.diff(raw) < 0)
    for value in samples:
        reconstructed = mapping.gripper_width_from_raw(mapping.gripper_raw_from_width(value))
        assert reconstructed == pytest.approx(value, abs=1.0 / abs(600 - 3000))
    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            mapping.gripper_raw_from_width(invalid)
        with pytest.raises(ValueError):
            mapping.gripper_width_from_raw(invalid)


def test_action_shape_finite_and_full_chunk_preservation():
    raw = np.arange(90, dtype=np.float32).reshape(1, 15, 6)
    local = validate_postprocessed_action_chunk(raw)
    assert local.shape == (15, 6)
    assert np.array_equal(local, raw[0])
    assert np.array_equal(local[0], raw[0, 0]) and np.array_equal(local[14], raw[0, 14])
    assert validate_postprocessed_action_chunk(raw[0]).shape == (15, 6)
    for bad in (np.zeros((15, 5)), np.zeros((14, 6))):
        with pytest.raises(ValueError):
            validate_postprocessed_action_chunk(bad)
    for bad in (np.full((15, 6), np.nan), np.full((15, 6), np.inf)):
        with pytest.raises(ValueError):
            validate_postprocessed_action_chunk(bad)


def test_incremental_all_actions_and_per_step_gripper_clip():
    actions = np.zeros((15, 6))
    actions[:, 0] = np.arange(1, 16) / 1000.0
    actions[:, 3] = 0.01
    actions[:, 4] = -0.005
    actions[:, 5] = np.array([0.7, -0.9] * 7 + [0.7])
    states = reconstruct_incremental_states([0, 0, 0, 0, 0, 0.5], actions)
    assert states.shape == (16, 6)
    assert np.isclose(states[-1, 0], actions[:, 0].sum())
    assert np.isclose(states[-1, 3], 0.15) and np.isclose(states[-1, 4], -0.075)
    assert np.all((states[:, 5] >= 0.0) & (states[:, 5] <= 1.0))
    assert states[1, 5] == 1.0 and states[2, 5] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        reconstruct_incremental_states(np.zeros(6), np.zeros((14, 6)))


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.15, 0.45])
def test_reanchor_explicit_arrival_and_fractional_interpolation(tau):
    actions = np.zeros((15, 6))
    actions[:, 0] = 0.01
    actual = np.array([0.123, 0, 0, 0, 0, 0.5])
    result = reanchor_trajectory(actions, np.array([0, 0, 0, 0, 0, 0.5]), actual, 10.0, 10.0 + tau)
    assert result["status"] == "VALID"
    assert np.array_equal(result["actual_arrival_state"], actual)
    assert np.array_equal(result["states"][0], actual)
    assert 0.0 <= result["alpha"] <= 1.0


def test_reanchor_expiry_and_negative_latency():
    actions = np.zeros((15, 6))
    assert reanchor_trajectory(actions, np.zeros(6), np.zeros(6), 0.0, 1.5)["status"] == "EXPIRED"
    with pytest.raises(ValueError):
        reanchor_trajectory(actions, np.zeros(6), np.zeros(6), 1.0, 0.9)


def test_overlap_weights_and_interpolation_statuses():
    old_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    new_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    old = np.zeros((5, 6))
    new = np.ones((5, 6))
    result = overlap_ensemble(old_times, old, new_times, new, current_time=0.1, ensemble_req_dur=0.2)
    assert result["status"] == "SUCCESS"
    assert result["source_mode"][:4] == ["OLD", "BLEND", "BLEND", "NEW"]
    assert np.allclose(result["merged_states"][:4, 0], [0.0, 1 / 3, 2 / 3, 1.0])
    assert sample_trajectory_at_time(old_times, old, -0.1)["status"] == "BEFORE_START"
    assert sample_trajectory_at_time(old_times, old, 0.2)["status"] == "VALID"
    assert sample_trajectory_at_time(old_times, old, 0.6)["status"] == "EXPIRED"
    ticks = generate_30hz_schedule(4.0, 0.1)
    assert np.allclose(ticks, [4.0, 4.0 + 1 / 30, 4.0 + 2 / 30, 4.1])


def test_bp_delta_exact_hold_tick0_and_no_double_terms():
    config, _ = config_and_mapping()
    gravity = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    q_nom_hold = np.array([1, 2, 3, 4, 5.0])
    q_actual_hold = q_nom_hold - np.array([10, -10, 1, -1, 0.2])
    hold = controller.compute_hold(q_nom_hold, q_actual_hold)
    expected_corr = np.clip(0.5 * (q_nom_hold - q_actual_hold), -2.0, 2.0)
    expected_support = gravity.support_bias_deg(np.deg2rad(q_actual_hold))
    assert np.allclose(hold, q_nom_hold + expected_support + expected_corr)
    origin = controller.begin_trajectory(q_actual_hold, q_nom_hold, hold)
    assert np.array_equal(controller.compute_trajectory(q_nom_hold, q_actual_hold), hold)
    delta_nom = np.array([0.2, -0.1, 0.3, 0, -0.2])
    delta_actual = np.array([0.1, -0.2, 0.25, 0, -0.1])
    command = controller.compute_trajectory(q_nom_hold + delta_nom, q_actual_hold + delta_actual)
    current_support = gravity.support_bias_deg(np.deg2rad(q_actual_hold + delta_actual))
    expected = (
        hold
        + delta_nom
        + (current_support - origin.b_support_0)
        + np.clip(0.5 * (delta_nom - delta_actual), -2.0, 2.0)
    )
    assert np.allclose(command, expected)
    assert controller.origin is origin


def test_scheduler_request_matching_immutability_and_arrival_state():
    scheduler = ChunkScheduler()
    first = observation(0, 0.0)
    context = scheduler.register_request(first)
    first.s_obs.setflags(write=True)
    first.s_obs[0] = 999.0
    assert context.s_obs[0] != 999.0
    assert context.t_q_obs == first.t_q_obs
    assert context.frame_metadata == first.frame_metadata
    with pytest.raises(TypeError):
        context.frame_metadata["mutated"] = True
    actual = np.array([0.25, 0, 0.2, 0, 0, 0.5])
    result = scheduler.handle_response(0, np.zeros((1, 15, 6)), 0.15, actual)
    assert result["status"] == "INTEGRATED"
    assert np.array_equal(result["reanchor"]["states"][0], actual)
    assert scheduler.handle_response(0, np.zeros((15, 6)), 0.2, actual)["status"] == "DUPLICATE_RESPONSE"
    assert scheduler.handle_response(99, np.zeros((15, 6)), 0.2, actual)["status"] == "UNKNOWN_RESPONSE"
    scheduler.register_request(observation(1, 0.9, actual))
    assert scheduler.sample(1.0)["status"] == "VALID"
    second = scheduler.handle_response(1, np.zeros((15, 6)), 1.05, actual)
    assert second["status"] == "INTEGRATED"
    assert second["ensemble"]["status"] in {"SUCCESS", "SHORT_OVERLAP_FALLBACK", "NO_VALID_OVERLAP"}


def test_scheduler_out_of_order_and_expired_response():
    scheduler = ChunkScheduler()
    scheduler.register_request(observation(1, 0.0))
    scheduler.register_request(observation(2, 0.01))
    assert scheduler.handle_response(2, np.zeros((15, 6)), 0.2, np.zeros(6))["status"] == "INTEGRATED"
    assert scheduler.handle_response(1, np.zeros((15, 6)), 0.21, np.zeros(6))["status"] == "OUT_OF_ORDER_RESPONSE"
    scheduler.register_request(observation(3, 1.0))
    assert scheduler.handle_response(3, np.zeros((15, 6)), 2.5, np.zeros(6))["status"] == "EXPIRED_RESPONSE"


class FakeBus:
    def __init__(self, mismatch=None, fail_write=False):
        self.pid = {key: {name: 0 for name in ARM_JOINTS} for key in ("p", "i", "d")}
        self.register_key = {"P_Coefficient": "p", "I_Coefficient": "i", "D_Coefficient": "d"}
        self.mismatch = mismatch
        self.fail_write = fail_write
        self.goal_writes = []
        self.connected = False
        self.connect_handshakes = []
        self.disconnect_torque_flags = []

    def connect(self, handshake=True):
        self.connected = True
        self.connect_handshakes.append(bool(handshake))

    def write(self, register, name, value, normalize=False):
        self.pid[self.register_key[register]][name] = int(value)

    def sync_read(self, register, motors=None, normalize=True):
        if register in self.register_key:
            key = self.register_key[register]
            values = self.pid[key].copy()
            if self.mismatch == key:
                values[ARM_JOINTS[0]] += 1
            return values
        if register == "Present_Position":
            return {name: 0 for name in ALL_MOTORS}
        raise KeyError(register)

    def sync_write(self, register, values, normalize=True):
        if self.fail_write:
            raise ConnectionError("fake failure")
        self.goal_writes.append((register, dict(values), normalize))

    def disconnect(self, disable_torque):
        self.disconnect_torque_flags.append(bool(disable_torque))
        self.connected = False


def make_motor(bus):
    config = load_runtime_config(CONFIG_PATH)
    return MotorIO(
        config.deployment["motor"]["port"],
        config.motor_calibration,
        config.deployment["motor"]["pid"],
        bus=bus,
    )


@pytest.mark.parametrize(
    ("width", "expected_raw"),
    [(0.441305, 600), (0.0, 3000), (0.441305 * 0.5, 1800)],
)
def test_gripper_mock_motor_boundary(width, expected_raw):
    config, mapping = config_and_mapping()
    bus = FakeBus()
    motor = make_motor(bus)
    motor.connect(permit_motion=True)
    goals = {name: 2000 for name in ARM_JOINTS}
    goals["gripper"] = mapping.gripper_raw_from_width(width)
    motor.write_goal_ticks(goals)
    assert len(bus.goal_writes) == 1
    register, written, normalize = bus.goal_writes[0]
    assert register == "Goal_Position"
    assert written["gripper"] == expected_raw
    assert normalize is False


@pytest.mark.parametrize("mismatch", ["p", "i", "d"])
def test_mock_pid_mismatch_blocks_motion(mismatch):
    motor = make_motor(FakeBus(mismatch=mismatch))
    with pytest.raises(SafetyError):
        motor.connect(permit_motion=True)
    assert not motor.motion_permitted


def test_mock_motor_single_goal_boundary_and_comm_failure():
    bus = FakeBus()
    motor = make_motor(bus)
    motor.connect(permit_motion=True)
    goals = {name: 2000 for name in ALL_MOTORS}
    assert motor.write_goal_ticks(goals) == goals
    assert len(bus.goal_writes) == 1 and bus.goal_writes[0][0] == "Goal_Position"
    failing = make_motor(FakeBus(fail_write=True))
    failing.connect(permit_motion=True)
    with pytest.raises(SafetyError):
        failing.write_goal_ticks(goals)
    assert failing.last_safe_goal is None and not failing.motion_permitted


def test_mock_read_only_connect_performs_no_pid_or_goal_write():
    bus = FakeBus()
    motor = make_motor(bus)
    motor.connect(read_only=True)
    assert not motor.motion_permitted
    assert motor.read_only
    assert bus.connect_handshakes == [False]
    assert all(value == 0 for register in bus.pid.values() for value in register.values())
    assert bus.goal_writes == []
    state = motor.read_present_positions(normalized=False)
    assert state.normalized is False and motor.present_position_reads == 1
    assert motor.write_audit.goal_position_attempts == 0
    with pytest.raises(SafetyError):
        motor.write_goal_ticks({name: 2000 for name in ALL_MOTORS})
    assert motor.write_audit.goal_position_attempts == 1
    assert motor.write_audit.goal_position_physical_writes == 0
    with pytest.raises(SafetyError):
        motor.bus.write("P_Coefficient", ARM_JOINTS[0], 64, normalize=False)
    with pytest.raises(SafetyError):
        motor.bus.disable_torque()
    with pytest.raises(SafetyError):
        motor.bus.configure_motors()
    audit = motor.write_audit
    assert audit.pid_attempts == 1 and audit.pid_physical_writes == 0
    assert audit.torque_attempts == 1 and audit.torque_physical_writes == 0
    assert audit.other_register_attempts == 1 and audit.other_register_physical_writes == 0
    motor.disconnect()
    assert bus.disconnect_torque_flags == [False]


def test_safety_limits_and_stale_status():
    limits = {name: (-1.0, 1.0) for name in ARM_JOINTS}
    assert np.array_equal(require_joint_limits(np.zeros(5), ARM_JOINTS, limits), np.zeros(5))
    with pytest.raises(SafetyError):
        require_joint_limits([2, 0, 0, 0, 0], ARM_JOINTS, limits)
    with pytest.raises(SafetyError):
        require_ik_success(np.zeros(5), False)
    with pytest.raises(SafetyError):
        require_ik_success([np.nan, 0, 0, 0, 0], True)
    with pytest.raises(SafetyError):
        require_raw_goal_ticks({name: 2000 for name in ARM_JOINTS}, ALL_MOTORS)
    assert target_status(-1, 0, 1) == TargetStatus.BEFORE_START
    assert target_status(0.5, 0, 1) == TargetStatus.VALID
    assert target_status(2, 0, 1) == TargetStatus.EXPIRED


def test_camera_pure_preparation_and_frame_snapshot():
    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    rgb = prepare_transport_rgb(bgr)
    assert rgb.shape == (256, 256, 3) and rgb[0, 0, 2] == 255
    frame = FrameSnapshot(rgb, 1.0, 1, bgr.shape)
    assert not frame.image_rgb.flags.writeable


class FakeCapture:
    def __init__(self, fail=False):
        self.fail = fail
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        import time

        time.sleep(0.002)
        return (False, None) if self.fail else (True, np.zeros((12, 16, 3), dtype=np.uint8))

    def release(self):
        self.released = True


def test_camera_fake_backend_freshness_and_dropout():
    import time

    capture = FakeCapture()
    camera = LatestFrameCamera(0, 16, 12, 0.5, capture_factory=lambda *args: capture, clock=lambda: 1.0)
    camera.start()
    deadline = time.monotonic() + 0.2
    frame = None
    while frame is None and time.monotonic() < deadline:
        try:
            frame = camera.latest(now=1.1)
        except CameraError:
            time.sleep(0.002)
    assert frame is not None and frame.timestamp == 1.0
    with pytest.raises(CameraError):
        camera.latest(now=2.0)
    camera.stop()
    assert capture.released

    failed_capture = FakeCapture(fail=True)
    failed = LatestFrameCamera(0, 16, 12, 0.5, capture_factory=lambda *args: failed_capture)
    failed.start()
    time.sleep(0.01)
    with pytest.raises(CameraError):
        failed.latest()
    failed.stop()


def test_ssh_worker_payload_is_shell_quoted_and_control_socket_is_optional(monkeypatch):
    commands = []

    class FakeProcess:
        pass

    def fake_popen(command, **_kwargs):
        commands.append(command)
        return FakeProcess()

    remote = {
        "checkpoint": "/remote/checkpoint",
        "connect_timeout_s": 10.0,
        "ready_timeout_s": 120.0,
        "port": 17970,
        "user": "robot",
        "host": "example.invalid",
        "python": "/remote/python",
    }
    transport = SubprocessSSHTransport(remote)
    monkeypatch.setenv("GOPRO_UMI_SSH_CONTROL_PATH", "/tmp/validated-control")
    monkeypatch.setenv("GOPRO_UMI_REMOTE_BASE_MODEL_PATH", "/tmp/base-model-metadata")
    monkeypatch.setenv("GOPRO_UMI_REMOTE_HF_HOME", "/tmp/hf-home")
    worker = transport._worker_source()
    assert "base_model_override = '/tmp/base-model-metadata'" in worker
    assert "hf_home_override = '/tmp/hf-home'" in worker
    assert "policy_config.load_vlm_weights = False" in worker
    assert "strict=base_model_override is not None" in worker
    monkeypatch.setattr(remote_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(transport, "_read_json", lambda _timeout: {"status": "ready"})
    transport.start()
    command = commands[0]
    assert command[1:3] == ["-S", "/tmp/validated-control"]
    assert command[-1].startswith("'") and command[-1].endswith("'")
    assert ";exec(" in command[-1]
