#!/usr/bin/env python3
"""Thin canonical SmolVLA deployment orchestrator.

STEP 3 validates only ``--mode static``. Live modes require an explicit gate and
remain hardware validation work for STEP 4.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from config.runtime_config import RuntimeConfig, load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator
from control.joint_mapping import JointMapping
from control.motor_io import ARM_JOINTS, MotorIO
from control.safety import SafetyError, require_ik_success, require_joint_limits
from ik.ik_solver_v7 import DLSInverseKinematicsV7
from inference.camera import LatestFrameCamera
from inference.observation import (
    ActualStateSnapshot,
    build_aligned_observation,
    build_observation,
    state_from_motor_sample,
)
from inference.se3_contract import (
    StateHistory,
    chunk10_to_incremental6,
    relative_state10,
)
from inference.remote_smolvla import RemoteRequest, RemoteSmolVLAClient, SubprocessSSHTransport
from telemetry.runtime_logger import RuntimeLogger
from trajectory.chunk_scheduler import ChunkScheduler


DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DEPLOY_DIR / "config" / "deployment.yaml"


class RunMode(str, Enum):
    STATIC = "static"
    SHADOW = "shadow"
    PREVIEW = "preview"
    EXECUTE = "execute"


@dataclass
class RuntimeParts:
    config: RuntimeConfig
    mapping: JointMapping
    ik: DLSInverseKinematicsV7
    gravity: GravityCompensator
    controller: BPDeltaController
    scheduler: ChunkScheduler
    motor: MotorIO
    camera: LatestFrameCamera
    remote: RemoteSmolVLAClient


def build_parts(config: RuntimeConfig) -> RuntimeParts:
    deployment = config.deployment
    mapping = JointMapping(config.motor_mapping_path, config.motor_calibration)
    ik = DLSInverseKinematicsV7(str(DEPLOY_DIR.parent / "urdf" / "so_arm_with_gopro_final.urdf"))
    gravity = GravityCompensator.from_runtime_config(config)
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    scheduler = ChunkScheduler(
        deployment["scheduler"]["request_stride_s"],
        deployment["scheduler"]["transition_overlap_s"],
        config.action_period_s,
    )
    motor = MotorIO(
        deployment["motor"]["port"],
        config.motor_calibration,
        deployment["motor"]["pid"],
        disable_torque_on_disconnect=deployment["motor"]["disable_torque_on_disconnect"],
    )
    camera_config = deployment["camera"]
    camera = LatestFrameCamera(
        camera_config["device"],
        camera_config["width"],
        camera_config["height"],
        camera_config["stale_timeout_s"],
    )
    remote_config = deployment["remote"]
    remote = RemoteSmolVLAClient(
        SubprocessSSHTransport(remote_config),
        remote_config["request_timeout_s"],
        remote_config["request_queue_size"],
        remote_config["response_queue_size"],
    )
    return RuntimeParts(config, mapping, ik, gravity, controller, scheduler, motor, camera, remote)


class CanonicalRuntime:
    def __init__(self, parts: RuntimeParts, mode: RunMode):
        self.parts = parts
        self.mode = mode
        self.request_id = 0
        self.q_seed = np.deg2rad(parts.config.start_urdf_deg)
        self.q_cmd_hold: np.ndarray | None = None
        model_config = parts.config.deployment.get("model", {})
        self.action_contract = str(model_config.get("action_contract", "incremental_v1"))
        if self.action_contract not in {"incremental_v1", "se3_relwp_v1"}:
            raise SafetyError(f"unknown action_contract: {self.action_contract}")
        alignment_config = parts.config.deployment.get("observation_alignment", {}) or {}
        self.alignment_enabled = bool(alignment_config.get("enable", False))
        self.camera_latency_s = float(alignment_config.get("camera_glass_to_host_latency_s", 0.088))
        if not np.isfinite(self.camera_latency_s) or self.camera_latency_s < 0.0:
            raise SafetyError("camera_glass_to_host_latency_s must be finite and >= 0")
        self.state_history = StateHistory()
        self._pending_t_obs: dict[int, np.ndarray] = {}

    def _actual_state(self) -> ActualStateSnapshot:
        degree_state = self.parts.motor.read_present_positions(normalized=True)
        tick_state = self.parts.motor.read_present_positions(normalized=False)
        positions = {name: degree_state.positions[name] for name in ARM_JOINTS}
        positions["gripper"] = tick_state.positions["gripper"]
        actual = state_from_motor_sample(
            positions,
            max(degree_state.timestamp, tick_state.timestamp),
            self.parts.mapping,
            self.parts.ik,
        )
        if actual.transform is not None and (self.action_contract == "se3_relwp_v1" or self.alignment_enabled):
            self.state_history.push(actual.timestamp, actual.transform, actual.state[5], actual.state)
        return actual

    def _goals(self, q_command_deg: np.ndarray, gripper_width: float) -> dict[str, int]:
        raw_degrees = self.parts.mapping.urdf_degrees_to_raw_degrees(q_command_deg)
        goals = self.parts.mapping.arm_degrees_to_raw_ticks(raw_degrees)
        goals["gripper"] = self.parts.mapping.gripper_raw_from_width(gripper_width)
        return goals

    def _establish_start_and_hold(self) -> ActualStateSnapshot:
        config = self.parts.config
        motor_config = config.deployment["motor"]
        if self.mode is RunMode.EXECUTE:
            start_ticks = self.parts.mapping.arm_degrees_to_raw_ticks(config.start_raw_deg)
            self.parts.motor.move_to_start_ticks(
                start_ticks,
                int(motor_config["start_move_steps"]),
                float(motor_config["start_move_duration_s"]),
            )
            time.sleep(float(motor_config["settle_duration_s"]))
        actual = self._actual_state()
        deadline = time.monotonic() + float(motor_config["hold_presettle_s"])
        while True:
            self.q_cmd_hold = self.parts.controller.compute_hold(config.start_urdf_deg, actual.q_urdf_degrees)
            if self.mode is RunMode.EXECUTE:
                goals = self._goals(self.q_cmd_hold, actual.state[5])
                self.parts.motor.write_goal_ticks(goals)
            if self.mode is not RunMode.EXECUTE or time.monotonic() >= deadline:
                break
            time.sleep(config.control_period_s)
            actual = self._actual_state()
        return self._actual_state()

    def _submit_if_due(self, now: float) -> None:
        if not self.parts.scheduler.should_request(now):
            return
        actual = self._actual_state()
        frame = self.parts.camera.latest(now)
        if self.alignment_enabled:
            # UMI-style latency matching: back-date the frame to its capture time and
            # interpolate the robot state there, so image and state are simultaneous
            # (the model then sees a "normal dataset" tuple; latency never enters training).
            t_img = float(frame.timestamp) - self.camera_latency_s
            aligned_transform, _, aligned_state = self.state_history.interpolate(t_img)
            if aligned_state is None:
                raise SafetyError("state history has no state6 samples for alignment")
            payload_state = None
            if self.action_contract == "se3_relwp_v1":
                previous_transform, previous_gripper, _ = self.state_history.interpolate(
                    t_img - self.parts.config.action_period_s
                )
                payload_state = relative_state10(aligned_transform, previous_transform, previous_gripper)
            snapshot = build_aligned_observation(
                self.request_id, frame, actual, t_img, aligned_state, aligned_transform, payload_state
            )
            if self.action_contract == "se3_relwp_v1":
                self._pending_t_obs[self.request_id] = np.asarray(aligned_transform, dtype=np.float64).copy()
            self.parts.scheduler.register_request(snapshot)
            self.parts.remote.submit(RemoteRequest.from_observation(snapshot))
            self.request_id += 1
            return
        payload_state = None
        if self.action_contract == "se3_relwp_v1":
            if actual.transform is None:
                raise SafetyError("se3_relwp_v1 requires FK transforms on state snapshots")
            previous_transform, previous_gripper, _ = self.state_history.lagged(actual.timestamp)
            payload_state = relative_state10(actual.transform, previous_transform, previous_gripper)
        snapshot = build_observation(self.request_id, frame, actual, payload_state=payload_state)
        if self.action_contract == "se3_relwp_v1":
            self._pending_t_obs[self.request_id] = np.asarray(actual.transform, dtype=np.float64).copy()
        self.parts.scheduler.register_request(snapshot)
        self.parts.remote.submit(RemoteRequest.from_observation(snapshot))
        self.request_id += 1

    def _receive(self) -> None:
        response = self.parts.remote.poll()
        while response is not None:
            if response.error is not None or response.action is None:
                raise SafetyError(f"remote inference failed for request {response.request_id}")
            arrival = self._actual_state()
            action = response.action
            if self.action_contract == "se3_relwp_v1":
                t_obs_transform = self._pending_t_obs.pop(response.request_id, None)
                if t_obs_transform is None:
                    raise SafetyError(f"missing observation transform for request {response.request_id}")
                context = self.parts.scheduler.pending.get(response.request_id)
                if context is None:
                    raise SafetyError(f"no pending context for request {response.request_id}")
                action = chunk10_to_incremental6(t_obs_transform, context.s_obs, action)
            result = self.parts.scheduler.handle_response(
                response.request_id, action, response.t_arr, arrival.state
            )
            if result["status"] not in {"INTEGRATED", "EXPIRED_RESPONSE"}:
                raise SafetyError(f"scheduler rejected response: {result['status']}")
            response = self.parts.remote.poll()

    def _control_tick(self, now: float) -> dict:
        sampled = self.parts.scheduler.sample(now)
        if sampled["status"] != "VALID":
            return {"trajectory_status": sampled["status"], "write_status": "SKIPPED"}
        target = sampled["action"]
        actual = self._actual_state()
        rotation = Rotation.from_euler("ZYX", [actual.yaw_radians, target[4], target[3]]).as_matrix()
        q_nom = self.parts.ik.solve(target[:3], rotation, self.q_seed)
        require_ik_success(q_nom, self.parts.ik.last_converged)
        require_joint_limits(q_nom, self.parts.ik.joint_names, self.parts.ik.PHYSICAL_JOINT_LIMITS_RAD)
        require_joint_limits(q_nom, self.parts.ik.joint_names, self.parts.ik.safe_limits)
        q_nom_deg = np.rad2deg(q_nom)
        if self.parts.controller.origin is None:
            if self.q_cmd_hold is None:
                raise SafetyError("HOLD command is unavailable at trajectory onset")
            self.parts.controller.begin_trajectory(actual.q_urdf_degrees, q_nom_deg, self.q_cmd_hold)
        q_command = self.parts.controller.compute_trajectory(q_nom_deg, actual.q_urdf_degrees)
        require_joint_limits(np.deg2rad(q_command), self.parts.ik.joint_names, self.parts.ik.PHYSICAL_JOINT_LIMITS_RAD)
        require_joint_limits(np.deg2rad(q_command), self.parts.ik.joint_names, self.parts.ik.safe_limits)
        goals = self._goals(q_command, float(target[5]))
        if self.mode is RunMode.EXECUTE:
            self.parts.motor.write_goal_ticks(goals)
            write_status = "WRITTEN"
        else:
            write_status = "PREVIEW_ONLY"
        self.q_seed = q_nom.copy()
        return {
            "trajectory_status": "VALID",
            "q_actual": actual.q_urdf_degrees,
            "q_nom": q_nom_deg,
            "q_cmd": q_command,
            "write_status": write_status,
            **self.parts.controller.diagnostics(),
        }

    def run(self, logger: RuntimeLogger) -> None:
        execute = self.mode is RunMode.EXECUTE
        self.parts.motor.connect(permit_motion=execute, read_only=not execute)
        self.parts.camera.start()
        self.parts.remote.start()
        try:
            start_state = self._establish_start_and_hold()
            grip_tick = float(self.parts.motor.read_present_positions(normalized=False).positions.get("gripper", self.parts.mapping.gripper_open_raw))
            phys_open = self.parts.mapping.physical_open_fraction_from_raw(grip_tick)
            ds_grip = float(start_state.state[5])
            norm_grip = (ds_grip - 0.24479102) / 0.19513467
            print(f"[GRIPPER CONTRACT] motor_raw={grip_tick:.0f}, physical_open_fraction={phys_open:.4f}, dataset_gripper={ds_grip:.4f}, model_normalized={norm_grip:+.4f}")
            next_tick = time.monotonic()
            while True:
                now = time.monotonic()
                self._receive()
                self._submit_if_due(now)
                record = self._control_tick(now)
                record["monotonic_timestamp"] = now
                logger.log(record)
                next_tick += self.parts.config.control_period_s
                time.sleep(max(0.0, next_tick - time.monotonic()))
        finally:
            self.parts.remote.close()
            self.parts.camera.stop()
            self.parts.motor.disconnect()


def static_validate(config_path: Path) -> dict:
    config = load_runtime_config(config_path)
    parts = build_parts(config)
    mapped = parts.mapping.urdf_degrees_to_raw_degrees(config.start_urdf_deg)
    transform = parts.ik.forward_kinematics(np.deg2rad(config.start_urdf_deg))
    gravity_reference = parts.gravity.evaluate(np.deg2rad(config.start_urdf_deg))
    expected_support = config.static_support_bias_deg + config.gravity_reference_bias_deg
    if (
        not np.allclose(mapped, config.start_raw_deg)
        or not np.isfinite(transform).all()
        or not np.array_equal(gravity_reference.total_support_bias_deg, expected_support)
    ):
        raise RuntimeError("static mapping/FK validation failed")
    return {"status": "PASS", "mode": "static", "hardware_access": 0, "network_access": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[mode.value for mode in RunMode], default=RunMode.STATIC.value)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-live-io", action="store_true")
    parser.add_argument("--telemetry", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = RunMode(args.mode)
    if mode is RunMode.STATIC:
        print(static_validate(args.config))
        return 0
    if not args.confirm_live_io:
        raise SystemExit("live I/O is blocked unless --confirm-live-io is explicitly supplied")
    config = load_runtime_config(args.config)
    telemetry = args.telemetry or (config.root / "runtime_logs" / f"{int(time.time())}.jsonl")
    runtime = CanonicalRuntime(build_parts(config), mode)
    try:
        with RuntimeLogger(telemetry) as logger:
            runtime.run(logger)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
