#!/usr/bin/env python3
"""STEP 8.24D: wrist_flex-only raw-step diagnostic harness.

Default mode is read-only preflight.  This module deliberately does not use a
robot follower lifecycle: that lifecycle configures and torque-cycles every motor.
Do not execute against hardware until this file has reviewer approval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


WRITE_ALLOWLIST = {"wrist_flex"}
ARM_INITIALIZATION_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ALL_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXPECTED_PID = {"P_Coefficient": 64, "I_Coefficient": 0, "D_Coefficient": 32}
ALLOWED_DELTAS_RAW_STEP = {-2, -1, 0, 1, 2}
MICRO_SEQUENCE_RAW_STEP = (1, 0, -1, 0, 2, 0, -2, 0)
BASELINE_MONITOR_SECONDS = 1.0
BASELINE_SAMPLE_PERIOD_SECONDS = 0.10
MAX_BASELINE_DRIFT_RAW_STEP = 2

# REVIEW REQUIRED: this is intentionally not an approved physical limit.  In
# execute mode it additionally requires an identical explicit CLI acknowledgement.
PROPOSED_MAX_WRIST_GOAL_PRESENT_GAP_RAW_STEP = 12
ARM_TOKEN = "STEP8_24D_WRIST_ONLY"
INITIALIZATION_ARM_TOKEN = "STEP8_24E_ARM_INIT"
SWEEP_ARM_TOKEN = "STEP8_24F_WRIST_RAW_SWEEP"
THREE_JOINT_SWEEP_ARM_TOKEN = "STEP8_24G_THREE_JOINT_RAW_SWEEP"
VISIBLE_RESPONSE_ARM_TOKEN = "STEP8_25_VISIBLE_JOINT_RESPONSE"
THREE_JOINT_SWEEP_NAMES = ("shoulder_lift", "elbow_flex", "wrist_flex")
WRIST_SWEEP_DELTAS_RAW_STEP = (1, -1, 2, -2, 4, -4, 8, -8)
SWEEP_BASELINE_SECONDS = 0.50
SWEEP_RESPONSE_SECONDS = 0.45
SWEEP_SAMPLE_PERIOD_SECONDS = 0.01
MAX_SWEEP_BASELINE_JITTER_RAW_STEP = 2
MAX_SWEEP_ABSOLUTE_DISPLACEMENT_RAW_STEP = 16
RETURN_TO_P0_TOLERANCE_RAW_STEP = 2
RETURN_STABLE_WINDOW_SECONDS = 0.15
RETURN_STABLE_TIMEOUT_SECONDS = 1.0
STS3215_RAW_STEPS_PER_REVOLUTION = 4096
VISIBLE_RESPONSE_DEGREES = (1.5, -1.5, 2.0, -2.0)
VISIBLE_RESPONSE_HOLD_SECONDS = 0.90
VISIBLE_RESPONSE_CALIBRATION_MARGIN_RAW_STEP = 1
INFERENCE_START_ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
INFERENCE_START_PHYSICAL_DEGREES = (-6.945, -81.462, 65.760, 46.132, 1.538)
START_POSE_DURATION_SECONDS = 3.5
START_POSE_UPDATE_HZ = 30.0
START_POSE_SETTLE_SECONDS = 1.0
START_POSE_TRIM_GAIN = 0.5
START_POSE_TRIM_MAX_ADJUSTMENT_DEGREES = 0.75
START_POSE_TRIM_WAIT_SECONDS = 0.4
START_POSE_TRIM_MAX_ITERATIONS = 8
START_POSE_CONVERGENCE_TOLERANCE_DEGREES = 0.5
START_POSE_POST_CONVERGENCE_HOLD_SECONDS = 0.5


class BusProtocol(Protocol):
    calibration: Mapping[str, Any]

    def connect(self, handshake: bool = True) -> None: ...
    def disconnect(self, disable_torque: bool = True) -> None: ...
    def sync_read(self, register_name: str, motors: Any = None, *, normalize: bool = True) -> dict[str, Any]: ...
    def sync_write(self, register_name: str, values: dict[str, Any], *, normalize: bool = True) -> None: ...


@dataclass
class WriteAudit:
    goal_write_count_wrist: int = 0
    goal_write_count_pan: int = 0
    goal_write_count_lift: int = 0
    goal_write_count_elbow: int = 0
    goal_write_count_wrist_roll: int = 0
    goal_write_count_gripper: int = 0
    torque_write_count: int = 0
    pid_write_count: int = 0
    config_write_count: int = 0

    def assert_forbidden_counters_zero(self) -> None:
        assert self.goal_write_count_pan == 0
        assert self.goal_write_count_lift == 0
        assert self.goal_write_count_elbow == 0
        assert self.goal_write_count_wrist_roll == 0
        assert self.goal_write_count_gripper == 0
        assert self.torque_write_count == 0
        assert self.pid_write_count == 0
        assert self.config_write_count == 0


@dataclass(frozen=True)
class WristBaseline:
    wrist_goal_raw_step: int
    wrist_present_raw_step: int
    wrist_range_min_raw_step: int
    wrist_range_max_raw_step: int


@dataclass(frozen=True)
class PreflightReport:
    baseline: WristBaseline
    telemetry_raw: Mapping[str, Mapping[str, int]]


def _integer_raw_step(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise RuntimeError(f"{field_name} must be an integer raw hardware step, got {value!r}")
    return int(value)


def _calibration_range_raw_step(bus: BusProtocol, motor_name: str) -> tuple[int, int]:
    calibration = bus.calibration[motor_name]
    return (
        _integer_raw_step(calibration.range_min, f"{motor_name}.range_min_raw_step"),
        _integer_raw_step(calibration.range_max, f"{motor_name}.range_max_raw_step"),
    )


def _read_raw_register_all(bus: BusProtocol, register_name: str) -> dict[str, int]:
    # Position registers are always explicitly raw hardware steps.
    values = bus.sync_read(register_name, normalize=False)
    missing = set(ALL_MOTOR_NAMES) - set(values)
    if missing:
        raise RuntimeError(f"{register_name} missing motors: {sorted(missing)}")
    return {motor_name: _integer_raw_step(values[motor_name], f"{register_name}.{motor_name}") for motor_name in ALL_MOTOR_NAMES}


def _read_raw_register_arm(bus: BusProtocol, register_name: str) -> dict[str, int]:
    values = bus.sync_read(register_name, motors=list(ARM_INITIALIZATION_MOTOR_NAMES), normalize=False)
    missing = set(ARM_INITIALIZATION_MOTOR_NAMES) - set(values)
    if missing:
        raise RuntimeError(f"{register_name} missing arm motors: {sorted(missing)}")
    return {
        motor_name: _integer_raw_step(values[motor_name], f"{register_name}.{motor_name}")
        for motor_name in ARM_INITIALIZATION_MOTOR_NAMES
    }


def _read_nonposition_register_all(bus: BusProtocol, register_name: str) -> dict[str, int]:
    values = bus.sync_read(register_name, normalize=False)
    missing = set(ALL_MOTOR_NAMES) - set(values)
    if missing:
        raise RuntimeError(f"{register_name} missing motors: {sorted(missing)}")
    return {motor_name: int(values[motor_name]) for motor_name in ALL_MOTOR_NAMES}


def _validate_preconditions(telemetry_raw: Mapping[str, Mapping[str, int]]) -> None:
    if any(value != 1 for value in telemetry_raw["Torque_Enable"].values()):
        raise RuntimeError("preflight refused: torque is not ON for every motor")
    for register_name, expected_value in EXPECTED_PID.items():
        if any(value != expected_value for value in telemetry_raw[register_name].values()):
            raise RuntimeError(f"preflight refused: {register_name} is not {expected_value} for every motor")
    if any(value != 0 for value in telemetry_raw["Status"].values()):
        raise RuntimeError("preflight refused: nonzero motor Status")


def _print_arm_precondition_readout(telemetry_raw: Mapping[str, Mapping[str, int]]) -> None:
    arm_motor_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    for register_name in ("Torque_Enable", "P_Coefficient", "I_Coefficient", "D_Coefficient"):
        print(f"{register_name}:")
        for motor_name in arm_motor_names:
            print(f"{motor_name}: {telemetry_raw[register_name][motor_name]}")


def _validate_gripper_read_only(telemetry_raw: Mapping[str, Mapping[str, int]], bus: BusProtocol) -> None:
    gripper_present_raw_step = telemetry_raw["Present_Position"]["gripper"]
    gripper_range_min_raw_step, gripper_range_max_raw_step = _calibration_range_raw_step(bus, "gripper")
    if not gripper_range_min_raw_step <= gripper_present_raw_step <= gripper_range_max_raw_step:
        raise RuntimeError("preflight refused: gripper Present_Position is outside configured calibration range")


def _monitor_baseline_read_only(
    bus: BusProtocol,
    initial_present_raw: Mapping[str, int],
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> None:
    deadline = monotonic_fn() + BASELINE_MONITOR_SECONDS
    while monotonic_fn() < deadline:
        present_raw = _read_raw_register_all(bus, "Present_Position")
        moving = {
            motor_name: present_raw[motor_name] - initial_present_raw[motor_name]
            for motor_name in ALL_MOTOR_NAMES
            if abs(present_raw[motor_name] - initial_present_raw[motor_name]) > MAX_BASELINE_DRIFT_RAW_STEP
        }
        if moving:
            raise RuntimeError(f"preflight refused: material motion detected: {moving}")
        sleep_fn(BASELINE_SAMPLE_PERIOD_SECONDS)


def preflight_read_only(
    bus: BusProtocol,
    *,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> PreflightReport:
    """Read and validate state. This function never writes a motor register."""
    telemetry_raw = {
        "Torque_Enable": _read_nonposition_register_all(bus, "Torque_Enable"),
        "P_Coefficient": _read_nonposition_register_all(bus, "P_Coefficient"),
        "I_Coefficient": _read_nonposition_register_all(bus, "I_Coefficient"),
        "D_Coefficient": _read_nonposition_register_all(bus, "D_Coefficient"),
        "Present_Position": _read_raw_register_all(bus, "Present_Position"),
        "Goal_Position": _read_raw_register_all(bus, "Goal_Position"),
        "Present_Temperature": _read_nonposition_register_all(bus, "Present_Temperature"),
        "Present_Load": _read_nonposition_register_all(bus, "Present_Load"),
        "Status": _read_nonposition_register_all(bus, "Status"),
    }
    _print_arm_precondition_readout(telemetry_raw)
    _validate_preconditions(telemetry_raw)
    _validate_gripper_read_only(telemetry_raw, bus)
    _monitor_baseline_read_only(bus, telemetry_raw["Present_Position"], monotonic_fn, sleep_fn)

    wrist_range_min_raw_step, wrist_range_max_raw_step = _calibration_range_raw_step(bus, "wrist_flex")
    wrist_goal_raw_step = telemetry_raw["Goal_Position"]["wrist_flex"]
    wrist_present_raw_step = telemetry_raw["Present_Position"]["wrist_flex"]
    for delta_raw_step in ALLOWED_DELTAS_RAW_STEP:
        wrist_target_raw_step = wrist_goal_raw_step + delta_raw_step
        if not wrist_range_min_raw_step + 2 <= wrist_target_raw_step <= wrist_range_max_raw_step - 2:
            raise RuntimeError("preflight refused: G0 +/- 2 violates wrist calibrated safety margin")

    return PreflightReport(
        baseline=WristBaseline(
            wrist_goal_raw_step=wrist_goal_raw_step,
            wrist_present_raw_step=wrist_present_raw_step,
            wrist_range_min_raw_step=wrist_range_min_raw_step,
            wrist_range_max_raw_step=wrist_range_max_raw_step,
        ),
        telemetry_raw=telemetry_raw,
    )


class WristOnlyGoalWriter:
    """The sole Goal_Position writer in this module; it has no motor argument."""

    def __init__(self, bus: BusProtocol, baseline: WristBaseline, audit: WriteAudit):
        self._bus = bus
        self._baseline = baseline
        self._audit = audit

    def write_wrist_raw_delta(self, delta_raw_step: int) -> None:
        if isinstance(delta_raw_step, bool) or not isinstance(delta_raw_step, int):
            raise RuntimeError("delta_raw_step must be an integer")
        if delta_raw_step not in ALLOWED_DELTAS_RAW_STEP:
            raise RuntimeError(f"delta_raw_step must be one of {sorted(ALLOWED_DELTAS_RAW_STEP)}")
        wrist_target_raw_step = self._baseline.wrist_goal_raw_step + delta_raw_step
        if not isinstance(wrist_target_raw_step, int) or abs(delta_raw_step) > 2:
            raise RuntimeError("invalid wrist raw-step target")
        if not (
            self._baseline.wrist_range_min_raw_step + 2
            <= wrist_target_raw_step
            <= self._baseline.wrist_range_max_raw_step - 2
        ):
            raise RuntimeError("wrist raw-step target violates calibrated safety margin")
        motors = {"wrist_flex"}
        assert set(motors) <= WRITE_ALLOWLIST
        assert set(motors) == {"wrist_flex"}
        self._bus.sync_write("Goal_Position", {"wrist_flex": wrist_target_raw_step}, normalize=False)
        self._audit.goal_write_count_wrist += 1
        wrist_goal_readback_raw_step = _integer_raw_step(
            self._bus.sync_read("Goal_Position", motors=["wrist_flex"], normalize=False)["wrist_flex"],
            "wrist_goal_readback_raw_step",
        )
        if wrist_goal_readback_raw_step != wrist_target_raw_step:
            raise RuntimeError(
                f"wrist Goal_Position readback mismatch: {wrist_goal_readback_raw_step} != {wrist_target_raw_step}"
            )
        self._audit.assert_forbidden_counters_zero()


class ArmKnownStateInitializer:
    """Fixed five-arm initialization writer; gripper is structurally absent."""

    def __init__(self, bus: BusProtocol, audit: WriteAudit):
        self._bus = bus
        self._audit = audit

    @staticmethod
    def _assert_five_arm_only(motor_values: Mapping[str, int]) -> None:
        motors = set(motor_values)
        assert motors == set(ARM_INITIALIZATION_MOTOR_NAMES)
        assert "gripper" not in motors

    def _write_torque_arm_only(self, enabled: int) -> None:
        torque_values = {motor_name: enabled for motor_name in ARM_INITIALIZATION_MOTOR_NAMES}
        self._assert_five_arm_only(torque_values)
        self._bus.sync_write("Torque_Enable", torque_values, normalize=False)
        self._audit.torque_write_count += 1

    def _write_goal_arm_only(self, present_raw_step: Mapping[str, int]) -> None:
        goal_raw_step = {
            motor_name: _integer_raw_step(present_raw_step[motor_name], f"{motor_name}.present_raw_step")
            for motor_name in ARM_INITIALIZATION_MOTOR_NAMES
        }
        self._assert_five_arm_only(goal_raw_step)
        self._bus.sync_write("Goal_Position", goal_raw_step, normalize=False)
        self._audit.goal_write_count_pan += 1
        self._audit.goal_write_count_lift += 1
        self._audit.goal_write_count_elbow += 1
        self._audit.goal_write_count_wrist += 1
        self._audit.goal_write_count_wrist_roll += 1

    def _write_pid_arm_only(self) -> None:
        for register_name, expected_value in EXPECTED_PID.items():
            pid_values = {motor_name: expected_value for motor_name in ARM_INITIALIZATION_MOTOR_NAMES}
            self._assert_five_arm_only(pid_values)
            self._bus.sync_write(register_name, pid_values, normalize=False)
            self._audit.pid_write_count += 1

    def initialize_known_arm_state(self, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        present_raw_step = _read_raw_register_arm(self._bus, "Present_Position")
        print(f"initial arm Present_Position raw steps:\n{present_raw_step}")
        self._write_torque_arm_only(0)
        self._write_goal_arm_only(present_raw_step)
        self._write_pid_arm_only()

        goal_readback_raw_step = _read_raw_register_arm(self._bus, "Goal_Position")
        if goal_readback_raw_step != present_raw_step:
            raise RuntimeError(
                f"initialization refused: Goal_Position readback mismatch: {goal_readback_raw_step} != {present_raw_step}"
            )
        for register_name, expected_value in EXPECTED_PID.items():
            pid_readback = _read_nonposition_register_all(self._bus, register_name)
            arm_values = {motor_name: pid_readback[motor_name] for motor_name in ARM_INITIALIZATION_MOTOR_NAMES}
            if any(value != expected_value for value in arm_values.values()):
                raise RuntimeError(f"initialization refused: {register_name} readback mismatch: {arm_values}")

        self._write_torque_arm_only(1)
        sleep_fn(0.25)
        post_torque = _read_nonposition_register_all(self._bus, "Torque_Enable")
        post_present_raw_step = _read_raw_register_arm(self._bus, "Present_Position")
        post_goal_raw_step = _read_raw_register_arm(self._bus, "Goal_Position")
        post_pid = {
            register_name: _read_nonposition_register_all(self._bus, register_name)
            for register_name in EXPECTED_PID
        }
        print(f"post-init Torque_Enable arm:\n{ {m: post_torque[m] for m in ARM_INITIALIZATION_MOTOR_NAMES} }")
        print(f"post-init P/I/D arm:\n{ {r: {m: post_pid[r][m] for m in ARM_INITIALIZATION_MOTOR_NAMES} for r in EXPECTED_PID} }")
        print(f"post-init Present_Position raw steps:\n{post_present_raw_step}")
        print(f"post-init Goal_Position raw steps:\n{post_goal_raw_step}")
        print(
            "post-init Goal_minus_Present_raw:\n"
            f"{ {m: post_goal_raw_step[m] - post_present_raw_step[m] for m in ARM_INITIALIZATION_MOTOR_NAMES} }"
        )
        assert self._audit.goal_write_count_gripper == 0
        assert self._audit.config_write_count == 0


@dataclass(frozen=True)
class SweepSample:
    initial_raw_step: int
    first_raw_step: int
    minimum_raw_step: int
    maximum_raw_step: int
    final_raw_step: int
    unique_raw_steps: tuple[int, ...]
    first_response_latency_seconds: float | None


@dataclass(frozen=True)
class SweepResult:
    requested_delta_raw_step: int
    commanded_goal_raw_step: int
    present_before_write_raw_step: int
    response: SweepSample
    return_sample: SweepSample


class TrueRawWristSweep:
    """Fixed-delta wrist-only diagnostic; no non-wrist write API exists here."""

    def __init__(self, bus: BusProtocol, audit: WriteAudit):
        self._bus = bus
        self._audit = audit
        self._wrist_goal_raw_step: int | None = None
        self._wrist_present_raw_step: int | None = None

    def _read_wrist_raw_step(self, register_name: str) -> int:
        values = self._bus.sync_read(register_name, motors=["wrist_flex"], normalize=False)
        return _integer_raw_step(values["wrist_flex"], f"wrist_flex.{register_name}_raw_step")

    def _read_wrist_nonposition(self, register_name: str) -> int:
        values = self._bus.sync_read(register_name, motors=["wrist_flex"], normalize=False)
        return int(values["wrist_flex"])

    def _write_wrist_goal_raw_step(self, wrist_target_raw_step: int) -> None:
        if not isinstance(wrist_target_raw_step, int):
            raise RuntimeError("wrist_target_raw_step must be an integer")
        motor_values = {"wrist_flex": wrist_target_raw_step}
        assert set(motor_values) == {"wrist_flex"}
        assert set(motor_values) <= WRITE_ALLOWLIST
        self._bus.sync_write("Goal_Position", motor_values, normalize=False)
        self._audit.goal_write_count_wrist += 1
        goal_readback_raw_step = self._read_wrist_raw_step("Goal_Position")
        if goal_readback_raw_step != wrist_target_raw_step:
            raise RuntimeError(
                f"sweep refused: Goal_Position readback {goal_readback_raw_step} != {wrist_target_raw_step}"
            )
        assert self._audit.goal_write_count_gripper == 0
        assert self._audit.torque_write_count == 0
        assert self._audit.pid_write_count == 0
        assert self._audit.config_write_count == 0

    def _sample_present_raw_step(
        self,
        duration_seconds: float,
        baseline_jitter_raw_step: int,
        monotonic_fn: Callable[[], float],
        sleep_fn: Callable[[float], None],
        detect_latency: bool,
    ) -> SweepSample:
        if self._wrist_present_raw_step is None:
            raise RuntimeError("sweep baseline is not initialized")
        start = monotonic_fn()
        samples = [self._read_wrist_raw_step("Present_Position")]
        first_response_latency_seconds: float | None = None
        while monotonic_fn() - start < duration_seconds:
            sleep_fn(SWEEP_SAMPLE_PERIOD_SECONDS)
            present_raw_step = self._read_wrist_raw_step("Present_Position")
            samples.append(present_raw_step)
            if abs(present_raw_step - self._wrist_present_raw_step) > MAX_SWEEP_ABSOLUTE_DISPLACEMENT_RAW_STEP:
                raise RuntimeError("sweep aborted: unexpected large wrist displacement")
            if (
                detect_latency
                and first_response_latency_seconds is None
                and abs(present_raw_step - samples[0]) > baseline_jitter_raw_step
            ):
                first_response_latency_seconds = monotonic_fn() - start
        return SweepSample(
            initial_raw_step=samples[0],
            first_raw_step=samples[1] if len(samples) > 1 else samples[0],
            minimum_raw_step=min(samples),
            maximum_raw_step=max(samples),
            final_raw_step=samples[-1],
            unique_raw_steps=tuple(sorted(set(samples))),
            first_response_latency_seconds=first_response_latency_seconds,
        )

    def _verify_wrist_preconditions(self) -> None:
        if self._read_wrist_nonposition("Torque_Enable") != 1:
            raise RuntimeError("sweep refused: wrist_flex Torque_Enable is not 1")
        for register_name, expected_value in EXPECTED_PID.items():
            if self._read_wrist_nonposition(register_name) != expected_value:
                raise RuntimeError(f"sweep refused: wrist_flex {register_name} is not {expected_value}")

    def _require_return_to_original_p0_stable(
        self,
        monotonic_fn: Callable[[], float],
        sleep_fn: Callable[[float], None],
    ) -> None:
        if self._wrist_present_raw_step is None:
            raise RuntimeError("sweep baseline is not initialized")
        deadline = monotonic_fn() + RETURN_STABLE_TIMEOUT_SECONDS
        stable_window_start: float | None = None
        while monotonic_fn() < deadline:
            present_raw_step = self._read_wrist_raw_step("Present_Position")
            if abs(present_raw_step - self._wrist_present_raw_step) <= RETURN_TO_P0_TOLERANCE_RAW_STEP:
                if stable_window_start is None:
                    stable_window_start = monotonic_fn()
                elif monotonic_fn() - stable_window_start >= RETURN_STABLE_WINDOW_SECONDS:
                    return
            else:
                stable_window_start = None
            sleep_fn(SWEEP_SAMPLE_PERIOD_SECONDS)
        raise RuntimeError(
            "sweep aborted: wrist_flex did not return within +/-2 raw steps of original P0 in stable timeout"
        )

    def run(
        self,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> list[SweepResult]:
        self._verify_wrist_preconditions()
        self._wrist_goal_raw_step = self._read_wrist_raw_step("Goal_Position")
        self._wrist_present_raw_step = self._read_wrist_raw_step("Present_Position")
        if abs(self._wrist_goal_raw_step - self._wrist_present_raw_step) > 2:
            raise RuntimeError("sweep refused: abs(G0-P0) exceeds 2 raw steps")
        wrist_range_min_raw_step, wrist_range_max_raw_step = _calibration_range_raw_step(self._bus, "wrist_flex")
        if not (
            wrist_range_min_raw_step <= self._wrist_goal_raw_step - 8
            and self._wrist_goal_raw_step + 8 <= wrist_range_max_raw_step
        ):
            raise RuntimeError("sweep refused: G0 +/- 8 is outside wrist calibration range")

        baseline = self._sample_present_raw_step(
            SWEEP_BASELINE_SECONDS, 0, monotonic_fn, sleep_fn, detect_latency=False
        )
        baseline_jitter_raw_step = baseline.maximum_raw_step - baseline.minimum_raw_step
        print(
            "wrist baseline raw: "
            f"min={baseline.minimum_raw_step} max={baseline.maximum_raw_step} "
            f"unique={list(baseline.unique_raw_steps)} peak_to_peak={baseline_jitter_raw_step}"
        )
        if baseline_jitter_raw_step > MAX_SWEEP_BASELINE_JITTER_RAW_STEP:
            raise RuntimeError("sweep refused: baseline wrist jitter exceeds 2 raw steps")

        results: list[SweepResult] = []
        for requested_delta_raw_step in WRIST_SWEEP_DELTAS_RAW_STEP:
            wrist_target_raw_step = self._wrist_goal_raw_step + requested_delta_raw_step
            present_before_write_raw_step = self._read_wrist_raw_step("Present_Position")
            self._write_wrist_goal_raw_step(wrist_target_raw_step)
            response = self._sample_present_raw_step(
                SWEEP_RESPONSE_SECONDS,
                baseline_jitter_raw_step,
                monotonic_fn,
                sleep_fn,
                detect_latency=True,
            )
            # This is the sole normal-path restoration and occurs only after a successful response sample.
            self._write_wrist_goal_raw_step(self._wrist_goal_raw_step)
            return_sample = self._sample_present_raw_step(
                SWEEP_RESPONSE_SECONDS,
                baseline_jitter_raw_step,
                monotonic_fn,
                sleep_fn,
                detect_latency=False,
            )
            if self._read_wrist_raw_step("Goal_Position") != self._wrist_goal_raw_step:
                raise RuntimeError("sweep aborted: G0 return Goal_Position readback mismatch")
            self._require_return_to_original_p0_stable(monotonic_fn, sleep_fn)
            results.append(
                SweepResult(
                    requested_delta_raw_step=requested_delta_raw_step,
                    commanded_goal_raw_step=wrist_target_raw_step,
                    present_before_write_raw_step=present_before_write_raw_step,
                    response=response,
                    return_sample=return_sample,
                )
            )
        self._print_summary(results, baseline_jitter_raw_step)
        return results

    def _print_summary(self, results: list[SweepResult], baseline_jitter_raw_step: int) -> None:
        assert self._wrist_present_raw_step is not None
        print("delta | max_signed_response_raw | max_abs_response_raw | final_response_raw | response_above_jitter | return_final_error_raw")
        signed_responses: list[int] = []
        for result in results:
            signed = max(
                (result.response.minimum_raw_step - self._wrist_present_raw_step,
                 result.response.maximum_raw_step - self._wrist_present_raw_step),
                key=abs,
            )
            absolute = abs(signed)
            final_response = result.response.final_raw_step - self._wrist_present_raw_step
            return_error = result.return_sample.final_raw_step - self._wrist_present_raw_step
            signed_responses.append(signed)
            print(f"{result.requested_delta_raw_step:+d} | {signed} | {absolute} | {final_response} | {absolute > baseline_jitter_raw_step} | {return_error}")
        abs_responses = [abs(value) for value in signed_responses]
        if all(value == 0 for value in abs_responses):
            classification = "no measurable response"
        elif all(value <= baseline_jitter_raw_step for value in abs_responses):
            classification = "response comparable to baseline jitter"
        else:
            classification = "finite raw-step response observed"
            directional = {r.requested_delta_raw_step: s for r, s in zip(results, signed_responses)}
            if any(abs(directional[positive] + directional[-positive]) > baseline_jitter_raw_step for positive in (1, 2, 4, 8)):
                classification += "; directional asymmetry observed"
        print(f"descriptive classification: {classification}")


class ThreeJointRawSweep:
    """STEP8.24G fixed-order, one-joint-at-a-time raw response characterization."""

    def __init__(self, bus: BusProtocol, audit: WriteAudit):
        self._bus, self._audit = bus, audit

    def _read(self, register_name: str, joint_name: str) -> int:
        assert joint_name in THREE_JOINT_SWEEP_NAMES
        return _integer_raw_step(
            self._bus.sync_read(register_name, motors=[joint_name], normalize=False)[joint_name],
            f"{joint_name}.{register_name}_raw_step",
        )

    def _write_goal(self, joint_name: str, joint_target_raw_step: int) -> None:
        assert joint_name in THREE_JOINT_SWEEP_NAMES
        motor_values = {joint_name: joint_target_raw_step}
        assert set(motor_values) == {joint_name}
        assert set(motor_values).issubset(set(THREE_JOINT_SWEEP_NAMES))
        assert "gripper" not in motor_values
        self._bus.sync_write("Goal_Position", motor_values, normalize=False)
        if joint_name == "shoulder_lift": self._audit.goal_write_count_lift += 1
        elif joint_name == "elbow_flex": self._audit.goal_write_count_elbow += 1
        else: self._audit.goal_write_count_wrist += 1
        if self._read("Goal_Position", joint_name) != joint_target_raw_step:
            raise RuntimeError(f"three-joint sweep aborted: {joint_name} Goal_Position readback mismatch")
        assert self._audit.goal_write_count_gripper == 0
        assert self._audit.torque_write_count == self._audit.pid_write_count == self._audit.config_write_count == 0

    def _sample(self, joint_name: str, p0_raw_step: int, duration: float, jitter: int, clock, sleep, latency: bool, max_displacement: int = MAX_SWEEP_ABSOLUTE_DISPLACEMENT_RAW_STEP) -> SweepSample:
        start = clock(); samples = [self._read("Present_Position", joint_name)]; first = None
        while clock() - start < duration:
            sleep(SWEEP_SAMPLE_PERIOD_SECONDS); value = self._read("Present_Position", joint_name); samples.append(value)
            if abs(value - p0_raw_step) > max_displacement:
                raise RuntimeError(f"three-joint sweep aborted: unexpected large {joint_name} displacement")
            if latency and first is None and abs(value - samples[0]) > jitter: first = clock() - start
        return SweepSample(samples[0], samples[1] if len(samples)>1 else samples[0], min(samples), max(samples), samples[-1], tuple(sorted(set(samples))), first)

    def _require_return(self, joint_name: str, p0_raw_step: int, clock, sleep) -> None:
        deadline, stable_start = clock() + RETURN_STABLE_TIMEOUT_SECONDS, None
        while clock() < deadline:
            if abs(self._read("Present_Position", joint_name) - p0_raw_step) <= RETURN_TO_P0_TOLERANCE_RAW_STEP:
                stable_start = clock() if stable_start is None else stable_start
                if clock() - stable_start >= RETURN_STABLE_WINDOW_SECONDS: return
            else: stable_start = None
            sleep(SWEEP_SAMPLE_PERIOD_SECONDS)
        raise RuntimeError(f"three-joint sweep aborted: {joint_name} return-to-P0 timeout")

    def run(self, *, monotonic_fn=time.monotonic, sleep_fn=time.sleep) -> dict[str, list[SweepResult]]:
        all_results: dict[str, list[SweepResult]] = {}
        for joint_name in THREE_JOINT_SWEEP_NAMES:
            if self._read("Torque_Enable", joint_name) != 1: raise RuntimeError(f"three-joint sweep refused: {joint_name} torque is not ON")
            for reg, expected in EXPECTED_PID.items():
                if self._read(reg, joint_name) != expected: raise RuntimeError(f"three-joint sweep refused: {joint_name} {reg} is not {expected}")
            g0_raw_step, p0_raw_step = self._read("Goal_Position", joint_name), self._read("Present_Position", joint_name)
            if abs(g0_raw_step-p0_raw_step) > 3: raise RuntimeError(f"three-joint sweep refused: {joint_name} abs(G0-P0)>3")
            low, high = _calibration_range_raw_step(self._bus, joint_name)
            if not low <= g0_raw_step-8 and g0_raw_step+8 <= high: raise RuntimeError(f"three-joint sweep refused: {joint_name} G0+/-8 outside range")
            baseline = self._sample(joint_name, p0_raw_step, SWEEP_BASELINE_SECONDS, 0, monotonic_fn, sleep_fn, False)
            jitter = baseline.maximum_raw_step - baseline.minimum_raw_step
            print(f"{joint_name} baseline raw: min={baseline.minimum_raw_step} max={baseline.maximum_raw_step} unique={list(baseline.unique_raw_steps)} p2p={jitter}")
            if jitter > MAX_SWEEP_BASELINE_JITTER_RAW_STEP: raise RuntimeError(f"three-joint sweep refused: {joint_name} baseline jitter >2")
            results=[]
            for delta in WRIST_SWEEP_DELTAS_RAW_STEP:
                target = g0_raw_step + delta; before = self._read("Present_Position", joint_name)
                self._write_goal(joint_name, target)
                response = self._sample(joint_name, p0_raw_step, SWEEP_RESPONSE_SECONDS, jitter, monotonic_fn, sleep_fn, True)
                self._write_goal(joint_name, g0_raw_step)
                returned = self._sample(joint_name, p0_raw_step, SWEEP_RESPONSE_SECONDS, jitter, monotonic_fn, sleep_fn, False)
                if self._read("Goal_Position", joint_name) != g0_raw_step: raise RuntimeError(f"three-joint sweep aborted: {joint_name} G0 readback mismatch")
                self._require_return(joint_name, p0_raw_step, monotonic_fn, sleep_fn)
                result = SweepResult(delta, target, before, response, returned)
                results.append(result)
                max_signed = max(
                    response.minimum_raw_step - p0_raw_step,
                    response.maximum_raw_step - p0_raw_step,
                    key=abs,
                )
                print(
                    f"{joint_name} delta={delta:+d} G0={g0_raw_step} P0={p0_raw_step} target={target} "
                    f"max_signed={max_signed} max_abs={abs(max_signed)} "
                    f"final={response.final_raw_step-p0_raw_step} "
                    f"above_jitter={abs(max_signed)>jitter} return_error={returned.final_raw_step-p0_raw_step} "
                    f"latency={response.first_response_latency_seconds}"
                )
            all_results[joint_name]=results
        self._print_matrix(all_results)
        return all_results

    @staticmethod
    def _print_matrix(all_results: Mapping[str, list[SweepResult]]) -> None:
        print("joint | delta | response_raw | return_error_raw")
        for joint_name, results in all_results.items():
            p0 = results[0].present_before_write_raw_step
            response = {
                r.requested_delta_raw_step: max(r.response.minimum_raw_step-p0, r.response.maximum_raw_step-p0, key=abs)
                for r in results
            }
            for r in results: print(f"{joint_name} | {r.requested_delta_raw_step:+d} | {response[r.requested_delta_raw_step]} | {r.return_sample.final_raw_step-p0}")
            measurable = [abs(d) for d,v in response.items() if abs(v)>0]
            smallest = min(measurable) if measurable else "none"
            ratio4 = (response[4]/4, response[-4]/-4); ratio8=(response[8]/8,response[-8]/-8)
            print(f"{joint_name}: smallest measurable |delta|={smallest}; positive/negative asymmetry={response[4] != -response[-4] or response[8] != -response[-8]}; ratio +/-4={ratio4}; ratio +/-8={ratio8}")


class VisibleJointResponseTest(ThreeJointRawSweep):
    """STEP8.25 fixed, visibly meaningful amplitudes; no controller or tuning path."""

    @staticmethod
    def degrees_to_raw_step_delta(requested_degrees: float) -> int:
        return int(round(requested_degrees * STS3215_RAW_STEPS_PER_REVOLUTION / 360.0))

    def run(self, *, monotonic_fn=time.monotonic, sleep_fn=time.sleep) -> dict[str, list[SweepResult]]:
        raw_deltas = tuple(self.degrees_to_raw_step_delta(d) for d in VISIBLE_RESPONSE_DEGREES)
        all_results: dict[str, list[SweepResult]] = {}
        for joint_name in THREE_JOINT_SWEEP_NAMES:
            if self._read("Torque_Enable", joint_name) != 1: raise RuntimeError(f"visible test refused: {joint_name} torque is not ON")
            for reg, expected in EXPECTED_PID.items():
                if self._read(reg, joint_name) != expected: raise RuntimeError(f"visible test refused: {joint_name} {reg} is not {expected}")
            g0_raw_step, p0_raw_step = self._read("Goal_Position", joint_name), self._read("Present_Position", joint_name)
            if abs(g0_raw_step-p0_raw_step) > 3: raise RuntimeError(f"visible test refused: {joint_name} abs(G0-P0)>3")
            low, high = _calibration_range_raw_step(self._bus, joint_name)
            results=[]
            for requested_degrees, delta_raw_step in zip(VISIBLE_RESPONSE_DEGREES, raw_deltas):
                target = g0_raw_step + delta_raw_step
                if not low + VISIBLE_RESPONSE_CALIBRATION_MARGIN_RAW_STEP <= target <= high - VISIBLE_RESPONSE_CALIBRATION_MARGIN_RAW_STEP:
                    print(
                        f"SKIP: calibration range joint={joint_name} command_deg={requested_degrees:+.1f} "
                        f"target={target} allowed=[{low + VISIBLE_RESPONSE_CALIBRATION_MARGIN_RAW_STEP}, "
                        f"{high - VISIBLE_RESPONSE_CALIBRATION_MARGIN_RAW_STEP}]"
                    )
                    continue
                before = self._read("Present_Position", joint_name)
                self._write_goal(joint_name, target)
                response = self._sample(joint_name, p0_raw_step, VISIBLE_RESPONSE_HOLD_SECONDS, 0, monotonic_fn, sleep_fn, True, max_displacement=60)
                self._write_goal(joint_name, g0_raw_step)
                returned = self._sample(joint_name, p0_raw_step, VISIBLE_RESPONSE_HOLD_SECONDS, 0, monotonic_fn, sleep_fn, False, max_displacement=60)
                if self._read("Goal_Position", joint_name) != g0_raw_step: raise RuntimeError(f"visible test aborted: {joint_name} G0 readback mismatch")
                self._require_return(joint_name, p0_raw_step, monotonic_fn, sleep_fn)
                result=SweepResult(delta_raw_step, target, before, response, returned); results.append(result)
                achieved_raw=max(response.minimum_raw_step-p0_raw_step,response.maximum_raw_step-p0_raw_step,key=abs)
                achieved_deg=achieved_raw*360.0/STS3215_RAW_STEPS_PER_REVOLUTION
                ratio=achieved_raw/delta_raw_step
                return_deg=(returned.final_raw_step-p0_raw_step)*360.0/STS3215_RAW_STEPS_PER_REVOLUTION
                print(f"{joint_name} command_deg={requested_degrees:+.1f} delta_raw={delta_raw_step:+d} target={target} achieved_raw={achieved_raw} achieved_deg={achieved_deg:.3f} tracking_ratio={ratio:.3f} latency={response.first_response_latency_seconds} settling_residual_raw={response.final_raw_step-p0_raw_step} return_residual_deg={return_deg:.3f}")
            all_results[joint_name]=results
        print("joint | command_deg | achieved_deg | tracking_ratio | return_error_deg")
        for joint_name, results in all_results.items():
            if not results:
                print(f"{joint_name} | no calibration-valid visible stimuli")
                continue
            p0=results[0].present_before_write_raw_step
            ratios={}
            for r in results:
                degrees = r.requested_delta_raw_step * 360.0 / STS3215_RAW_STEPS_PER_REVOLUTION
                achieved=max(r.response.minimum_raw_step-p0,r.response.maximum_raw_step-p0,key=abs)
                achieved_deg=achieved*360.0/STS3215_RAW_STEPS_PER_REVOLUTION; ratio=achieved/r.requested_delta_raw_step
                print(f"{joint_name} | {degrees:+.1f} | {achieved_deg:.3f} | {ratio:.3f} | {(r.return_sample.final_raw_step-p0)*360.0/STS3215_RAW_STEPS_PER_REVOLUTION:.3f}")
                ratios[degrees]=ratio
            pairs = ((1.5, -1.5), (2.0, -2.0))
            summaries = []
            for positive, negative in pairs:
                positive_key = self.degrees_to_raw_step_delta(positive) * 360.0 / STS3215_RAW_STEPS_PER_REVOLUTION
                negative_key = self.degrees_to_raw_step_delta(negative) * 360.0 / STS3215_RAW_STEPS_PER_REVOLUTION
                if positive_key in ratios and negative_key in ratios:
                    summaries.append(f"mean ratio +/-{positive:.1f}={(ratios[positive_key] + ratios[negative_key]) / 2:.3f}")
                else:
                    summaries.append(f"mean ratio +/-{positive:.1f}=n/a (calibration skip)")
            print(f"{joint_name}: {'; '.join(summaries)}")
        return all_results


class InferenceStartPoseMove:
    """The only calibrated-degree five-arm Goal writer; gripper is structurally excluded."""

    def __init__(self, bus: BusProtocol, audit: WriteAudit):
        self._bus, self._audit = bus, audit

    @staticmethod
    def _read_present_physical_degrees(bus: BusProtocol) -> dict[str, float]:
        present = bus.sync_read(
            "Present_Position", motors=list(INFERENCE_START_ARM_JOINT_NAMES), normalize=True
        )
        if set(present) != set(INFERENCE_START_ARM_JOINT_NAMES):
            raise RuntimeError("start-pose refused: incomplete arm Present_Position read")
        return {joint_name: float(present[joint_name]) for joint_name in INFERENCE_START_ARM_JOINT_NAMES}

    def _write_arm_goal_physical_degrees(self, goal_physical_degrees: Mapping[str, float]) -> None:
        if set(goal_physical_degrees) != set(INFERENCE_START_ARM_JOINT_NAMES) or "gripper" in goal_physical_degrees:
            raise RuntimeError("start-pose refused: Goal_Position write must contain exactly the five arm joints")
        self._bus.sync_write("Goal_Position", dict(goal_physical_degrees), normalize=True)
        self._audit.goal_write_count_pan += 1
        self._audit.goal_write_count_lift += 1
        self._audit.goal_write_count_elbow += 1
        self._audit.goal_write_count_wrist += 1
        self._audit.goal_write_count_wrist_roll += 1
        assert self._audit.goal_write_count_gripper == 0

    def _verify_arm_preconditions(self) -> None:
        for register_name, expected_value in (("Torque_Enable", 1), *EXPECTED_PID.items()):
            values = self._bus.sync_read(
                register_name, motors=list(INFERENCE_START_ARM_JOINT_NAMES), normalize=False
            )
            if not set(INFERENCE_START_ARM_JOINT_NAMES).issubset(values) or any(
                int(values[joint_name]) != expected_value
                for joint_name in INFERENCE_START_ARM_JOINT_NAMES
            ):
                raise RuntimeError(f"start-pose refused: {register_name} is not {expected_value} for every arm joint")

    def move_and_settle(self, *, monotonic_fn=time.monotonic, sleep_fn=time.sleep) -> None:
        target_physical_degrees = dict(zip(INFERENCE_START_ARM_JOINT_NAMES, INFERENCE_START_PHYSICAL_DEGREES, strict=True))
        self._verify_arm_preconditions()
        assert self._audit.torque_write_count == self._audit.pid_write_count == self._audit.config_write_count == 0
        current_physical_degrees = self._read_present_physical_degrees(self._bus)
        print(f"inference start target physical degrees: {target_physical_degrees}")
        steps = int(START_POSE_DURATION_SECONDS * START_POSE_UPDATE_HZ)
        for step in range(steps):
            alpha_t = (step + 1) / steps
            alpha = 3.0 * alpha_t**2 - 2.0 * alpha_t**3
            interpolated_physical_degrees = {
                joint_name: float(current_physical_degrees[joint_name] + alpha * (target_physical_degrees[joint_name] - current_physical_degrees[joint_name]))
                for joint_name in INFERENCE_START_ARM_JOINT_NAMES
            }
            self._write_arm_goal_physical_degrees(interpolated_physical_degrees)
            sleep_fn(1.0 / START_POSE_UPDATE_HZ)
        sleep_fn(START_POSE_SETTLE_SECONDS)

        previous_goal_physical_degrees = dict(target_physical_degrees)
        for trim_iteration in range(START_POSE_TRIM_MAX_ITERATIONS + 1):
            actual_physical_degrees = self._read_present_physical_degrees(self._bus)
            # This sign is intentional: error = desired physical operating point - measured present position.
            errors_degrees = {
                joint_name: target_physical_degrees[joint_name] - actual_physical_degrees[joint_name]
                for joint_name in INFERENCE_START_ARM_JOINT_NAMES
            }
            if all(abs(error) <= START_POSE_CONVERGENCE_TOLERANCE_DEGREES for error in errors_degrees.values()):
                print(f"PASS start pose after {trim_iteration} trim iterations")
                print(f"start-pose target degrees: {target_physical_degrees}")
                print(f"start-pose present degrees: {actual_physical_degrees}")
                print(f"start-pose error degrees: {errors_degrees}")
                print(f"start-pose goal degrees: {previous_goal_physical_degrees}")
                sleep_fn(START_POSE_POST_CONVERGENCE_HOLD_SECONDS)
                return
            if trim_iteration == START_POSE_TRIM_MAX_ITERATIONS:
                print(f"ABORT start pose after {START_POSE_TRIM_MAX_ITERATIONS} trim iterations")
                print(f"start-pose final goal degrees: {previous_goal_physical_degrees}")
                print(f"start-pose final present degrees: {actual_physical_degrees}")
                print(f"start-pose final error degrees: {errors_degrees}")
                raise RuntimeError("start-pose aborted: physical Present_Position did not converge within 0.5 degrees")
            delta_goal_degrees = {
                joint_name: max(
                    -START_POSE_TRIM_MAX_ADJUSTMENT_DEGREES,
                    min(START_POSE_TRIM_MAX_ADJUSTMENT_DEGREES, START_POSE_TRIM_GAIN * errors_degrees[joint_name]),
                )
                for joint_name in INFERENCE_START_ARM_JOINT_NAMES
            }
            previous_goal_physical_degrees = {
                joint_name: previous_goal_physical_degrees[joint_name] + delta_goal_degrees[joint_name]
                for joint_name in INFERENCE_START_ARM_JOINT_NAMES
            }
            print(f"start-pose trim iteration {trim_iteration + 1}/{START_POSE_TRIM_MAX_ITERATIONS}")
            print(f"start-pose target degrees: {target_physical_degrees}")
            print(f"start-pose present degrees: {actual_physical_degrees}")
            print(f"start-pose error degrees: {errors_degrees}")
            print(f"start-pose goal degrees: {previous_goal_physical_degrees}")
            self._write_arm_goal_physical_degrees(previous_goal_physical_degrees)
            sleep_fn(START_POSE_TRIM_WAIT_SECONDS)
        raise AssertionError("unreachable trim-loop exit")


def analyze_step821_corrections(log_path: Path) -> dict[str, dict[str, float]] | None:
    """Offline-only conversion of q_corr_applied_rad to absolute raw-step magnitudes."""
    if not log_path.is_file(): return None
    import numpy as np
    data = np.load(log_path)
    if "q_corr_applied_rad" not in data.files or data["q_corr_applied_rad"].ndim != 2: return None
    corrections_raw_step = np.abs(data["q_corr_applied_rad"]) * (4096.0 / (2.0 * np.pi))
    log_joint_names=("shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll")
    report={}
    for joint_name in THREE_JOINT_SWEEP_NAMES:
        values=corrections_raw_step[:,log_joint_names.index(joint_name)]
        report[joint_name]={"median":float(np.median(values)),"p90":float(np.percentile(values,90)),"p95":float(np.percentile(values,95)),"max":float(np.max(values)),"lt2":float(np.mean(values<2)),"lt4":float(np.mean(values<4)),"ge4":float(np.mean(values>=4))}
    return report


def print_command_plan(report: PreflightReport, audit: WriteAudit) -> None:
    baseline = report.baseline
    print("motor:\nwrist_flex")
    print(f"baseline raw Goal:\n{baseline.wrist_goal_raw_step}")
    print(f"baseline raw Present:\n{baseline.wrist_present_raw_step}")
    print(f"baseline G0-P0 raw steps:\n{baseline.wrist_goal_raw_step - baseline.wrist_present_raw_step}")
    print("planned sequence:\n+1, 0, -1, 0, +2, 0, -2, 0")
    print("normalize:\nFalse")
    print(f"gripper writes:\n{audit.goal_write_count_gripper}")
    print("other joint writes:\n0")
    print(f"torque writes:\n{audit.torque_write_count}")
    print(f"PID writes:\n{audit.pid_write_count}")
    print(
        "proposed wrist G0-P0 preflight limit (review required):\n"
        f"{PROPOSED_MAX_WRIST_GOAL_PRESENT_GAP_RAW_STEP} raw steps"
    )


def run_micro_test_mockable(bus: BusProtocol, report: PreflightReport, audit: WriteAudit) -> None:
    """Physical sequence after all CLI gates. No recovery write is attempted on failure."""
    writer = WristOnlyGoalWriter(bus, report.baseline, audit)
    for delta_raw_step in MICRO_SEQUENCE_RAW_STEP:
        writer.write_wrist_raw_delta(delta_raw_step)
    audit.assert_forbidden_counters_zero()


def validate_armed_execution_gate(report: PreflightReport, approved_limit_raw_step: int | None) -> None:
    """Reject unapproved or excessive G0-P0 only in the physically armed phase."""
    if approved_limit_raw_step != PROPOSED_MAX_WRIST_GOAL_PRESENT_GAP_RAW_STEP:
        raise RuntimeError("physical phase refused: proposed G0-P0 limit lacks explicit approval")
    wrist_goal_present_gap_raw_step = (
        report.baseline.wrist_goal_raw_step - report.baseline.wrist_present_raw_step
    )
    if abs(wrist_goal_present_gap_raw_step) > approved_limit_raw_step:
        raise RuntimeError(
            f"physical phase refused: wrist G0-P0={wrist_goal_present_gap_raw_step} exceeds "
            f"approved limit {approved_limit_raw_step} raw steps"
        )


def _load_calibration(calibration_path: Path) -> dict[str, Any]:
    from lerobot.motors import MotorCalibration

    records = json.loads(calibration_path.read_text())
    return {motor_name: MotorCalibration(**record) for motor_name, record in records.items()}


def create_direct_bus(port: str, calibration_path: Path) -> BusProtocol:
    """Construct communication only; callers may connect it but must never configure it."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    calibration = _load_calibration(calibration_path)
    missing = set(ALL_MOTOR_NAMES) - set(calibration)
    if missing:
        raise RuntimeError(f"calibration file missing motors: {sorted(missing)}")
    return FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="serial port supplied by the supervised deployment configuration")
    parser.add_argument("--calibration-json", required=True, type=Path, help="actual project motor calibration JSON")
    parser.add_argument("--execute-micro-test", action="store_true", help="enable the physically writing phase")
    parser.add_argument("--arm-token", default="", help=f"required literal token: {ARM_TOKEN}")
    parser.add_argument(
        "--initialize-known-arm-state",
        action="store_true",
        help="enable the separate five-arm direct-bus initialization phase",
    )
    parser.add_argument(
        "--init-arm-token",
        default="",
        help=f"required literal token: {INITIALIZATION_ARM_TOKEN}",
    )
    parser.add_argument("--execute-wrist-raw-sweep", action="store_true", help="enable the STEP8.24F wrist-only sweep")
    parser.add_argument("--sweep-arm-token", default="", help=f"required literal token: {SWEEP_ARM_TOKEN}")
    parser.add_argument("--execute-three-joint-raw-sweep", action="store_true", help="enable the STEP8.24G three-joint sweep")
    parser.add_argument("--three-joint-sweep-token", default="", help=f"required literal token: {THREE_JOINT_SWEEP_ARM_TOKEN}")
    parser.add_argument("--execute-visible-joint-response", action="store_true", help="enable the STEP8.25 visible response test")
    parser.add_argument("--visible-response-token", default="", help=f"required literal token: {VISIBLE_RESPONSE_ARM_TOKEN}")
    parser.add_argument(
        "--approve-g0-p0-limit-raw-step",
        type=int,
        default=None,
        help="must equal the currently proposed reviewer-approved G0-P0 limit",
    )
    parser.add_argument("--confirm-local", action="store_true", help="require a final typed local confirmation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sum((args.execute_micro_test, args.initialize_known_arm_state, args.execute_wrist_raw_sweep, args.execute_three_joint_raw_sweep, args.execute_visible_joint_response)) > 1:
        print("ABORT: physical modes are mutually exclusive", file=sys.stderr)
        return 2
    # The direct bus connection does not call any robot lifecycle/configure method.
    bus = create_direct_bus(args.port, args.calibration_json)
    audit = WriteAudit()
    try:
        bus.connect(handshake=True)
        if args.execute_visible_joint_response:
            if args.visible_response_token != VISIBLE_RESPONSE_ARM_TOKEN: raise RuntimeError("visible test refused: invalid token")
            if not args.confirm_local: raise RuntimeError("visible test refused: --confirm-local is required")
            if input("Type EXECUTE_VISIBLE_JOINT_RESPONSE to continue: ").strip() != "EXECUTE_VISIBLE_JOINT_RESPONSE": raise RuntimeError("visible test refused: local confirmation did not match")
            InferenceStartPoseMove(bus, audit).move_and_settle()
            # VisibleJointResponseTest captures G0/P0 only here, after start-pose settling.
            VisibleJointResponseTest(bus, audit).run()
            print("STEP8.25 COMPLETE: visible joint response test only.")
            return 0
        if args.execute_three_joint_raw_sweep:
            if args.three_joint_sweep_token != THREE_JOINT_SWEEP_ARM_TOKEN: raise RuntimeError("three-joint sweep refused: invalid token")
            if not args.confirm_local: raise RuntimeError("three-joint sweep refused: --confirm-local is required")
            if input("Type EXECUTE_THREE_JOINT_RAW_SWEEP to continue: ").strip() != "EXECUTE_THREE_JOINT_RAW_SWEEP": raise RuntimeError("three-joint sweep refused: local confirmation did not match")
            ThreeJointRawSweep(bus, audit).run()
            print("STEP8.24G COMPLETE: three-joint raw sweeps only.")
            return 0
        if args.execute_wrist_raw_sweep:
            if args.sweep_arm_token != SWEEP_ARM_TOKEN:
                raise RuntimeError("sweep refused: invalid --sweep-arm-token")
            if not args.confirm_local:
                raise RuntimeError("sweep refused: --confirm-local is required")
            if input("Type EXECUTE_WRIST_RAW_SWEEP to continue: ").strip() != "EXECUTE_WRIST_RAW_SWEEP":
                raise RuntimeError("sweep refused: local confirmation did not match")
            TrueRawWristSweep(bus, audit).run()
            print("STEP8.24F COMPLETE: wrist raw sweep only; no micro-test was run.")
            return 0
        if args.initialize_known_arm_state:
            if args.init_arm_token != INITIALIZATION_ARM_TOKEN:
                raise RuntimeError("initialization refused: invalid --init-arm-token")
            if not args.confirm_local:
                raise RuntimeError("initialization refused: --confirm-local is required")
            if input("Type INITIALIZE_FIVE_ARM_MOTORS to continue: ").strip() != "INITIALIZE_FIVE_ARM_MOTORS":
                raise RuntimeError("initialization refused: local confirmation did not match")
            ArmKnownStateInitializer(bus, audit).initialize_known_arm_state()
            print("INITIALIZATION COMPLETE: five arm motors remain torque ON; no micro-test was run.")
            return 0
        report = preflight_read_only(bus)
        print_command_plan(report, audit)
        if not args.execute_micro_test:
            print("PHASE A COMPLETE: read-only preflight; no motor writes performed.")
            return 0
        if args.arm_token != ARM_TOKEN:
            raise RuntimeError("physical phase refused: invalid --arm-token")
        validate_armed_execution_gate(report, args.approve_g0_p0_limit_raw_step)
        if not args.confirm_local:
            raise RuntimeError("physical phase refused: --confirm-local is required")
        if input("Type WRIST_ONLY_EXECUTE to continue: ").strip() != "WRIST_ONLY_EXECUTE":
            raise RuntimeError("physical phase refused: local confirmation did not match")
        run_micro_test_mockable(bus, report, audit)
        print("PHASE B COMPLETE: wrist-only sequence finished.")
        return 0
    except Exception as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    finally:
        # Explicitly preserve torque state. No Goal=Present, recovery, or full-arm write.
        if getattr(bus, "is_connected", False):
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    raise SystemExit(main())
