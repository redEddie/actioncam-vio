"""Strict, side-effect-free loader for canonical deployment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


CONFIG_DIR = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Raised when canonical deployment configuration violates the contract."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ConfigError(f"{name} must contain {length} finite values")
    return result


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated immutable view over deployment, mapping, and start records."""

    root: Path
    deployment_path: Path
    physical_start_path: Path
    motor_mapping_path: Path
    deployment: Mapping[str, Any]
    physical_start: Mapping[str, Any]
    motor_mapping: Mapping[str, Any]
    joint_order: tuple[str, ...]
    signs: np.ndarray
    encoder_zero_deg: np.ndarray
    start_raw_deg: np.ndarray
    start_urdf_deg: np.ndarray
    static_support_bias_deg: np.ndarray
    gravity_reference_bias_deg: np.ndarray

    @property
    def action_period_s(self) -> float:
        return float(self.deployment["model"]["action_period_s"])

    @property
    def control_period_s(self) -> float:
        return 1.0 / float(self.deployment["controller"]["control_rate_hz"])

    @property
    def k_ext(self) -> float:
        return float(self.deployment["controller"]["k_ext"])

    @property
    def q_corr_clamp_deg(self) -> float:
        return float(self.deployment["controller"]["q_corr_clamp_deg"])

    @property
    def model_bundle(self) -> Path:
        return (self.deployment_path.parent / self.deployment["model"]["local_bundle"]).resolve()

    @property
    def motor_calibration(self) -> Path:
        return (self.deployment_path.parent / self.deployment["motor"]["calibration"]).resolve()

    @property
    def gravity(self) -> Mapping[str, Any]:
        return self.deployment["gravity_compensation"]

    @property
    def gravity_urdf(self) -> Path:
        return (self.deployment_path.parent / self.gravity["urdf"]).resolve()


def load_runtime_config(
    deployment_path: Path | str = CONFIG_DIR / "deployment.yaml",
    physical_start_path: Path | str = CONFIG_DIR / "physical_start.json",
    motor_mapping_path: Path | str = CONFIG_DIR / "motor_mapping.json",
) -> RuntimeConfig:
    """Load and cross-validate canonical records without device or network I/O."""

    deployment_path = Path(deployment_path).resolve()
    physical_start_path = Path(physical_start_path).resolve()
    motor_mapping_path = Path(motor_mapping_path).resolve()

    deployment = _mapping(yaml.safe_load(deployment_path.read_text()), "deployment")
    physical_start = _mapping(json.loads(physical_start_path.read_text()), "physical_start")
    motor_mapping = _mapping(json.loads(motor_mapping_path.read_text()), "motor_mapping")

    forbidden_remote_keys = {"password", "passwd", "sshpass"}
    remote = _mapping(deployment.get("remote"), "remote")
    present_forbidden = forbidden_remote_keys.intersection(str(key).lower() for key in remote)
    if present_forbidden:
        raise ConfigError("remote credentials must not be stored in deployment configuration")

    model = _mapping(deployment.get("model"), "model")
    controller = _mapping(deployment.get("controller"), "controller")
    gravity = _mapping(deployment.get("gravity_compensation"), "gravity_compensation")
    motor = _mapping(deployment.get("motor"), "motor")
    scheduler = _mapping(deployment.get("scheduler"), "scheduler")

    if float(model.get("action_rate_hz")) != 10.0:
        raise ConfigError("model action rate must be 10 Hz")
    if float(model.get("action_period_s")) != 0.1:
        raise ConfigError("model action period must be 0.1 s")
    if int(model.get("chunk_size")) != 15:
        raise ConfigError("model chunk size must be 15")
    if int(model.get("action_dimension")) != 6:
        raise ConfigError("model action dimension must be 6")
    if float(model.get("horizon_s")) != 1.5:
        raise ConfigError("model horizon must be 1.5 s")
    if not np.isclose(float(model["chunk_size"]) * float(model["action_period_s"]), float(model["horizon_s"])):
        raise ConfigError("model chunk, period, and horizon disagree")
    if float(controller.get("control_rate_hz")) != 30.0:
        raise ConfigError("control rate must be 30 Hz")
    if float(controller.get("k_ext")) != 0.5:
        raise ConfigError("canonical K_ext must be 0.5")
    _positive(controller.get("q_corr_clamp_deg"), "q_corr_clamp_deg")
    _positive(scheduler.get("request_stride_s"), "request_stride_s")
    _positive(scheduler.get("transition_overlap_s"), "transition_overlap_s")

    pid = _mapping(motor.get("pid"), "motor.pid")
    if (int(pid.get("p")), int(pid.get("i")), int(pid.get("d"))) != (64, 0, 32):
        raise ConfigError("canonical PID must be P64/I0/D32")

    joint_order = tuple(motor_mapping.get("arm_joint_order", ()))
    expected_order = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )
    if joint_order != expected_order:
        raise ConfigError("unexpected canonical joint order")
    if tuple(physical_start.get("joint_order", ())) != joint_order:
        raise ConfigError("physical start joint order disagrees with motor mapping")

    joints = _mapping(motor_mapping.get("joints"), "motor_mapping.joints")
    if tuple(joints) != joint_order:
        raise ConfigError("motor mapping joint entries are incomplete or out of order")
    signs = _finite_vector([joints[name]["sign"] for name in joint_order], 5, "signs")
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ConfigError("every sign must be +1 or -1")
    encoder_zero = _finite_vector(
        [joints[name]["encoder_zero_deg"] for name in joint_order], 5, "encoder_zero_deg"
    )
    gripper = _mapping(motor_mapping.get("gripper"), "motor_mapping.gripper")
    normalized_open = float(gripper.get("normalized_width_open"))
    normalized_closed = float(gripper.get("normalized_width_closed"))
    if not np.isfinite([normalized_open, normalized_closed]).all() or (
        normalized_open,
        normalized_closed,
    ) != (1.0, 0.0):
        raise ConfigError("canonical gripper semantics must be G=1 open and G=0 closed")
    endpoint_values = np.asarray(
        [gripper.get("open_raw"), gripper.get("closed_raw")], dtype=np.float64
    )
    if (
        not np.isfinite(endpoint_values).all()
        or not np.equal(endpoint_values, np.rint(endpoint_values)).all()
        or not ((0.0 <= endpoint_values) & (endpoint_values <= 4095.0)).all()
        or endpoint_values[0] == endpoint_values[1]
    ):
        raise ConfigError("gripper raw endpoints must be distinct integer ticks in [0,4095]")
    support_bias = _mapping(controller.get("support_bias"), "controller.support_bias")
    static_support_bias = _finite_vector(
        support_bias.get("static_deg"), 5, "controller.support_bias.static_deg"
    )
    gravity_reference_bias = _finite_vector(
        gravity.get("reference_bias_deg"), 5, "gravity_compensation.reference_bias_deg"
    )
    expected_static_support = np.array([0.264, 0.0, 0.0, 0.0, 0.352])
    expected_gravity_reference = np.array([0.0, 0.615, 2.286, 0.703, 0.0])
    if not np.array_equal(static_support_bias, expected_static_support):
        raise ConfigError("canonical static support bias does not match the frozen contract")
    if not np.array_equal(gravity_reference_bias, expected_gravity_reference):
        raise ConfigError("canonical gravity reference bias does not match the frozen contract")
    if not np.allclose(static_support_bias[[1, 2, 3]], 0.0, atol=0.0, rtol=0.0):
        raise ConfigError("static support is only allowed on shoulder_pan and wrist_roll")
    if not np.allclose(gravity_reference_bias[[0, 4]], 0.0, atol=0.0, rtol=0.0):
        raise ConfigError("gravity reference bias is only allowed on pitch joints")
    if gravity.get("enabled") is not True:
        raise ConfigError("canonical gravity compensation must be enabled")
    if gravity.get("reference_pose_source") != "physical_start.urdf_start_deg":
        raise ConfigError("gravity reference must use the physical-start URDF pose")
    if _positive(gravity.get("ratio_clip"), "gravity_compensation.ratio_clip") != 1.5:
        raise ConfigError("canonical gravity ratio clip must be 1.5")
    if (
        _positive(
            gravity.get("denominator_guard_nm"),
            "gravity_compensation.denominator_guard_nm",
        )
        != 0.0001
    ):
        raise ConfigError("canonical gravity denominator guard must be 1e-4 Nm")
    if _positive(gravity.get("gravity_m_s2"), "gravity_compensation.gravity_m_s2") != 9.81:
        raise ConfigError("canonical gravity acceleration must be 9.81 m/s^2")
    for link_key in ("gripper_link", "tool_reference_link", "stock_moving_jaw_link"):
        if not isinstance(gravity.get(link_key), str) or not gravity[link_key]:
            raise ConfigError(f"gravity_compensation.{link_key} must be a nonempty link name")
    camera_payload = _mapping(gravity.get("camera"), "gravity_compensation.camera")
    for key in ("mass_kg", "backward_m", "upward_m"):
        _positive(camera_payload.get(key), f"gravity_compensation.camera.{key}")
    if tuple(float(camera_payload[key]) for key in ("mass_kg", "backward_m", "upward_m")) != (
        0.240,
        0.030,
        0.055,
    ):
        raise ConfigError("canonical camera payload does not match the frozen contract")
    jaw_payload = _mapping(gravity.get("custom_jaw"), "gravity_compensation.custom_jaw")
    for key in ("assembly_mass_kg", "length_scale", "com_fraction_from_base"):
        _positive(jaw_payload.get(key), f"gravity_compensation.custom_jaw.{key}")
    if tuple(
        float(jaw_payload[key])
        for key in ("assembly_mass_kg", "length_scale", "com_fraction_from_base")
    ) != (0.280, 2.0, 1.0 / 3.0):
        raise ConfigError("canonical custom-jaw payload does not match the frozen contract")
    if float(jaw_payload["com_fraction_from_base"]) >= 1.0:
        raise ConfigError("custom jaw CoM fraction must be less than one")
    start_raw = _finite_vector(physical_start.get("raw_start_deg"), 5, "raw_start_deg")
    start_urdf = _finite_vector(physical_start.get("urdf_start_deg"), 5, "urdf_start_deg")
    if not np.allclose(start_urdf * signs + encoder_zero, start_raw, atol=1e-9, rtol=0.0):
        raise ConfigError("RAW start != URDF start * sign + encoder zero")

    root = deployment_path.parents[2]
    result = RuntimeConfig(
        root=root,
        deployment_path=deployment_path,
        physical_start_path=physical_start_path,
        motor_mapping_path=motor_mapping_path,
        deployment=deployment,
        physical_start=physical_start,
        motor_mapping=motor_mapping,
        joint_order=joint_order,
        signs=signs,
        encoder_zero_deg=encoder_zero,
        start_raw_deg=start_raw,
        start_urdf_deg=start_urdf,
        static_support_bias_deg=static_support_bias,
        gravity_reference_bias_deg=gravity_reference_bias,
    )
    if not result.model_bundle.is_dir():
        raise ConfigError(f"model bundle not found: {result.model_bundle}")
    if not result.motor_calibration.is_file():
        raise ConfigError(f"motor calibration not found: {result.motor_calibration}")
    if not result.gravity_urdf.is_file():
        raise ConfigError(f"gravity URDF not found: {result.gravity_urdf}")
    expected_gravity_sha = str(gravity.get("urdf_sha256", ""))
    actual_gravity_sha = hashlib.sha256(result.gravity_urdf.read_bytes()).hexdigest()
    if expected_gravity_sha != actual_gravity_sha:
        raise ConfigError("gravity URDF SHA256 does not match deployment configuration")
    return result


__all__ = ["ConfigError", "RuntimeConfig", "load_runtime_config"]
