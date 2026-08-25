"""Immutable pairing of a fresh RGB frame and a physical FK robot state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from control.joint_mapping import JointMapping
from .camera import FrameSnapshot


def _readonly_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ActualStateSnapshot:
    timestamp: float
    raw_degrees: np.ndarray
    q_urdf_degrees: np.ndarray
    state: np.ndarray
    yaw_radians: float
    transform: np.ndarray | None = None  # full FK 4x4 (se3_relwp_v1 contract)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_degrees", _readonly_vector(self.raw_degrees, (5,), "raw_degrees"))
        object.__setattr__(self, "q_urdf_degrees", _readonly_vector(self.q_urdf_degrees, (5,), "q_urdf_degrees"))
        object.__setattr__(self, "state", _readonly_vector(self.state, (6,), "state"))
        if self.transform is not None:
            matrix = np.asarray(self.transform, dtype=np.float64).copy()
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("transform must be a finite 4x4 matrix")
            matrix.setflags(write=False)
            object.__setattr__(self, "transform", matrix)
        if not np.isfinite(self.timestamp) or not np.isfinite(self.yaw_radians):
            raise ValueError("state timestamps and yaw must be finite")


@dataclass(frozen=True)
class ObservationSnapshot:
    request_id: int
    t_obs: float
    t_frame: float
    t_q_obs: float
    frame_sequence: int
    frame_metadata: Mapping[str, Any]
    image_rgb: np.ndarray
    raw_degrees: np.ndarray
    q_obs_degrees: np.ndarray
    s_obs: np.ndarray
    yaw_radians: float
    payload_state: np.ndarray | None = None  # model-facing state; defaults to s_obs (incremental_v1)
    t_obs_transform: np.ndarray | None = None  # FK 4x4 at observation (se3_relwp_v1)

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb, dtype=np.uint8).copy()
        if image.shape != (256, 256, 3):
            raise ValueError("observation image must be RGB 256x256")
        image.setflags(write=False)
        object.__setattr__(self, "image_rgb", image)
        object.__setattr__(self, "raw_degrees", _readonly_vector(self.raw_degrees, (5,), "raw_degrees"))
        object.__setattr__(self, "q_obs_degrees", _readonly_vector(self.q_obs_degrees, (5,), "q_obs_degrees"))
        object.__setattr__(self, "s_obs", _readonly_vector(self.s_obs, (6,), "s_obs"))
        if self.payload_state is not None:
            payload = np.asarray(self.payload_state, dtype=np.float64).copy()
            if payload.ndim != 1 or not np.isfinite(payload).all():
                raise ValueError("payload_state must be a finite vector")
            payload.setflags(write=False)
            object.__setattr__(self, "payload_state", payload)
        if self.t_obs_transform is not None:
            matrix = np.asarray(self.t_obs_transform, dtype=np.float64).copy()
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("t_obs_transform must be a finite 4x4 matrix")
            matrix.setflags(write=False)
            object.__setattr__(self, "t_obs_transform", matrix)
        object.__setattr__(self, "frame_metadata", MappingProxyType(dict(self.frame_metadata)))
        if not isinstance(self.request_id, int) or self.request_id < 0:
            raise ValueError("request_id must be a non-negative integer")
        if not np.isfinite([self.t_obs, self.t_frame, self.t_q_obs, self.yaw_radians]).all():
            raise ValueError("observation metadata must be finite")


def state_from_motor_sample(
    positions: Mapping[str, float],
    timestamp: float,
    mapping: JointMapping,
    fk_solver: Any,
) -> ActualStateSnapshot:
    """Build actual XYZ/Roll/Pitch/gripper state; yaw remains separate metadata."""

    missing = [name for name in mapping.joint_order + ("gripper",) if name not in positions]
    if missing:
        raise ValueError(f"motor sample is missing: {', '.join(missing)}")
    raw_degrees = np.asarray([positions[name] for name in mapping.joint_order], dtype=np.float64)
    q_degrees = mapping.raw_degrees_to_urdf_degrees(raw_degrees)
    transform = np.asarray(fk_solver.forward_kinematics(np.deg2rad(q_degrees)), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("FK returned an invalid transform")
    yaw, pitch, roll = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYX", degrees=False)
    gripper = mapping.gripper_width_from_raw(float(positions["gripper"]))
    state = np.asarray([*transform[:3, 3], roll, pitch, gripper], dtype=np.float64)
    return ActualStateSnapshot(float(timestamp), raw_degrees, q_degrees, state, float(yaw), transform)


def build_observation(
    request_id: int,
    frame: FrameSnapshot,
    actual: ActualStateSnapshot,
    payload_state: np.ndarray | None = None,
) -> ObservationSnapshot:
    """Pair the latest explicit frame with the fresh state used for this request."""

    t_obs = max(float(frame.timestamp), float(actual.timestamp))
    metadata = {"sequence": frame.sequence, "source_shape": frame.source_shape}
    return ObservationSnapshot(
        request_id=request_id,
        t_obs=t_obs,
        t_frame=frame.timestamp,
        t_q_obs=actual.timestamp,
        frame_sequence=frame.sequence,
        frame_metadata=metadata,
        image_rgb=frame.image_rgb,
        raw_degrees=actual.raw_degrees,
        q_obs_degrees=actual.q_urdf_degrees,
        s_obs=actual.state,
        yaw_radians=actual.yaw_radians,
        payload_state=payload_state,
        t_obs_transform=actual.transform,
    )


def build_aligned_observation(
    request_id: int,
    frame: FrameSnapshot,
    actual: ActualStateSnapshot,
    aligned_t: float,
    aligned_state: np.ndarray,
    aligned_transform: np.ndarray | None,
    payload_state: np.ndarray | None = None,
) -> ObservationSnapshot:
    """UMI-style latency-matched observation: the state is interpolated at the back-dated image
    capture time ``aligned_t`` so image and state describe the same physical instant. ``t_obs``
    is the aligned (capture) time, which also makes the reanchor tau include camera latency."""

    metadata = {"sequence": frame.sequence, "source_shape": frame.source_shape, "alignment": "umi_v1"}
    aligned_state = np.asarray(aligned_state, dtype=np.float64)
    yaw = actual.yaw_radians
    if aligned_transform is not None:
        yaw = float(Rotation.from_matrix(np.asarray(aligned_transform)[:3, :3]).as_euler("ZYX")[0])
    return ObservationSnapshot(
        request_id=request_id,
        t_obs=float(aligned_t),
        t_frame=frame.timestamp,
        t_q_obs=actual.timestamp,
        frame_sequence=frame.sequence,
        frame_metadata=metadata,
        image_rgb=frame.image_rgb,
        raw_degrees=actual.raw_degrees,
        q_obs_degrees=actual.q_urdf_degrees,
        s_obs=aligned_state,
        yaw_radians=yaw,
        payload_state=payload_state,
        t_obs_transform=aligned_transform,
    )


__all__ = [
    "ActualStateSnapshot",
    "ObservationSnapshot",
    "build_aligned_observation",
    "build_observation",
    "state_from_motor_sample",
]
