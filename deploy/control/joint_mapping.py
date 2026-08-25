"""Pure conversions between URDF, encoder-degree, normalized, and raw domains."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class JointMappingError(ValueError):
    """Raised when a command crosses a motor domain with invalid data."""


class JointMapping:
    """Immutable mapping loaded from the canonical JSON records."""

    def __init__(self, mapping_path: Path | str, calibration_path: Path | str):
        mapping = json.loads(Path(mapping_path).read_text())
        calibration = json.loads(Path(calibration_path).read_text())
        self.joint_order = tuple(mapping["arm_joint_order"])
        joints = mapping["joints"]
        self.signs = np.asarray([joints[name]["sign"] for name in self.joint_order], dtype=np.float64)
        self.encoder_zero_deg = np.asarray(
            [joints[name]["encoder_zero_deg"] for name in self.joint_order], dtype=np.float64
        )
        self.calibration = calibration
        self.gripper = mapping["gripper"]
        if len(self.joint_order) != 5 or self.signs.shape != (5,) or self.encoder_zero_deg.shape != (5,):
            raise JointMappingError("canonical arm mapping must have five joints")
        if not np.isin(self.signs, (-1.0, 1.0)).all() or not np.isfinite(self.encoder_zero_deg).all():
            raise JointMappingError("invalid sign or encoder-zero mapping")
        normalized_open = float(self.gripper["normalized_width_open"])
        normalized_closed = float(self.gripper["normalized_width_closed"])
        endpoints = np.asarray(
            [self.gripper["open_raw"], self.gripper["closed_raw"]], dtype=np.float64
        )
        if (normalized_open, normalized_closed) != (1.0, 0.0):
            raise JointMappingError("gripper mapping must use G=1 open and G=0 closed")
        if (
            not np.isfinite(endpoints).all()
            or not np.equal(endpoints, np.rint(endpoints)).all()
            or endpoints[0] == endpoints[1]
        ):
            raise JointMappingError("gripper raw endpoints must be finite, distinct integers")
        self.gripper_open_raw = int(endpoints[0])
        self.gripper_closed_raw = int(endpoints[1])
        self.dataset_open = float(self.gripper.get("dataset_open", 0.441305))
        self.dataset_closed = float(self.gripper.get("dataset_closed", 0.0))
        if not (np.isfinite(self.dataset_open) and np.isfinite(self.dataset_closed) and self.dataset_open > self.dataset_closed):
            raise JointMappingError("dataset gripper endpoints must be finite with dataset_open > dataset_closed")

    def _arm_vector(self, values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (5,) or not np.isfinite(result).all():
            raise JointMappingError(f"{name} must contain five finite arm values")
        return result

    def validate_joint_order(self, names: Sequence[str]) -> None:
        if tuple(names) != self.joint_order:
            raise JointMappingError(f"joint order must be {self.joint_order}")

    def raw_degrees_to_urdf_degrees(self, raw_degrees: Sequence[float]) -> np.ndarray:
        raw = self._arm_vector(raw_degrees, "raw_degrees")
        return (raw - self.encoder_zero_deg) / self.signs

    def urdf_degrees_to_raw_degrees(self, urdf_degrees: Sequence[float]) -> np.ndarray:
        urdf = self._arm_vector(urdf_degrees, "urdf_degrees")
        return urdf * self.signs + self.encoder_zero_deg

    def raw_dict_to_urdf_degrees(self, raw_by_name: Mapping[str, float]) -> np.ndarray:
        if any(name not in raw_by_name for name in self.joint_order):
            raise JointMappingError("raw joint mapping is incomplete")
        return self.raw_degrees_to_urdf_degrees([raw_by_name[name] for name in self.joint_order])

    def arm_degrees_to_raw_ticks(self, raw_degrees: Sequence[float], resolution: int = 4096) -> dict[str, int]:
        """Match LeRobot's degree unnormalization without using its private API."""

        values = self._arm_vector(raw_degrees, "raw_degrees")
        if resolution <= 1:
            raise JointMappingError("resolution must be greater than one")
        result: dict[str, int] = {}
        for name, value in zip(self.joint_order, values, strict=True):
            record = self.calibration[name]
            midpoint = (float(record["range_min"]) + float(record["range_max"])) / 2.0
            tick = int(value * (resolution - 1) / 360.0 + midpoint)
            if not 0 <= tick < resolution:
                raise JointMappingError(f"{name} raw tick is outside [0,{resolution - 1}]")
            result[name] = tick
        return result

    def physical_open_fraction_from_raw(self, raw_position: float) -> float:
        raw = float(raw_position)
        if not np.isfinite(raw):
            raise JointMappingError("gripper raw position must be finite")
        span = self.gripper_open_raw - self.gripper_closed_raw
        return float(np.clip((raw - self.gripper_closed_raw) / span, 0.0, 1.0))

    def dataset_gripper_from_raw(self, raw_position: float) -> float:
        u_open = self.physical_open_fraction_from_raw(raw_position)
        return float(self.dataset_closed + u_open * (self.dataset_open - self.dataset_closed))

    def raw_from_dataset_gripper(self, dataset_gripper: float) -> int:
        val = float(dataset_gripper)
        if not np.isfinite(val):
            raise JointMappingError("dataset gripper value must be finite")
        u_open = float(np.clip((val - self.dataset_closed) / (self.dataset_open - self.dataset_closed), 0.0, 1.0))
        span = self.gripper_open_raw - self.gripper_closed_raw
        return int(round(self.gripper_closed_raw + u_open * span))

    def gripper_width_from_raw(self, raw_position: float) -> float:
        """Map motor raw ticks to dataset-scale gripper coordinate G_DATASET."""
        return self.dataset_gripper_from_raw(raw_position)

    def gripper_raw_from_width(self, width: float) -> int:
        """Map dataset-scale gripper coordinate G_DATASET back to motor raw ticks."""
        return self.raw_from_dataset_gripper(width)


def degrees_to_radians(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise JointMappingError("degree values must be finite")
    return np.deg2rad(values)


def radians_to_degrees(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise JointMappingError("radian values must be finite")
    return np.rad2deg(values)


__all__ = ["JointMapping", "JointMappingError", "degrees_to_radians", "radians_to_degrees"]
