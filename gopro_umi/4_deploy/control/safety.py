"""Pure safety gates used before inverse kinematics and motor commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


class SafetyError(RuntimeError):
    """A target failed a write-blocking canonical safety gate."""


class TargetStatus(str, Enum):
    BEFORE_START = "BEFORE_START"
    VALID = "VALID"
    EXPIRED = "EXPIRED"
    FAULT = "FAULT"


def finite_vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise SafetyError(f"{name} must contain {size} finite values")
    return result


def finite_trajectory(values: Sequence[Sequence[float]], width: int = 6) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise SafetyError(f"trajectory must be finite and have shape [N,{width}]")
    return result


def require_ik_success(q_radians: Sequence[float], converged: bool) -> np.ndarray:
    q = finite_vector(q_radians, 5, "IK result")
    if not converged:
        raise SafetyError("IK did not converge")
    return q


def require_joint_limits(
    q_radians: Sequence[float], joint_order: Sequence[str], limits: Mapping[str, tuple[float, float]]
) -> np.ndarray:
    q = finite_vector(q_radians, 5, "joint target")
    if tuple(joint_order) != tuple(limits):
        raise SafetyError("joint limit order does not match the command order")
    violations = [
        name
        for name, value in zip(joint_order, q, strict=True)
        if value < float(limits[name][0]) or value > float(limits[name][1])
    ]
    if violations:
        raise SafetyError(f"joint target outside hard limits: {', '.join(violations)}")
    return q


def require_raw_goal_ticks(goals: Mapping[str, int | float], names: Sequence[str], minimum=0, maximum=4095) -> dict[str, int]:
    if set(goals) != set(names):
        raise SafetyError("raw goal mapping does not match the required motor set")
    checked: dict[str, int] = {}
    for name in names:
        value = goals[name]
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
            raise SafetyError(f"{name} raw goal must be a finite integer")
        integer = int(value)
        if not minimum <= integer <= maximum:
            raise SafetyError(f"{name} raw goal outside [{minimum},{maximum}]")
        checked[name] = integer
    return checked


def target_status(now: float, first_time: float, last_time: float) -> TargetStatus:
    if not np.isfinite([now, first_time, last_time]).all() or last_time < first_time:
        return TargetStatus.FAULT
    if now < first_time - 1e-9:
        return TargetStatus.BEFORE_START
    if now > last_time + 1e-9:
        return TargetStatus.EXPIRED
    return TargetStatus.VALID


def communication_fault(exc: Exception) -> SafetyError:
    return SafetyError(f"motor communication failed: {type(exc).__name__}")


__all__ = [
    "SafetyError",
    "TargetStatus",
    "communication_fault",
    "finite_trajectory",
    "finite_vector",
    "require_ik_success",
    "require_joint_limits",
    "require_raw_goal_ticks",
    "target_status",
]
