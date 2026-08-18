#!/usr/bin/env python3
"""Shared 6D yaw-free, step-to-step incremental action primitives.

This module is deliberately independent of LeRobot and robot hardware.  Both
dataset generation and deployment import these functions so the action meaning
cannot silently diverge between training and runtime.
"""
from __future__ import annotations

import numpy as np


ACTION_DIM = 6
ACTION_DT_S = 0.100
ACTION_RATE_HZ = 10
CHUNK_SIZE = 15
ACTION_ORDER = ("dx", "dy", "dz", "droll", "dpitch", "dgripper")
STATE_ORDER = ("x", "y", "z", "roll", "pitch", "gripper")
ANGLE_INDICES = (3, 4)


def _states(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.shape[-1:] != (ACTION_DIM,):
        raise ValueError(f"{name} must end in dimension {ACTION_DIM}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def wrapped_difference(next_angle: np.ndarray, previous_angle: np.ndarray) -> np.ndarray:
    """Shortest signed angular displacement from previous to next, in radians."""
    return wrap_angle(np.asarray(next_angle) - np.asarray(previous_angle))


def incremental_actions(states: np.ndarray) -> np.ndarray:
    """Return A[k] = S[k+1] - S[k], wrapping roll/pitch differences."""
    value = _states(states, name="states")
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("states must have shape [N>=2, 6]")
    actions = np.diff(value, axis=0)
    actions[:, ANGLE_INDICES] = wrapped_difference(
        value[1:, ANGLE_INDICES], value[:-1, ANGLE_INDICES]
    )
    return actions


def integrate_actions(initial_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Sequentially reconstruct S[k+1] = S[k] + A[k].

    The return includes the initial state and therefore has shape [N+1, 6].
    This function intentionally does not expose an observation-anchor-relative
    reconstruction mode.
    """
    initial = _states(initial_state, name="initial_state")
    delta = _states(actions, name="actions")
    if initial.shape != (ACTION_DIM,) or delta.ndim != 2:
        raise ValueError("expected initial_state [6] and actions [N,6]")
    result = np.empty((len(delta) + 1, ACTION_DIM), dtype=np.float64)
    result[0] = initial
    for index, action in enumerate(delta):
        result[index + 1] = result[index] + action
        result[index + 1, ANGLE_INDICES] = wrap_angle(result[index + 1, ANGLE_INDICES])
    return result


def state_difference(next_state: np.ndarray, previous_state: np.ndarray) -> np.ndarray:
    """6D displacement with shortest-path roll/pitch components."""
    next_value = _states(next_state, name="next_state")
    previous_value = _states(previous_state, name="previous_state")
    result = next_value - previous_value
    result[..., ANGLE_INDICES] = wrapped_difference(
        next_value[..., ANGLE_INDICES], previous_value[..., ANGLE_INDICES]
    )
    return result


def add_displacement(state: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Add a 6D displacement to a state and wrap roll/pitch."""
    result = _states(state, name="state") + _states(displacement, name="displacement")
    result[..., ANGLE_INDICES] = wrap_angle(result[..., ANGLE_INDICES])
    return result


def interpolate_state(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Linear XYZ/gripper and shortest-path angular interpolation."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    left_value = _states(left, name="left")
    delta = state_difference(right, left_value)
    return add_displacement(left_value, float(fraction) * delta)


def smooth_recovery_states(
    clean_states: np.ndarray,
    initial_error: np.ndarray,
    recovery_steps: int,
) -> np.ndarray:
    """Apply a C1-smooth decaying empirical error and return S'[0..N].

    Error magnitude follows 1 - smoothstep(k / R), reaches exactly zero at R,
    and remains zero afterwards.  Actions must be recomputed from this returned
    trajectory with :func:`incremental_actions`.
    """
    clean = _states(clean_states, name="clean_states")
    error = _states(initial_error, name="initial_error")
    if clean.ndim != 2 or clean.shape[0] < 2 or error.shape != (ACTION_DIM,):
        raise ValueError("expected clean_states [N>=2,6] and initial_error [6]")
    if not 1 <= recovery_steps < len(clean):
        raise ValueError("recovery_steps must be in [1, len(clean_states)-1]")
    k = np.arange(len(clean), dtype=np.float64)
    u = np.clip(k / float(recovery_steps), 0.0, 1.0)
    decay = 1.0 - (3.0 * u**2 - 2.0 * u**3)
    result = clean + decay[:, None] * error
    result[:, ANGLE_INDICES] = wrap_angle(result[:, ANGLE_INDICES])
    return result
