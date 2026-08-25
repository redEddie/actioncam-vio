"""Step-to-step incremental SmolVLA trajectory reconstruction."""

from __future__ import annotations

from typing import Sequence

import numpy as np


ACTION_SHAPE = (15, 6)


def validate_incremental_actions(actions: Sequence[Sequence[float]], expected_shape=ACTION_SHAPE) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float64)
    if result.shape != tuple(expected_shape):
        raise ValueError(f"incremental actions must have shape {tuple(expected_shape)}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("incremental actions contain NaN or Inf")
    return result.copy()


def reconstruct_incremental_states(
    anchor_state: Sequence[float], actions: Sequence[Sequence[float]], expected_shape=ACTION_SHAPE
) -> np.ndarray:
    """Return anchor plus all cumulative states; output shape is ``[16,6]``."""

    anchor = np.asarray(anchor_state, dtype=np.float64)
    if anchor.shape != (6,) or not np.isfinite(anchor).all():
        raise ValueError("anchor_state must contain six finite values")
    action_array = validate_incremental_actions(actions, expected_shape)
    states = np.empty((action_array.shape[0] + 1, 6), dtype=np.float64)
    states[0] = anchor
    states[0, 5] = np.clip(states[0, 5], 0.0, 1.0)
    for index, delta in enumerate(action_array):
        states[index + 1] = states[index] + delta
        states[index + 1, 5] = np.clip(states[index + 1, 5], 0.0, 1.0)
    return states


__all__ = ["ACTION_SHAPE", "validate_incremental_actions", "reconstruct_incremental_states"]
