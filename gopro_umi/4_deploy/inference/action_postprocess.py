"""Validate the remote policy's already-postprocessed physical action chunk."""

from __future__ import annotations

from typing import Any

import numpy as np


WIRE_BATCH_SHAPE = (1, 15, 6)
LOCAL_ACTION_SHAPE = (15, 6)


def validate_postprocessed_action_chunk(value: Any) -> np.ndarray:
    """Return an owned ``float64 [15,6]`` copy without renormalizing it."""

    result = np.asarray(value)
    if result.shape == WIRE_BATCH_SHAPE:
        result = result[0]
    elif result.shape != LOCAL_ACTION_SHAPE:
        raise ValueError(
            f"postprocessed action must have shape {WIRE_BATCH_SHAPE} or {LOCAL_ACTION_SHAPE}, got {result.shape}"
        )
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError("postprocessed action must be numeric")
    result = np.asarray(result, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("postprocessed action contains NaN or Inf")
    return result.copy()


__all__ = ["LOCAL_ACTION_SHAPE", "WIRE_BATCH_SHAPE", "validate_postprocessed_action_chunk"]
