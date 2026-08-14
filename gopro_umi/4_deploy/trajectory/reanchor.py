"""Latency-aware re-anchoring against an explicitly supplied arrival state."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .incremental import reconstruct_incremental_states, validate_incremental_actions


def reanchor_trajectory(
    actions: Sequence[Sequence[float]],
    s_obs: Sequence[float],
    s_actual_arrival: Sequence[float],
    t_obs: float,
    t_arr: float,
    dt: float = 0.1,
) -> dict:
    """Discard elapsed model time and offset future states to physical arrival state."""

    action_array = validate_incremental_actions(actions)
    observed = np.asarray(s_obs, dtype=np.float64)
    actual = np.asarray(s_actual_arrival, dtype=np.float64)
    if observed.shape != (6,) or actual.shape != (6,):
        raise ValueError("s_obs and s_actual_arrival must each have shape (6,)")
    if not np.isfinite(observed).all() or not np.isfinite(actual).all():
        raise ValueError("reanchor states must be finite")
    if not np.isfinite(t_obs) or not np.isfinite(t_arr) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestamps and dt must be finite, with dt > 0")

    tau = float(t_arr - t_obs)
    if tau < 0.0:
        raise ValueError("arrival timestamp precedes observation timestamp")
    count = action_array.shape[0]
    horizon = count * dt
    if tau >= horizon:
        return {"status": "EXPIRED", "tau": tau, "horizon": horizon}

    raw_states = reconstruct_incremental_states(observed, action_array)
    model_times = np.arange(count + 1, dtype=np.float64) * dt
    interval = min(int(np.floor(tau / dt)), count - 1)
    alpha = float(np.clip((tau - model_times[interval]) / dt, 0.0, 1.0))
    predicted_arrival = (1.0 - alpha) * raw_states[interval] + alpha * raw_states[interval + 1]
    offset = actual - predicted_arrival

    future_mask = model_times > tau + 1e-9
    future_model_times = model_times[future_mask]
    corrected = raw_states[future_mask] + offset
    corrected[:, 5] = np.clip(corrected[:, 5], 0.0, 1.0)
    first_future_index = int(np.flatnonzero(future_mask)[0])
    absolute_times = np.concatenate(([t_arr], t_obs + future_model_times))
    states = np.vstack((actual, corrected))
    states[:, 5] = np.clip(states[:, 5], 0.0, 1.0)
    return {
        "status": "VALID",
        "tau": tau,
        "horizon": horizon,
        "current_interval_index": interval,
        "alpha": alpha,
        "predicted_arrival_state": predicted_arrival,
        "actual_arrival_state": actual.copy(),
        "offset": offset,
        "model_times": np.concatenate(([tau], future_model_times)),
        "arrival_relative_times": absolute_times - t_arr,
        "absolute_times": absolute_times,
        "states": states,
        "raw_states": raw_states,
        "first_future_index": first_future_index,
        "future_model_times": future_model_times,
        "corrected_future_states": corrected,
        "P_raw_tau": predicted_arrival,
        "P_raw": raw_states,
    }


__all__ = ["reanchor_trajectory"]
