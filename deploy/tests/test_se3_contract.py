"""Unit tests for the se3_relwp_v1 deployment adapter."""
from pathlib import Path
import sys

DEPLOY = Path(__file__).resolve().parents[1]
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inference.se3_contract import (
    StateHistory,
    chunk10_to_incremental6,
    relative_state10,
    rot6d_to_matrix,
    transform_to_10d,
    validate_se3_action_chunk,
)
from trajectory.incremental import reconstruct_incremental_states


def _pose(x, y, z, yaw, pitch, roll):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def _yawfree(T, grip):
    yaw, pitch, roll = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX")
    return np.array([*T[:3, 3], roll, pitch, np.clip(grip, 0.0, 1.0)])


def test_chunk_roundtrip_reproduces_se3_waypoints():
    rng = np.random.default_rng(0)
    T_obs = _pose(0.35, -0.02, 0.06, 0.4, -0.05, -1.5)
    grips = np.clip(0.4 + 0.05 * rng.standard_normal(15), 0.0, 1.0)
    absolute, chunk = [], []
    for k in range(15):
        Tk = _pose(
            0.35 + 0.01 * (k + 1), -0.02 + 0.005 * k, 0.06 + 0.002 * k,
            0.4 + 0.02 * k, -0.05 + 0.01 * k, -1.5 + 0.03 * k,
        )
        absolute.append(Tk)
        chunk.append(transform_to_10d(np.linalg.solve(T_obs, Tk), grips[k]))
    s_obs6 = _yawfree(T_obs, 0.41)
    actions6 = chunk10_to_incremental6(T_obs, s_obs6, np.asarray(chunk))
    states = reconstruct_incremental_states(s_obs6, actions6)
    expected = np.stack([_yawfree(T, g) for T, g in zip(absolute, grips)])
    np.testing.assert_allclose(states[1:], expected, atol=1e-9)


def test_roll_unwrap_across_pi():
    T_obs = _pose(0.3, 0.0, 0.05, 0.0, 0.0, np.pi - 0.05)
    Tk = _pose(0.3, 0.0, 0.05, 0.0, 0.0, -np.pi + 0.05)  # +0.1 rad across the wrap
    chunk = np.tile(transform_to_10d(np.linalg.solve(T_obs, Tk), 0.5), (15, 1))
    s_obs6 = _yawfree(T_obs, 0.5)
    actions6 = chunk10_to_incremental6(T_obs, s_obs6, chunk)
    assert abs(actions6[0, 3] - 0.1) < 1e-9  # continuous, no 2-pi jump
    assert np.allclose(actions6[1:, 3], 0.0, atol=1e-9)


def test_relative_state_identity_and_lag():
    T = _pose(0.3, 0.1, 0.05, 0.2, -0.1, -1.4)
    state = relative_state10(T, T, 0.44)
    np.testing.assert_allclose(state[:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(state[3:9], [1, 0, 0, 0, 1, 0], atol=1e-12)
    assert state[9] == pytest.approx(0.44)
    np.testing.assert_allclose(rot6d_to_matrix(state[3:9]), np.eye(3), atol=1e-12)


def test_state_history_selects_nearest_lagged_sample():
    history = StateHistory()
    for i, t in enumerate([0.0, 0.033, 0.066, 0.100, 0.133]):
        history.push(t, _pose(0.3 + 0.01 * i, 0, 0.05, 0, 0, -1.5), 0.4 + 0.01 * i)
    transform, grip, _ = history.lagged(0.133, lag_s=0.1)
    assert grip == pytest.approx(0.41)  # sample at t=0.033 is closest to 0.033
    assert transform[0, 3] == pytest.approx(0.31)


def test_state_history_interpolation_alignment():
    history = StateHistory()
    for i, t in enumerate([0.0, 0.1]):
        T = _pose(0.30 + 0.02 * i, 0.0, 0.05, 0.0, 0.0, -1.5 + 0.2 * i)
        history.push(t, T, 0.4 + 0.2 * i, _yawfree(T, 0.4 + 0.2 * i))
    transform, grip, state6 = history.interpolate(0.05)  # midpoint
    assert transform[0, 3] == pytest.approx(0.31)
    assert grip == pytest.approx(0.5)
    assert state6[3] == pytest.approx(-1.4)      # roll lerp, wrap-aware
    assert state6[5] == pytest.approx(0.5)
    # clamped outside the buffer
    t0, g0, _ = history.interpolate(-1.0)
    assert t0[0, 3] == pytest.approx(0.30) and g0 == pytest.approx(0.4)


def test_validate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        validate_se3_action_chunk(np.zeros((15, 6)))
    with pytest.raises(ValueError):
        validate_se3_action_chunk(np.full((15, 10), np.nan))
    assert validate_se3_action_chunk(np.zeros((1, 15, 10))).shape == (15, 10)
