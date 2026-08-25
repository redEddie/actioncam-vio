#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_DIR))
from incremental_action_contract import (  # noqa: E402
    incremental_actions,
    integrate_actions,
    smooth_recovery_states,
    wrapped_difference,
)


class IncrementalActionContractTest(unittest.TestCase):
    def test_sequential_step_to_step_reconstruction(self) -> None:
        states = np.asarray(
            [
                [0.0, 0.0, 0.0, 3.13, -3.13, 0.1],
                [0.1, 0.0, 0.0, -3.13, 3.13, 0.2],
                [0.3, 0.1, 0.0, -3.00, 3.00, 0.4],
                [0.6, 0.1, 0.2, -2.80, 2.80, 0.7],
            ]
        )
        actions = incremental_actions(states)
        reconstructed = integrate_actions(states[0], actions)
        error = reconstructed - states
        error[:, 3:5] = wrapped_difference(reconstructed[:, 3:5], states[:, 3:5])
        np.testing.assert_allclose(error, 0.0, atol=1e-12)

    def test_negative_anchor_relative_interpretation_is_detected(self) -> None:
        # Non-constant increments make S0+A[k] visibly different from the
        # mandatory S[k]+A[k] interpretation after the first step.
        states = np.asarray(
            [
                [1.0, 2.0, 3.0, 0.1, -0.2, 0.3],
                [1.1, 2.0, 3.0, 0.2, -0.2, 0.4],
                [1.4, 2.2, 3.0, 0.4, -0.1, 0.6],
                [2.0, 2.3, 3.5, 0.7, 0.1, 0.9],
            ]
        )
        actions = incremental_actions(states)
        correct = integrate_actions(states[0], actions)[1:]
        forbidden = states[0] + actions
        self.assertFalse(np.allclose(forbidden, correct))
        self.assertGreater(float(np.max(np.abs(forbidden[1:] - correct[1:]))), 0.1)

    def test_roll_pitch_difference_wraps(self) -> None:
        states = np.zeros((2, 6), dtype=np.float64)
        states[0, 3:5] = [np.deg2rad(179.0), np.deg2rad(-179.0)]
        states[1, 3:5] = [np.deg2rad(-179.0), np.deg2rad(179.0)]
        actions = incremental_actions(states)
        np.testing.assert_allclose(np.rad2deg(actions[0, 3:5]), [2.0, -2.0], atol=1e-10)

    def test_smooth_recovery_recomputes_incremental_actions(self) -> None:
        clean = np.zeros((16, 6), dtype=np.float64)
        clean[:, 0] = np.arange(16) * 0.01
        clean[:, 3] = np.arange(16) * 0.005
        error = np.asarray([0.015, -0.010, 0.008, 0.03, -0.02, 0.04])
        recovered = smooth_recovery_states(clean, error, recovery_steps=4)
        actions = incremental_actions(recovered)
        reconstructed = integrate_actions(recovered[0], actions)
        np.testing.assert_allclose(reconstructed, recovered, atol=1e-12)
        np.testing.assert_allclose(recovered[4:], clean[4:], atol=1e-12)
        # Smoothstep spreads correction across several actions instead of
        # placing the complete error in A0.
        one_step_correction = clean[1] - recovered[0]
        self.assertFalse(np.allclose(actions[0], one_step_correction))
        correction_component = actions[:, :3] - incremental_actions(clean)[:, :3]
        self.assertGreater(np.linalg.norm(correction_component[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
