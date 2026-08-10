import unittest

import numpy as np

import step8_26_fast_cartesian_gain_sweep as sweep


def synthetic_logs(steady_z_mm=4.0, clamp=False, flips=False):
    t = np.linspace(0.0, 10.0, 301)
    z_mm = np.where((t >= 1.0) & (t <= 4.0), steady_z_mm, 0.0)
    hold = (t >= 3.5) & (t <= 4.0)
    q_error = np.zeros((len(t), 5))
    if flips:
        q_error[hold, 0] = np.resize(np.array([1.0, -1.0]), np.sum(hold))
    q_corr = np.zeros((len(t), 5))
    if clamp:
        q_corr[hold] = sweep.CORRECTION_CLAMP_DEG
    return {
        "t_since_motion_start": t,
        "p_actual_x": np.zeros_like(t), "p_actual_y": np.zeros_like(t), "p_actual_z": z_mm / 1000.0,
        "p_target_x": np.zeros_like(t), "p_target_y": np.zeros_like(t),
        "p_target_z": np.where((t >= 1.0) & (t <= 4.0), 0.005, 0.0),
        "joint_tracking_error": q_error,
        "q_corr_applied_deg": q_corr,
    }


class TestStep826GainSweep(unittest.TestCase):
    def test_only_requested_gains_and_fixed_clamp(self):
        self.assertEqual(sweep.K_EXT_VALUES, (0.5, 0.8, 1.0))
        self.assertEqual(sweep.CORRECTION_CLAMP_DEG, 1.0)
        error, corr = sweep.external_correction(np.deg2rad([3.0]), np.deg2rad([0.0]), 1.0)
        self.assertAlmostEqual(np.rad2deg(error[0]), 3.0)
        self.assertAlmostEqual(np.rad2deg(corr[0]), 1.0)

    def test_reports_requested_metrics_and_material_thresholds(self):
        metrics = sweep.calculate_metrics(synthetic_logs(steady_z_mm=4.25), np.zeros(3))
        self.assertAlmostEqual(metrics.tracking_ratio_percent, 85.0)
        self.assertAlmostEqual(metrics.steady_delta_z_mm, 4.25)
        self.assertEqual(metrics.max_abs_q_error_deg.shape, (5,))
        self.assertFalse(metrics.hunting_or_oscillation)
        self.assertIn("+/-1.5", sweep.select_next_step(metrics))

    def test_hard_stop_detects_sustained_clamp_or_hunting(self):
        self.assertTrue(sweep.is_hard_stop(sweep.calculate_metrics(synthetic_logs(clamp=True), np.zeros(3))))
        self.assertTrue(sweep.is_hard_stop(sweep.calculate_metrics(synthetic_logs(flips=True), np.zeros(3))))

    def test_live_observer_stops_repeated_hold_sign_flips(self):
        observer = sweep.make_hard_stop_observer()
        stopped = False
        for tick in range(10):
            stopped = observer(2.0 + tick / 30.0, np.array([1.0 if tick % 2 else -1.0] * 5), np.zeros(5))
            if stopped:
                break
        self.assertTrue(stopped)

    def test_below_75_stops_gain_tuning(self):
        metrics = sweep.calculate_metrics(synthetic_logs(steady_z_mm=3.7), np.zeros(3))
        self.assertIn("stop K_ext tuning", sweep.select_next_step(metrics))


if __name__ == "__main__":
    unittest.main()
