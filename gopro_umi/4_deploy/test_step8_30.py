import unittest
import numpy as np
from step8_30_frozen_feedforward import Metrics, calculate_metrics

class TestStep830(unittest.TestCase):
    def test_calculate_metrics(self):
        t = np.linspace(0, 10.0, 300)
        tick_logs = {
            "t_since_motion_start": t,
            "p_actual_x": np.zeros(300),
            "p_actual_y": np.zeros(300),
            "p_actual_z": np.linspace(0, 0.005, 300),
            "p_target_x": np.zeros(300),
            "p_target_y": np.zeros(300),
            "p_target_z": np.linspace(0, 0.005, 300),
            "joint_tracking_error": np.zeros((300, 5)),
            "q_corr_applied_deg": np.zeros((300, 5)),
            "q_ff_phase_deg": np.zeros((300, 5)),
        }
        tick_logs["joint_tracking_error"][:, 2] = 1.0 # 1 deg error
        tick_logs["q_ff_phase_deg"][:, 2] = 1.0
        p0 = np.array([0, 0, 0])
        m = calculate_metrics(tick_logs, p0, 2.0)
        self.assertFalse(m.hunting_or_oscillation)
        self.assertEqual(m.steady_median_q_error_deg[2], 1.0)
        self.assertEqual(m.max_abs_q_ff_phase[2], 1.0)

if __name__ == '__main__':
    unittest.main()
