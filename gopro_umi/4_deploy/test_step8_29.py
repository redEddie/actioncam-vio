import unittest
import numpy as np
from step8_29_fine_kext_sweep import Metrics, calculate_metrics, is_hard_stop

class TestStep829(unittest.TestCase):
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
            "q_corr_raw_deg": np.zeros((300, 5)),
            "q_corr_applied_deg": np.zeros((300, 5)),
        }
        tick_logs["q_corr_raw_deg"][:, 2] = 2.5
        tick_logs["q_corr_applied_deg"][:, 2] = 2.0
        p0 = np.array([0, 0, 0])
        m = calculate_metrics(tick_logs, p0, 2.0)
        self.assertFalse(m.hunting_or_oscillation)
        self.assertTrue(m.clamp_fraction_per_joint[2] > 0.9)
        self.assertEqual(m.max_abs_q_corr_unclipped_deg[2], 2.5)
        self.assertTrue(is_hard_stop(m))

if __name__ == '__main__':
    unittest.main()
