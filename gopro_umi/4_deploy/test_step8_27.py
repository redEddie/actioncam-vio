import unittest
import numpy as np
from step8_27_large_clamp_test import Metrics, calculate_metrics, is_hard_stop, make_hard_stop_observer

class TestStep827(unittest.TestCase):
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
        }
        tick_logs["q_corr_applied_deg"][:, 2] = 2.0
        p0 = np.array([0, 0, 0])
        m = calculate_metrics(tick_logs, p0, 2.0)
        self.assertFalse(m.hunting_or_oscillation)
        self.assertTrue(m.clamp_fraction_per_joint[2] > 0.9)
        self.assertTrue(is_hard_stop(m))
        
    def test_observer(self):
        obs = make_hard_stop_observer(2.0)
        for _ in range(14):
            self.assertFalse(obs(3.0, np.zeros(5), np.array([0,0,2.0,0,0])))
        self.assertTrue(obs(3.0, np.zeros(5), np.array([0,0,2.0,0,0])))

if __name__ == '__main__':
    unittest.main()
