import numpy as np
from trajectory.reanchor import reanchor_trajectory

def test_zero_latency():
    actions = np.zeros((15, 6))
    actions[:, 0] = 0.01  # constant velocity 0.01 per step (100mm/s)
    s_obs = np.zeros(6)
    s_arr = np.zeros(6) # no change
    t_obs = 0.0
    t_arr = 0.0
    
    res = reanchor_trajectory(actions, s_obs, s_arr, t_obs, t_arr)
    assert res["status"] == "VALID"
    assert res["first_future_index"] == 1
    assert res["alpha"] == 0.0
    assert np.allclose(res["corrected_future_states"][0, 0], 0.01)

def test_exact_100ms_boundary():
    actions = np.zeros((15, 6))
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    t_obs = 0.0
    t_arr = 0.1
    
    res = reanchor_trajectory(actions, s_obs, s_arr, t_obs, t_arr)
    assert res["current_interval_index"] == 1
    assert res["alpha"] == 0.0
    assert res["first_future_index"] == 2
    assert len(res["future_model_times"]) == 14

def test_191_5_ms():
    actions = np.zeros((15, 6))
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    t_obs = 0.0
    t_arr = 0.1915
    
    res = reanchor_trajectory(actions, s_obs, s_arr, t_obs, t_arr)
    assert res["current_interval_index"] == 1
    assert np.isclose(res["alpha"], 0.915)
    assert res["first_future_index"] == 2
    
def test_199_8_ms():
    actions = np.zeros((15, 6))
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.1998)
    assert res["current_interval_index"] == 1
    assert np.isclose(res["alpha"], 0.998)
    assert res["first_future_index"] == 2
    
def test_201_5_ms():
    actions = np.zeros((15, 6))
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.2015)
    assert res["current_interval_index"] == 2
    assert np.isclose(res["alpha"], 0.015)
    assert res["first_future_index"] == 3
    
def test_1500_ms_expiry():
    actions = np.zeros((15, 6))
    res = reanchor_trajectory(actions, np.zeros(6), np.zeros(6), 0.0, 1.5)
    assert res["status"] == "EXPIRED"

def test_1600_ms_expiry():
    actions = np.zeros((15, 6))
    res = reanchor_trajectory(actions, np.zeros(6), np.zeros(6), 0.0, 1.6)
    assert res["status"] == "EXPIRED"

def test_linear_trajectory():
    actions = np.zeros((15, 6))
    actions[:, 0] = 0.01
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.150)
    # P_raw(150ms) should be 0.015
    assert np.isclose(res["P_raw_tau"][0], 0.015)
    # Target P_raw(200ms) = 0.02
    # offset = 0 - 0.015 = -0.015
    # P_corrected(200ms) = 0.02 - 0.015 = 0.005
    assert np.isclose(res["corrected_future_states"][0, 0], 0.005)

def test_arrival_offset():
    actions = np.zeros((15, 6))
    actions[:, 0] = 0.01
    s_obs = np.zeros(6)
    s_arr = np.zeros(6)
    s_arr[0] = 0.012  # actual state is 12mm
    
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.150)
    # offset = 0.012 - 0.015 = -0.003
    # P_corrected(200ms) = 0.02 - 0.003 = 0.017
    assert np.isclose(res["corrected_future_states"][0, 0], 0.017)

def test_shape_preservation():
    actions = np.random.randn(15, 6) * 0.01
    s_obs = np.random.randn(6)
    s_arr = np.random.randn(6)
    
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.1915)
    
    # Delta of future points should be identical
    # diff of raw vs corrected
    idx = res["first_future_index"]
    raw_future = res["P_raw"][idx:]
    corr_future = res["corrected_future_states"]
    
    # diff between consecutive points
    raw_diff = np.diff(raw_future[:, :5], axis=0)
    corr_diff = np.diff(corr_future[:, :5], axis=0)
    
    max_err = np.max(np.abs(raw_diff - corr_diff))
    assert max_err < 1e-9

def test_non_accumulating_offset():
    actions = np.ones((15, 6)) * 0.01
    s_obs = np.zeros(6)
    s_arr = np.ones(6) * 0.012
    
    res = reanchor_trajectory(actions, s_obs, s_arr, 0.0, 0.150)
    # P_raw(150) = 0.015
    # offset = 0.012 - 0.015 = -0.003
    # all differences between corrected and raw should be exactly offset
    idx = res["first_future_index"]
    raw_future = res["P_raw"][idx:]
    corr_future = res["corrected_future_states"]
    
    diffs = corr_future[:, :5] - raw_future[:, :5]
    expected_offset = -0.003
    max_err = np.max(np.abs(diffs - expected_offset))
    assert max_err < 1e-9

if __name__ == '__main__':
    test_zero_latency()
    test_exact_100ms_boundary()
    test_191_5_ms()
    test_199_8_ms()
    test_201_5_ms()
    test_1500_ms_expiry()
    test_1600_ms_expiry()
    test_linear_trajectory()
    test_arrival_offset()
    test_shape_preservation()
    test_non_accumulating_offset()
    print("All tests passed.")
