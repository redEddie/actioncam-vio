import numpy as np
from trajectory.overlap import overlap_ensemble

def test_perfect_agreement():
    times = np.linspace(0, 1.5, 16)
    states = np.zeros((16, 6))
    states[:, 0] = times  # x = t
    
    # OLD and NEW overlap perfectly
    res = overlap_ensemble(times, states, times, states, 0.0)
    assert res["status"] == "SUCCESS"
    assert np.allclose(res["merged_states"], states)

def test_constant_offset():
    times = np.linspace(0, 1.5, 16)
    old_states = np.zeros((16, 6))
    new_states = np.zeros((16, 6))
    new_states[:, 0] = 3.0  # offset of 3
    
    res = overlap_ensemble(times, old_states, times, new_states, 0.0)
    
    # At t=0, old is 0, new is 3. w_new = 0 -> 0
    # At t=0.1, w_new = 1/3 -> 1.0
    # At t=0.2, w_new = 2/3 -> 2.0
    # At t=0.3, w_new = 1 -> 3.0
    assert np.isclose(res["merged_states"][0, 0], 0.0)
    assert np.isclose(res["merged_states"][1, 0], 1.0)
    assert np.isclose(res["merged_states"][2, 0], 2.0)
    assert np.isclose(res["merged_states"][3, 0], 3.0)

def test_exact_200_ms():
    old_times = np.linspace(0, 0.2, 3)
    old_states = np.zeros((3, 6))
    
    new_times = np.linspace(0, 1.5, 16)
    new_states = np.ones((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, 0.0)
    assert res["status"] == "SUCCESS"
    assert np.isclose(res["merged_states"][0, 0], 0.0)
    assert np.isclose(res["merged_states"][1, 0], 1/3)
    assert np.isclose(res["merged_states"][2, 0], 2/3)
    assert np.isclose(res["merged_states"][3, 0], 1.0)

def test_greater_than_200_ms():
    old_times = np.linspace(0, 0.5, 6)
    old_states = np.zeros((6, 6))
    
    new_times = np.linspace(0, 1.5, 16)
    new_states = np.ones((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, 0.0)
    assert res["status"] == "SUCCESS"
    assert np.isclose(res["merged_states"][0, 0], 0.0)
    assert np.isclose(res["merged_states"][1, 0], 1/3)
    assert np.isclose(res["merged_states"][2, 0], 2/3)
    assert np.isclose(res["merged_states"][3, 0], 1.0)

def test_less_than_200_ms():
    old_times = np.linspace(0, 0.1, 2)
    old_states = np.zeros((2, 6))
    
    new_times = np.linspace(0, 1.5, 16)
    new_states = np.ones((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, 0.0)
    assert res["status"] == "SHORT_OVERLAP_FALLBACK"
    assert np.isclose(res["merged_states"][0, 0], 0.0)
    assert np.isclose(res["merged_states"][1, 0], 1.0)
    assert np.isclose(res["merged_states"][2, 0], 1.0)

def test_zero_overlap():
    old_times = np.linspace(0, 0.5, 6)
    old_states = np.zeros((6, 6))
    
    new_times = np.linspace(0.6, 2.1, 16)
    new_states = np.ones((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, 0.0)
    assert res["status"] == "NO_VALID_OVERLAP"
    assert "OLD" in res["source_mode"]
    assert "NEW" in res["source_mode"]

def test_expired_new():
    old_times = np.linspace(0, 0.5, 6)
    old_states = np.zeros((6, 6))
    
    new_times = np.linspace(0, 1.5, 16)
    new_states = np.ones((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, 2.0) # current_time = 2.0
    assert res["status"] == "NEW_CHUNK_EXPIRED"

def simulate_real_latency(latency_ms):
    # Nominal overlap is 600 ms (0.6s). 
    # OLD was observed at t=0, so it ends at 1.5s
    # NEW was nominally observed at t=0.9s, so it ends at 2.4s
    # NEW arrives at t_arr = 0.9 + latency_ms/1000
    
    latency = latency_ms / 1000.0
    current_time = 0.9 + latency
    
    old_times = np.linspace(0, 1.5, 16)
    old_states = np.zeros((16, 6))
    
    new_times = np.linspace(0.9, 2.4, 16)
    new_states = np.zeros((16, 6))
    
    res = overlap_ensemble(old_times, old_states, new_times, new_states, current_time)
    return res

def test_measured_latencies():
    for lat in [191.5, 199.8, 201.5]:
        res = simulate_real_latency(lat)
        assert res["status"] == "SUCCESS"
        actual = res["usable_overlap_duration"]
        expected = 0.6 - lat/1000.0
        assert np.isclose(actual, expected)

def test_continuity():
    lat = 191.5 / 1000.0
    current_time = 0.9 + lat
    old_times = np.linspace(0, 1.5, 16)
    old_states = np.zeros((16, 6))
    # OLD moves +0.01 per step
    for i in range(16):
        old_states[i, 0] = i * 0.01
        
    new_times = np.linspace(0.9, 2.4, 16)
    new_states = np.zeros((16, 6))
    # NEW moves -0.01 per step, starting from 0.09
    for i in range(16):
        new_states[i, 0] = 0.09 - i * 0.01
        
    res_ensemble = overlap_ensemble(old_times, old_states, new_times, new_states, current_time)
    merged_states = res_ensemble["merged_states"]
    
    # For direct switch, overlap_ensemble with dur=0
    res_direct = overlap_ensemble(old_times, old_states, new_times, new_states, current_time, ensemble_req_dur=10.0)
    # Actually direct switch can be manually computed
    direct_states = []
    for t in res_ensemble["merged_times"]:
        if t < current_time + 0.1: # before transition, use old
            from trajectory.overlap import interpolate_trajectory
            direct_states.append(interpolate_trajectory(old_times, old_states, t))
        else:
            from trajectory.overlap import interpolate_trajectory
            direct_states.append(interpolate_trajectory(new_times, new_states, t))
    direct_states = np.array(direct_states)
    
    dv_ens = np.max(np.abs(np.diff(merged_states[:, 0])))
    dv_dir = np.max(np.abs(np.diff(direct_states[:, 0])))
    
    print(f"ENSEMBLE_MAX_DV = {dv_ens:.4f}")
    print(f"DIRECT_SWITCH_MAX_DV = {dv_dir:.4f}")
    assert dv_ens < dv_dir

def test_startup():
    res = simulate_real_latency(970.75)
    assert res["status"] == "OLD_CHUNK_EXPIRED"
    
if __name__ == '__main__':
    test_perfect_agreement()
    test_constant_offset()
    test_exact_200_ms()
    test_greater_than_200_ms()
    test_less_than_200_ms()
    test_zero_overlap()
    test_expired_new()
    test_measured_latencies()
    test_continuity()
    test_startup()
    print("All unit tests passed.")
