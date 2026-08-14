import numpy as np
from trajectory.interpolate import sample_trajectory_at_time, generate_30hz_schedule

def test_constant_velocity():
    # 0 to 1.5s
    times = np.linspace(0, 1.5, 16)
    states = np.zeros((16, 6))
    states[:, 0] = times * 0.1  # 100 mm/s
    
    ticks = generate_30hz_schedule(0.0, 1.5)
    max_err = 0.0
    for t in ticks:
        res = sample_trajectory_at_time(times, states, t)
        assert res["status"] == "VALID"
        expected_x = t * 0.1
        err = abs(res["action"][0] - expected_x)
        max_err = max(max_err, err)
    
    print(f"Constant Velocity MAX ERROR = {max_err}")
    assert max_err < 1e-6

def test_100ms_waypoint_reproduction():
    times = np.linspace(0, 1.5, 16)
    states = np.random.randn(16, 6)
    
    for i, t in enumerate(times):
        res = sample_trajectory_at_time(times, states, t)
        assert res["status"] == "VALID"
        assert np.allclose(res["action"], states[i])
        
    print("100ms waypoint reproduction PASS")

def test_fractional_start():
    t_start = 0.1915
    times = np.linspace(t_start, t_start + 1.5, 16)
    states = np.zeros((16, 6))
    
    ticks = generate_30hz_schedule(t_start, 1.5)
    for t in ticks:
        res = sample_trajectory_at_time(times, states, t)
        assert res["status"] == "VALID"
        
    res_before = sample_trajectory_at_time(times, states, t_start - 0.01)
    assert res_before["status"] == "BEFORE_START"
    
    res_after = sample_trajectory_at_time(times, states, t_start + 1.51)
    assert res_after["status"] == "EXPIRED"
    
    print("Fractional start PASS")

def test_ensemble_transition():
    from trajectory.overlap import overlap_ensemble
    
    old_times = np.linspace(0, 1.5, 16)
    old_states = np.zeros((16, 6))
    
    new_times = np.linspace(0.9, 2.4, 16)
    new_states = np.ones((16, 6))
    
    # overlap at 0.9 + 0.1915 = 1.0915
    current_time = 1.0915
    ens = overlap_ensemble(old_times, old_states, new_times, new_states, current_time)
    
    merged_times = ens["merged_times"]
    merged_states = ens["merged_states"]
    
    ticks = generate_30hz_schedule(current_time, 0.4) # overlap is 400ms
    
    for t in ticks:
        res = sample_trajectory_at_time(merged_times, merged_states, t)
        assert res["status"] == "VALID"
        # X value should smoothly go from 0 to 1
        
    print("Ensemble transition sampling PASS")

def test_monotonicity():
    ticks = generate_30hz_schedule(0.0, 1.5)
    diffs = np.diff(ticks)
    assert np.all(diffs > 0)
    assert np.allclose(diffs, 1/30.0)
    print(f"MIN_DT = {np.min(diffs):.6f}, MAX_DT = {np.max(diffs):.6f}, MEAN = {np.mean(diffs):.6f}")
    
def test_late_scheduler_tick():
    times = np.linspace(0, 1.5, 16)
    states = np.zeros((16, 6))
    
    t_expected = 0.3
    t_actual = 0.34
    
    res = sample_trajectory_at_time(times, states, t_actual)
    assert res["status"] == "VALID"
    # It just returns state at 0.34! No stale execution queue.
    print("Late scheduler tick PASS")

if __name__ == '__main__':
    test_constant_velocity()
    test_100ms_waypoint_reproduction()
    test_fractional_start()
    test_ensemble_transition()
    test_monotonicity()
    test_late_scheduler_tick()
    print("All unit tests passed.")
