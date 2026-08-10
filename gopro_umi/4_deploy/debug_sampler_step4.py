import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TimestampedTrajectory:
    target_times: np.ndarray
    actions: np.ndarray

def sample_trajectory_at_time(
    trajectory: TimestampedTrajectory,
    current_time: float
) -> Dict[str, Any]:
    
    target_times = trajectory.target_times
    
    if current_time < target_times[0]:
        return {
            "status": "BEFORE_START",
            "index": 0,
            "target_time": target_times[0],
            "action": trajectory.actions[0],
            "timing_error": target_times[0] - current_time
        }
        
    if current_time > target_times[-1]:
        return {
            "status": "EXPIRED",
            "index": len(target_times) - 1,
            "target_time": target_times[-1],
            "action": trajectory.actions[-1],
            "timing_error": target_times[-1] - current_time
        }
        
    # Valid range
    idx = np.searchsorted(target_times, current_time, side="left")
    
    return {
        "status": "VALID",
        "index": int(idx),
        "target_time": float(target_times[idx]),
        "action": trajectory.actions[idx],
        "timing_error": float(target_times[idx] - current_time)
    }

def build_dummy_trajectory(anchor_time: float) -> TimestampedTrajectory:
    num_steps = 60
    dt = 1.0 / 60.0
    offsets = np.arange(1, num_steps + 1) * dt
    target_times = anchor_time + offsets
    actions = np.random.randn(60, 6).astype(np.float32)
    return TimestampedTrajectory(target_times, actions)

def test_ideal_30hz():
    anchor = 1000.0
    traj = build_dummy_trajectory(anchor)
    print("\n--- IDEAL 30HZ TEST ---")
    print(f"{'tick':>4} | {'rel_control_t':>13} | {'idx':>3} | {'rel_target_t':>12} | {'timing_error':>12}")
    
    indices = []
    actions_correct = True
    max_act_diff = 0.0
    
    for i in range(30):
        current_time = anchor + i * (1.0 / 30.0)
        res = sample_trajectory_at_time(traj, current_time)
        
        idx = res["index"]
        indices.append(idx)
        
        rel_ctrl = current_time - anchor
        rel_tgt = res["target_time"] - anchor
        
        act_diff = np.max(np.abs(res["action"] - traj.actions[idx]))
        if act_diff > max_act_diff: max_act_diff = act_diff
        
        if i < 10 or i >= 28:
            if i == 28: print(" ...")
            print(f"{i:4d} | {rel_ctrl:13.5f} | {idx:3d} | {rel_tgt:12.5f} | {res['timing_error']:12.5f}")
            
    is_monotonic = np.all(np.diff(indices) >= 0)
    print(f"IDEAL INDEX MONOTONIC? {'YES' if is_monotonic else 'NO'}")
    print(f"ACTION CORRECTNESS max_abs_difference: {max_act_diff}")
    
def test_jitter():
    anchor = 2000.0
    traj = build_dummy_trajectory(anchor)
    print("\n--- JITTER TEST ---")
    
    np.random.seed(42)
    # Jitter between -4ms to +4ms
    jitters = np.random.uniform(-0.004, 0.004, 30)
    
    indices = []
    timing_errors = []
    
    for i in range(30):
        current_time = anchor + i * (1.0 / 30.0) + jitters[i]
        res = sample_trajectory_at_time(traj, current_time)
        indices.append(res["index"])
        if res["status"] == "VALID":
            timing_errors.append(res["timing_error"])
            
    is_monotonic = np.all(np.diff(indices) >= 0)
    print(f"selected index monotonic: {'YES' if is_monotonic else 'NO'}")
    print(f"minimum timing_error: {np.min(timing_errors):.7f}")
    print(f"maximum timing_error: {np.max(timing_errors):.7f}")
    print(f"JITTER TEST PASS? {'YES' if is_monotonic and np.min(timing_errors) >= -1e-6 else 'NO'}")
    
def test_boundaries():
    anchor = 3000.0
    traj = build_dummy_trajectory(anchor)
    print("\n--- BOUNDARY TEST ---")
    
    queries = [
        ("anchor", anchor),
        ("anchor + 1/60", anchor + 1.0/60.0),
        ("anchor + 0.5", anchor + 0.5),
        ("anchor + 59/60", anchor + 59.0/60.0),
        ("anchor + 1.0", anchor + 1.0),
        ("anchor + 1.001", anchor + 1.001),
    ]
    
    for name, t in queries:
        res = sample_trajectory_at_time(traj, t)
        print(f"{name}:\n  status/index: {res['status']}/{res['index']}")

def test_real_loop():
    print("\n--- REAL 30HZ DIAGNOSTIC PERIOD ---")
    anchor = time.monotonic()
    traj = build_dummy_trajectory(anchor)
    
    periods = []
    indices = []
    errors = []
    
    prev_time = time.monotonic()
    target_dt = 1.0 / 30.0
    
    next_tick = prev_time + target_dt
    
    expired_detected = False
    
    # Run slightly more than 1 second to catch EXPIRED
    for i in range(33):
        # Sleep until next tick (simulate real loop)
        now = time.monotonic()
        sleep_t = next_tick - now
        if sleep_t > 0:
            time.sleep(sleep_t)
            
        curr_time = time.monotonic()
        dt = curr_time - prev_time
        prev_time = curr_time
        next_tick += target_dt
        
        if i > 0:
            periods.append(dt)
            
        res = sample_trajectory_at_time(traj, curr_time)
        indices.append(res["index"])
        errors.append(res["timing_error"])
        
        if res["status"] == "EXPIRED":
            expired_detected = True
            
    periods = np.array(periods)
    print(f"min: {periods.min():.7f}")
    print(f"max: {periods.max():.7f}")
    print(f"mean: {periods.mean():.7f}")
    print(f"std: {periods.std():.7f}")
    print(f"EXPIRED TRAJECTORY DETECTED? {'YES' if expired_detected else 'NO'}")

if __name__ == '__main__':
    test_ideal_30hz()
    test_jitter()
    test_boundaries()
    test_real_loop()
