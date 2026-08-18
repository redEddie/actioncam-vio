import numpy as np

def interpolate_trajectory(times, trajectory, target_time):
    """
    Linearly interpolate the trajectory at target_time.
    times: 1D array of timestamps
    trajectory: 2D array of states [N, 6]
    """
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return trajectory[0]
    if idx == len(times):
        return trajectory[-1]
    
    t0, t1 = times[idx-1], times[idx]
    p0, p1 = trajectory[idx-1], trajectory[idx]
    
    dt = t1 - t0
    if dt < 1e-9:
        return p1
    
    alpha = (target_time - t0) / dt
    return (1 - alpha) * p0 + alpha * p1

def sample_trajectory_at_time(merged_times, merged_states, t_query):
    """
    Query the trajectory at an absolute timestamp t_query.
    No hidden queues or state; purely timestamp-based.
    Returns:
      status: "VALID", "BEFORE_START", "EXPIRED"
      action: interpolated state (if VALID, else None)
    """
    if len(merged_times) == 0:
        return {"status": "EMPTY", "action": None}
        
    # Floating tolerance
    eps = 1e-9
    
    if t_query < merged_times[0] - eps:
        return {"status": "BEFORE_START", "action": None}
        
    if t_query > merged_times[-1] + eps:
        return {"status": "EXPIRED", "action": None}
        
    action = interpolate_trajectory(merged_times, merged_states, t_query)
    
    return {"status": "VALID", "action": action}

def generate_30hz_schedule(t_start, duration, min_dt=0.0333333333333333):
    """
    Generate absolute timestamps for a 30Hz control loop.
    t_tick[n] = t_start + n * (1/30)
    """
    num_ticks = int(np.ceil(duration / min_dt))
    # strict formula avoiding cumulative drift
    ticks = t_start + np.arange(num_ticks) * min_dt
    return ticks
