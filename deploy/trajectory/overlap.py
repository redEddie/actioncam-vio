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


def overlap_ensemble(old_times, old_states, new_times, new_states, current_time, dt=0.1, ensemble_req_dur=0.2):
    """
    old_times: valid absolute times for old chunk
    old_states: valid states for old chunk [N, 6]
    new_times: valid absolute times for new chunk
    new_states: valid states for new chunk [M, 6]
    current_time: current execution time
    """
    # 1. Filter futures
    eps = 1e-9
    
    old_valid_mask = old_times > (current_time - eps)
    if not np.any(old_valid_mask):
        old_times_valid = np.array([])
        old_states_valid = np.array([]).reshape(0, 6)
    else:
        old_times_valid = old_times[old_valid_mask]
        old_states_valid = old_states[old_valid_mask]
        
    new_valid_mask = new_times > (current_time - eps)
    if not np.any(new_valid_mask):
        new_times_valid = np.array([])
        new_states_valid = np.array([]).reshape(0, 6)
    else:
        new_times_valid = new_times[new_valid_mask]
        new_states_valid = new_states[new_valid_mask]
        
    if len(new_times_valid) == 0:
        return {
            "status": "NEW_CHUNK_EXPIRED",
            "merged_times": old_times_valid,
            "merged_states": old_states_valid,
            "source_mode": ["OLD"] * len(old_times_valid),
            "usable_overlap_duration": 0.0
        }
        
    if len(old_times_valid) == 0:
        return {
            "status": "OLD_CHUNK_EXPIRED",
            "merged_times": new_times_valid,
            "merged_states": new_states_valid,
            "source_mode": ["NEW"] * len(new_times_valid),
            "usable_overlap_duration": 0.0
        }
        
    # 2. Overlap Calculation
    overlap_start = max(current_time, old_times[0], new_times[0])
    overlap_end = min(old_times[-1], new_times[-1])
    usable_overlap = overlap_end - overlap_start
    
    if usable_overlap <= 1e-6:
        # No overlap or negative overlap
        all_times = np.unique(np.concatenate([old_times_valid, new_times_valid]))
        all_times = all_times[all_times >= (current_time - eps)]
        # Make sure current_time is included if it falls in between
        if current_time not in all_times:
            all_times = np.sort(np.append(all_times, current_time))
            
        merged_times = []
        merged_states = []
        source_mode = []
        
        for t in all_times:
            if t < new_times[0] - eps:
                merged_times.append(t)
                merged_states.append(interpolate_trajectory(old_times, old_states, t))
                source_mode.append("OLD")
            else:
                merged_times.append(t)
                merged_states.append(interpolate_trajectory(new_times, new_states, t))
                source_mode.append("NEW")
                
        return {
            "status": "NO_VALID_OVERLAP",
            "merged_times": np.array(merged_times),
            "merged_states": np.array(merged_states),
            "source_mode": source_mode,
            "usable_overlap_duration": usable_overlap
        }
        
    # 3. Ensemble
    # Create unified time grid
    all_times = np.unique(np.concatenate([old_times_valid, new_times_valid]))
    all_times = all_times[all_times >= (current_time - eps)]
    if current_time not in all_times:
        all_times = np.sort(np.append(all_times, current_time))
    
    merged_times = []
    merged_states = []
    source_mode = []
    
    if usable_overlap >= ensemble_req_dur - 1e-6:
        status = "SUCCESS"
        
        blend_idx = 0
        for t in all_times:
            if t < overlap_start - eps:
                merged_times.append(t)
                merged_states.append(interpolate_trajectory(old_times, old_states, t))
                source_mode.append("OLD")
            else:
                s_old = interpolate_trajectory(old_times, old_states, t)
                s_new = interpolate_trajectory(new_times, new_states, t)
                
                if t <= overlap_start + eps:
                    w_new = 0.0
                else:
                    blend_idx += 1
                    if blend_idx == 1:
                        w_new = 1/3
                    elif blend_idx == 2:
                        w_new = 2/3
                    else:
                        w_new = 1.0
                        
                w_old = 1.0 - w_new
                merged_times.append(t)
                merged_states.append(w_old * s_old + w_new * s_new)
                
                if w_new < 1e-6:
                    source_mode.append("OLD")
                elif w_new > 1.0 - 1e-6:
                    source_mode.append("NEW")
                else:
                    source_mode.append("BLEND")
    else:
        status = "SHORT_OVERLAP_FALLBACK"
        # Compress transition to available overlap duration
        blend_dur = usable_overlap
        
        for t in all_times:
            if t < overlap_start - eps:
                merged_times.append(t)
                merged_states.append(interpolate_trajectory(old_times, old_states, t))
                source_mode.append("OLD")
            elif t > overlap_end + eps:
                merged_times.append(t)
                merged_states.append(interpolate_trajectory(new_times, new_states, t))
                source_mode.append("NEW")
            else:
                s_old = interpolate_trajectory(old_times, old_states, t)
                s_new = interpolate_trajectory(new_times, new_states, t)
                
                if blend_dur < 1e-6:
                    w_new = 1.0
                else:
                    w_new = (t - overlap_start) / blend_dur
                    
                w_new = max(0.0, min(1.0, w_new))
                w_old = 1.0 - w_new
                
                merged_times.append(t)
                merged_states.append(w_old * s_old + w_new * s_new)
                
                if w_new < 1e-6:
                    source_mode.append("OLD")
                elif w_new > 1.0 - 1e-6:
                    source_mode.append("NEW")
                else:
                    source_mode.append("BLEND")
                    
    return {
        "status": status,
        "merged_times": np.array(merged_times),
        "merged_states": np.array(merged_states),
        "source_mode": source_mode,
        "usable_overlap_duration": usable_overlap
    }
