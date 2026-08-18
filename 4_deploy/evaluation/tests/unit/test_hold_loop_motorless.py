import time
import numpy as np

def run_motorless_hold_simulation(target_duration_s=2.0, target_hz=30.0, simulated_io_latency_ms=1.0, simulate_delay_at_tick=None):
    period_s = 1.0 / target_hz
    start_time = time.monotonic()
    next_tick = start_time
    
    timestamps = []
    skipped_ticks = 0
    executed_ticks = 0
    
    while True:
        now = time.monotonic()
        if now - start_time >= target_duration_s:
            break
            
        # Resynchronize if significantly behind
        if now > next_tick + period_s:
            skipped_ticks += int((now - next_tick) / period_s)
            next_tick = now
            
        delay = next_tick - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
            
        # Simulate IO operation
        t_write = time.monotonic()
        timestamps.append(t_write)
        executed_ticks += 1
        
        # Artificial delay injection
        if simulate_delay_at_tick is not None and executed_ticks == simulate_delay_at_tick:
            time.sleep(0.080) # 80ms delay
        else:
            time.sleep(simulated_io_latency_ms / 1000.0)
            
        next_tick += period_s

    total_duration_ms = (timestamps[-1] - timestamps[0]) * 1000.0 if len(timestamps) > 1 else 0.0
    intervals_ms = np.diff(timestamps) * 1000.0 if len(timestamps) > 1 else np.array([0.0])
    
    return {
        "DURATION_MS": (time.monotonic() - start_time) * 1000.0,
        "EXPECTED_TICKS": int(target_duration_s * target_hz),
        "EXECUTED_TICKS": executed_ticks,
        "SKIPPED_TICKS": skipped_ticks,
        "MEAN_INTERVAL_MS": float(np.mean(intervals_ms)),
        "MEDIAN_INTERVAL_MS": float(np.median(intervals_ms)),
        "P95_INTERVAL_MS": float(np.percentile(intervals_ms, 95)),
        "MAX_INTERVAL_MS": float(np.max(intervals_ms))
    }

def main():
    print("=" * 60)
    print(" MOTORLESS 30Hz HOLD SCHEDULER TESTS ")
    print("=" * 60)
    
    res1 = run_motorless_hold_simulation(2.0, 30.0, 1.0)
    print(f"Normal 1ms IO latency -> Executed {res1['EXECUTED_TICKS']} ticks, Mean interval: {res1['MEAN_INTERVAL_MS']:.2f}ms")
    
    res4 = run_motorless_hold_simulation(2.0, 30.0, 4.0)
    print(f"4ms IO latency        -> Executed {res4['EXECUTED_TICKS']} ticks, Mean interval: {res4['MEAN_INTERVAL_MS']:.2f}ms")
    
    res8 = run_motorless_hold_simulation(2.0, 30.0, 8.0)
    print(f"8ms IO latency        -> Executed {res8['EXECUTED_TICKS']} ticks, Mean interval: {res8['MEAN_INTERVAL_MS']:.2f}ms")
    
    res_late = run_motorless_hold_simulation(2.0, 30.0, 1.0, simulate_delay_at_tick=15)
    print(f"Deliberate 80ms delay -> Executed {res_late['EXECUTED_TICKS']} ticks, Skipped: {res_late['SKIPPED_TICKS']}")
    
    pass_condition = (res1['EXECUTED_TICKS'] >= 58 and 
                      abs(res1['MEAN_INTERVAL_MS'] - 33.33) <= 2.0 and
                      res_late['SKIPPED_TICKS'] >= 1)
                      
    print("=" * 60)
    print("MOTORLESS_SCHEDULER_RESULT =", "PASS" if pass_condition else "FAIL")
    print("=" * 60)

if __name__ == '__main__':
    main()
