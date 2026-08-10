# STEP 8.24A — ZERO-MOTION Unit/Command-Path Safety Audit
import numpy as np

# This is an offline forensic analysis, no hardware execution.

# Incident Trace
# file: debug_cartesian_step8_23.py
# line: 196
# function: sync_write
# intended unit: raw integer
# numeric: 2048 (approx)
# actual API interpretation: M100_100 normalized range (~degrees)
# resulting normalized command: 100.0 (clipped)
# expected hardware register: max calibration limit (e.g. 3162)

def unnormalize(val, min_, max_):
    bounded_val = min(100.0, max(-100.0, val))
    return int(((bounded_val + 100) / 200) * (max_ - min_) + min_)

def normalize(val, min_, max_):
    return (val - min_) / (max_ - min_) * 200 - 100

def dry_run():
    min_ = 913
    max_ = 3162
    
    print("15. REPRESENTATIVE DRY RUN")
    for deg in [46.0, 46.1, 46.2, 46.5, 47.0]:
        raw = unnormalize(deg, min_, max_)
        print(f"{deg}deg -> {raw} raw")
        
    print("\n16. ROUND-TRIP ERROR")
    max_err = 0
    for deg in [46.0, 46.1, 46.2, 46.5, 47.0]:
        raw = unnormalize(deg, min_, max_)
        recon = normalize(raw, min_, max_)
        err = abs(recon - deg)
        max_err = max(max_err, err)
    print(f"max_err: {max_err}")

if __name__ == "__main__":
    dry_run()
