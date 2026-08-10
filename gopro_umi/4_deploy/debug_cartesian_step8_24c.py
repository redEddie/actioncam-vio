import sys
import os
import time
from pathlib import Path
import traceback

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'

os.environ['HF_HOME'] = str(PROJECT_ROOT / 'smolvla_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))
sys.path.insert(0, str(DEPLOY_DIR))

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

WRITE_ALLOWED_MOTORS = {"wrist_flex"}
WRITE_FORBIDDEN_MOTORS = {"gripper", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_roll"}

WRITE_COUNTS = {
    "wrist_flex": 0,
    "shoulder_pan": 0,
    "shoulder_lift": 0,
    "elbow_flex": 0,
    "wrist_roll": 0,
    "gripper": 0
}

def write_wrist_flex_raw_delta(follower, delta: int, G0_raw: int, calib):
    assert delta in {-2, -1, 0, 1, 2}, "invalid delta"
    target = G0_raw + delta
    assert abs(target - G0_raw) <= 2, "deviation > 2"
    assert target >= calib.range_min + 10, "Violates raw min bounds"
    assert target <= calib.range_max - 10, "Violates raw max bounds"
    
    WRITE_COUNTS["wrist_flex"] += 1
    
    follower.bus.sync_write("Goal_Position", {"wrist_flex": target}, normalize=False)

def run_test_phase(follower, delta, G0_raw, P0_raw, calib):
    target_raw = G0_raw + delta
    
    norm_delta_before = follower.bus.sync_read("Goal_Position", normalize=True)["wrist_flex"]
    G0_norm = norm_delta_before
    
    write_wrist_flex_raw_delta(follower, delta, G0_raw, calib)
    
    goal_rb = int(follower.bus.sync_read("Goal_Position", normalize=False)["wrist_flex"])
    if goal_rb != target_raw:
        raise RuntimeError(f"Goal read-back {goal_rb} != {target_raw}")
    
    norm_after = follower.bus.sync_read("Goal_Position", normalize=True)["wrist_flex"]
    norm_delta = norm_after - G0_norm
    if abs(norm_delta) > 0.25:
        raise RuntimeError(f"Degree-scale large movement detected: {norm_delta} deg")
        
    t0 = time.monotonic()
    p_50 = p_100 = p_200 = p_500 = p_1000 = None
    max_change = 0
    
    while True:
        elapsed = time.monotonic() - t0
        if elapsed > 1.05:
            break
            
        p_raw = int(follower.bus.sync_read("Present_Position", normalize=False)["wrist_flex"])
        diff = p_raw - P0_raw
        if abs(diff) > abs(max_change):
            max_change = diff
            
        if abs(diff) > 12:
            raise RuntimeError(f"Emergency Abort: wrist actual displacement {diff} > 12 raw steps")
            
        if p_50 is None and elapsed >= 0.05: p_50 = p_raw
        elif p_100 is None and elapsed >= 0.10: p_100 = p_raw
        elif p_200 is None and elapsed >= 0.20: p_200 = p_raw
        elif p_500 is None and elapsed >= 0.50: p_500 = p_raw
        elif p_1000 is None and elapsed >= 1.00: p_1000 = p_raw
        time.sleep(0.01)
        
    return goal_rb, norm_delta, p_50, p_100, p_200, p_500, p_1000, max_change

def run_return(follower, G0_raw, P0_raw, calib):
    write_wrist_flex_raw_delta(follower, 0, G0_raw, calib)
    goal_rb = int(follower.bus.sync_read("Goal_Position", normalize=False)["wrist_flex"])
    time.sleep(1.0)
    p_raw = int(follower.bus.sync_read("Present_Position", normalize=False)["wrist_flex"])
    return goal_rb, p_raw - P0_raw

def main():
    print("Initializing follower without modifying PID or torque...")
    config = SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True, disable_torque_on_disconnect=False)
    follower = SO100Follower(config)
    follower.connect()
    
    # Read Baseline
    pres_raw_all = follower.bus.sync_read("Present_Position", normalize=False)
    goal_raw_all = follower.bus.sync_read("Goal_Position", normalize=False)
    
    P0_raw = int(pres_raw_all["wrist_flex"])
    G0_raw = int(goal_raw_all["wrist_flex"])
    
    calib = follower.bus.calibration["wrist_flex"]
    margin = min(G0_raw - calib.range_min, calib.range_max - G0_raw)
    
    if margin < 10:
        print(f"FAILED RAW MARGIN CHECK: margin={margin} steps")
        return
        
    theoretical_deg_step = 360.0 / 4096.0
    
    results = {}
    exception_occurred = False
    
    try:
        # A. +1
        results["+1"] = run_test_phase(follower, 1, G0_raw, P0_raw, calib)
        results["+1_ret"] = run_return(follower, G0_raw, P0_raw, calib)
        
        # B. -1
        results["-1"] = run_test_phase(follower, -1, G0_raw, P0_raw, calib)
        results["-1_ret"] = run_return(follower, G0_raw, P0_raw, calib)
        
        # C. +2
        results["+2"] = run_test_phase(follower, 2, G0_raw, P0_raw, calib)
        results["+2_ret"] = run_return(follower, G0_raw, P0_raw, calib)
        
        # D. -2
        results["-2"] = run_test_phase(follower, -2, G0_raw, P0_raw, calib)
        results["-2_ret"] = run_return(follower, G0_raw, P0_raw, calib)
        
    except Exception as e:
        exception_occurred = True
        print(f"EXCEPTION DURING TEST: {e}")
        traceback.print_exc()
        try:
            # ONLY return wrist_flex to G0_raw if communication allows
            write_wrist_flex_raw_delta(follower, 0, G0_raw, calib)
        except:
            pass
            
    # Read Final State for other joints
    pres_raw_all_after = follower.bus.sync_read("Present_Position", normalize=False)
    gripper_moved = pres_raw_all_after.get("gripper", 0) != pres_raw_all.get("gripper", 0)
    
    print("\n\n" + "="*40)
    print("STEP 8.24C CLEAN WRIST-ONLY RAW TEST")
    print("====================================")
    
    print("\n1. PRIOR STEP8.24B\nINVALID due gripper/unapproved rerun:\nYES")
    print("\n2. PRE-RUN WRITE AUDIT\nGoal_Position write sites:\n[write_wrist_flex_raw_delta]\nAll wrist_flex only:\nYES")
    print("\n3. GRIPPER WRITE PATH PRESENT?\nNO")
    print("\n4. CURRENT TEST MOTOR\nwrist_flex only:\nYES")
    print(f"\n5. WRIST BASELINE\nG0 raw:\n{G0_raw}\nP0 raw:\n{P0_raw}")
    print(f"\n6. SAFE RAW MARGIN\nmin margin:\n{margin}steps\nPASS")
    
    def print_phase(name, res):
        print(f"\n7. {name}" if "+1" in name else (f"\n9. {name}" if "-1" in name else (f"\n11. {name}" if "+2" in name else f"\n13. {name}")))
        if res:
            g_rb, n_delta, p50, p100, p200, p500, p1000, m_c = res
            print(f"Goal requested:\n{G0_raw + int(name.replace('+', '').replace('-', '-')) if '-' in name else G0_raw + int(name.replace('+', ''))}")
            print(f"Goal readback:\n{g_rb}")
            print(f"Present response:\n{m_c}steps")
            
    def print_ret(name, res):
        print(f"\n8. {name} RETURN" if "+1" in name else (f"\n10. {name} RETURN" if "-1" in name else (f"\n12. {name} RETURN" if "+2" in name else f"\n14. {name} RETURN")))
        if res:
            g_rb, resid = res
            print(f"Present residual:\n{resid}steps")
            
    if not exception_occurred:
        print_phase("+1", results["+1"])
        print_ret("+1", results["+1_ret"])
        print_phase("-1", results["-1"])
        print_ret("-1", results["-1_ret"])
        print_phase("+2", results["+2"])
        print_ret("+2", results["+2_ret"])
        print_phase("-2", results["-2"])
        print_ret("-2", results["-2_ret"])
        
        pos_thresh = "1" if abs(results["+1"][7]) >= 1 else ("2" if abs(results["+2"][7]) >= 1 else ">2")
        neg_thresh = "1" if abs(results["-1"][7]) >= 1 else ("2" if abs(results["-2"][7]) >= 1 else ">2")
    else:
        pos_thresh = "insufficient"
        neg_thresh = "insufficient"
        
    print(f"\n15. POSITIVE THRESHOLD\n{pos_thresh}")
    print(f"\n16. NEGATIVE THRESHOLD\n{neg_thresh}")
    
    dir_asym = "YES" if (pos_thresh != neg_thresh and pos_thresh != "insufficient") else ("INSUFFICIENT" if pos_thresh == "insufficient" else "NO")
    print(f"\n17. DIRECTIONAL ASYMMETRY\n{dir_asym}")
    
    if not exception_occurred:
        print(f"\n18. HYSTERESIS\npositive return residual:\n{results['+2_ret'][1]}steps\nnegative return residual:\n{results['-2_ret'][1]}steps")
    else:
        print("\n18. HYSTERESIS\npositive return residual:\ninsufficient\nnegative return residual:\ninsufficient")
        
    print(f"\n19. OTHER ARM JOINT GOAL WRITE COUNTS\npan:\n{WRITE_COUNTS['shoulder_pan']}\nlift:\n{WRITE_COUNTS['shoulder_lift']}\nelbow:\n{WRITE_COUNTS['elbow_flex']}\nwrist_roll:\n{WRITE_COUNTS['wrist_roll']}")
    print(f"\n20. GRIPPER GOAL WRITE COUNT\n{WRITE_COUNTS['gripper']}")
    print(f"\n21. UNEXPECTED GRIPPER MOVEMENT\n{'YES' if gripper_moved else 'NO'}")
    
    print(f"\n22. EXCEPTION DURING TEST\n{'YES' if exception_occurred else 'NO'}")
    print("\n23. AUTO-RERUN OCCURRED?\nNO")
    
    print(f"\n24. SAFETY INCIDENT\n{'YES' if exception_occurred or gripper_moved else 'NO'}")
    
    verdict = "PASS" if not exception_occurred and not gripper_moved else "FAIL"
    print(f"\n25. STEP8.24C VERDICT\n{verdict}")
    
    claim = "insufficient / invalid"
    if verdict == "PASS":
        if pos_thresh == "1" or neg_thresh == "1":
            claim = "1-step response observed"
        elif pos_thresh == "2" or neg_thresh == "2":
            claim = "finite <=2-step threshold observed"
        elif pos_thresh == ">2" and neg_thresh == ">2":
            claim = "threshold >2 steps"
        if dir_asym == "YES":
            claim = "directional behavior observed"
    print(f"\n26. BREAKAWAY CLAIM STATUS\n{claim}")
    
    rec = "repeat only after fixing a reported script issue"
    if verdict == "PASS":
        if claim == "threshold >2 steps":
            rec = "design bounded >2-step wrist test"
        elif claim == "finite <=2-step threshold observed" or claim == "directional behavior observed":
            rec = "investigate directional behavior" # Wait, the options are: inspect another load-bearing arm joint, design bounded >2-step wrist test, return to Cartesian controller diagnosis, investigate directional behavior
            
    print(f"\n27. NEXT RECOMMENDATION\n{rec}")
    
    print("\n========================================")
    follower.disconnect()

if __name__ == "__main__":
    main()
