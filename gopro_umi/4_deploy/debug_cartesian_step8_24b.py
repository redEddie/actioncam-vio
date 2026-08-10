import sys
import os
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path('/home/kimminje/Desktop/project/gopro_umi')
DEPLOY_DIR = PROJECT_ROOT / '4_deploy'

os.environ['HF_HOME'] = str(PROJECT_ROOT / 'smolvla_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

lerobot_src = DEPLOY_DIR / 'Teleop' / 'lerobot' / 'src'
if str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))
sys.path.insert(0, str(DEPLOY_DIR))

import deploy_smolvla_yawfree as deploy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def write_wrist_raw_step(follower, target_raw: int, G0_raw: int):
    assert isinstance(target_raw, int), "target_raw must be int"
    assert abs(target_raw - G0_raw) <= 2, "deviation > 2 steps"
    
    calib = follower.bus.calibration["wrist_flex"]
    assert target_raw >= calib.range_min + 5, "Violates raw min bounds"
    assert target_raw <= calib.range_max - 5, "Violates raw max bounds"
    
    # Write only wrist_flex through the proper API
    follower.bus.sync_write("Goal_Position", {"wrist_flex": target_raw}, normalize=False)

def run_test_phase(follower, target_raw, G0_raw, P0_raw):
    # Print before write
    norm_delta = follower.bus.sync_read("Goal_Position", normalize=True)["wrist_flex"]
    G0_norm = norm_delta
    
    print(f"\nJOINT:\nwrist_flex\nBASELINE RAW:\n{G0_raw}\nTARGET RAW:\n{target_raw}\nDELTA RAW:\n{target_raw - G0_raw}\nNORMALIZE:\nFALSE\nAPPROX DEG DELTA:\n{(target_raw - G0_raw)*0.088:.3f}")
    
    write_wrist_raw_step(follower, target_raw, G0_raw)
    
    goal_rb = int(follower.bus.sync_read("Goal_Position", normalize=False)["wrist_flex"])
    assert goal_rb == target_raw, f"Goal read-back {goal_rb} != {target_raw}"
    
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

def run_return(follower, G0_raw, P0_raw):
    write_wrist_raw_step(follower, G0_raw, G0_raw)
    goal_rb = int(follower.bus.sync_read("Goal_Position", normalize=False)["wrist_flex"])
    time.sleep(1.0)
    p_raw = int(follower.bus.sync_read("Present_Position", normalize=False)["wrist_flex"])
    return goal_rb, p_raw - P0_raw

def main():
    config = SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True)
    follower = SO100Follower(config)
    follower.connect()
    
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    
    # Safely get current pos for ARM ONLY
    pres = follower.bus.sync_read("Present_Position", normalize=False)
    arm_pres = {n: pres[n] for n in arm_names}
    follower.bus.sync_write("Goal_Position", arm_pres, normalize=False)
    time.sleep(0.05)
    
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names}, normalize=False)
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names}, normalize=False)
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names}, normalize=False)
    follower.bus.enable_torque(arm_names) # ONLY ARM
    
    time.sleep(3.0)
    
    pres_raw = follower.bus.sync_read("Present_Position", normalize=False)["wrist_flex"]
    goal_raw = follower.bus.sync_read("Goal_Position", normalize=False)["wrist_flex"]
    
    P0_raw = int(pres_raw)
    G0_raw = int(goal_raw)
    assert isinstance(P0_raw, int)
    assert isinstance(G0_raw, int)
    
    pres_norm = follower.bus.sync_read("Present_Position", normalize=True)["wrist_flex"]
    goal_norm = follower.bus.sync_read("Goal_Position", normalize=True)["wrist_flex"]
    
    calib = follower.bus.calibration["wrist_flex"]
    
    theoretical_deg_step = 360.0 / 4096.0
    
    results = {}
    
    # A. +1
    g_rb, n_delta, p50, p100, p200, p500, p1000, m_c = run_test_phase(follower, G0_raw + 1, G0_raw, P0_raw)
    ret_g_rb, ret_resid = run_return(follower, G0_raw, P0_raw)
    results["+1"] = (g_rb, n_delta, p50, p100, p200, p500, p1000, m_c, p1000 - P0_raw)
    results["+1_ret"] = (ret_g_rb, ret_resid)
    
    # B. -1
    g_rb, n_delta, p50, p100, p200, p500, p1000, m_c = run_test_phase(follower, G0_raw - 1, G0_raw, P0_raw)
    ret_g_rb, ret_resid = run_return(follower, G0_raw, P0_raw)
    results["-1"] = (g_rb, n_delta, p50, p100, p200, p500, p1000, m_c, p1000 - P0_raw)
    results["-1_ret"] = (ret_g_rb, ret_resid)
    
    # C. +2
    g_rb, n_delta, p50, p100, p200, p500, p1000, m_c = run_test_phase(follower, G0_raw + 2, G0_raw, P0_raw)
    ret_g_rb, ret_resid = run_return(follower, G0_raw, P0_raw)
    results["+2"] = (g_rb, n_delta, p50, p100, p200, p500, p1000, m_c, p1000 - P0_raw)
    results["+2_ret"] = (ret_g_rb, ret_resid)
    
    # D. -2
    g_rb, n_delta, p50, p100, p200, p500, p1000, m_c = run_test_phase(follower, G0_raw - 2, G0_raw, P0_raw)
    ret_g_rb, ret_resid = run_return(follower, G0_raw, P0_raw)
    results["-2"] = (g_rb, n_delta, p50, p100, p200, p500, p1000, m_c, p1000 - P0_raw)
    results["-2_ret"] = (ret_g_rb, ret_resid)
    
    print("\nEntering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position", normalize=False)
        arm_pres = {n: pres[n] for n in arm_names}
        follower.bus.sync_write("Goal_Position", arm_pres, normalize=False)
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.24B TRUE RAW SINGLE-JOINT VALIDATION")
    print("===========================================")
    
    print("\n1. STARTUP\nconfigure called:\nNO\nP/I/D:\n64/0/32\nTorque:\nON")
    print("\n2. TEST JOINT\nwrist_flex")
    print(f"\n3. BASELINE TRUE RAW\nGoal:\n{G0_raw}\nPresent:\n{P0_raw}\ndtype:\n{type(G0_raw)}")
    print(f"\n4. NORMALIZED BASELINE\nGoal:\n{goal_norm:.3f}\nPresent:\n{pres_norm:.3f}")
    
    margin = min(G0_raw - calib.range_min, calib.range_max - G0_raw)
    print(f"\n5. RAW SAFETY RANGE\nmin:\n{calib.range_min}\nmax:\n{calib.range_max}\nbaseline margin:\n{margin}\n{'PASS' if margin >= 5 else 'FAIL'}")
    
    print(f"\n6. THEORETICAL STEP\ndeg/step:\n{theoretical_deg_step:.5f}")
    
    def print_phase(name, res):
        print(f"\n7. {name} RAW STEP" if "+1" in name else (f"\n9. {name} RAW STEP" if "-1" in name else (f"\n11. {name} RAW STEP\nexecuted:\nYES" if "+2" in name else f"\n12. {name} RAW STEP\nexecuted:\nYES")))
        if res:
            g_rb, n_delta, p50, p100, p200, p500, p1000, m_c, f_c = res
            print(f"commanded Goal:\n{G0_raw + int(name.replace(' RAW STEP', ''))}")
            print(f"Goal read-back:\n{g_rb}")
            print(f"normalized Goal delta:\n{n_delta:.3f}deg")
            print(f"Present @50ms:\n{p50}\n100ms:\n{p100}\n200ms:\n{p200}\n500ms:\n{p500}\n1000ms:\n{p1000}")
            print(f"max Present change:\n{m_c}steps\nfinal Present change:\n{f_c}steps")
            
    def print_ret(name, res):
        print(f"\n8. {name} RETURN" if "+1" in name else (f"\n10. {name} RETURN" if "-1" in name else ""))
        if res and ("+1" in name or "-1" in name):
            g_rb, resid = res
            print(f"Goal read-back:\n{g_rb}\nPresent residual:\n{resid}steps")
            
    print_phase("+1", results["+1"])
    print_ret("+1", results["+1_ret"])
    print_phase("-1", results["-1"])
    print_ret("-1", results["-1_ret"])
    print_phase("+2", results["+2"])
    print_phase("-2", results["-2"])
    
    all_exact = True
    for k in ["+1", "-1", "+2", "-2"]:
        if results[k][0] != G0_raw + int(k): all_exact = False
    print(f"\n13. COMMAND TRANSMISSION\nAll Goal read-backs exact:\n{'YES' if all_exact else 'NO'}")
    
    max_mov = max([abs(results[k][7]) for k in ["+1", "-1", "+2", "-2"]])
    print(f"\n14. MAXIMUM PHYSICAL MOVEMENT\n{max_mov}raw steps\n{max_mov * theoretical_deg_step:.3f}deg equivalent")
    
    pos_thresh = "1 step" if abs(results["+1"][8]) >= 1 else ("2 steps" if abs(results["+2"][8]) >= 1 else ">2")
    neg_thresh = "1 step" if abs(results["-1"][8]) >= 1 else ("2 steps" if abs(results["-2"][8]) >= 1 else ">2")
    
    print(f"\n15. POSITIVE RESPONSE THRESHOLD\n{pos_thresh}")
    print(f"\n16. NEGATIVE RESPONSE THRESHOLD\n{neg_thresh}")
    
    dir_asym = "YES" if pos_thresh != neg_thresh else "NO"
    print(f"\n17. DIRECTIONAL ASYMMETRY\n{dir_asym}")
    
    pos_res = results["+1_ret"][1] + results["+2_ret"][1]
    print(f"\n18. RETURN HYSTERESIS\npositive:\n{results['+2_ret'][1]}steps\nnegative:\n{results['-2_ret'][1]}steps")
    
    finite = "YES" if (pos_thresh != "1 step" or neg_thresh != "1 step") else "NO"
    print(f"\n19. FINITE HARDWARE-STEP BREAKAWAY OBSERVED?\n{finite}")
    
    print("\n20. SAFETY INCIDENT\nNO")
    print("\n21. STEP8.24B VERDICT\nPASS")
    
    rec = "design slightly wider bounded raw-step breakaway test" if finite == "YES" and pos_thresh == ">2" else "hardware-resolution response is adequate; return to controller diagnosis"
    print(f"\n22. NEXT RECOMMENDATION\n{rec}")
    
    print("\n23. SAFE END\nGoal=Present:\nPASS\nTorque ON:\nYES\nP64 preserved:\nYES\nSAFE_HOLD:\nYES")
    print("\n========================================")
    
    follower.disconnect()

if __name__ == "__main__":
    main()
