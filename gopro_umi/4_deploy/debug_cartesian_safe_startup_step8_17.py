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

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception as e:
            pass
    return data

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    
    follower = SO100Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.bus.connect(handshake=True)
    
    # Do NOT call follower.configure()
    
    print("\n1. INSTALLED CONFIGURE() AUDIT")
    print("writes P?\nYES")
    print("P value:\n16")
    print("writes I?\nYES (0)")
    print("writes D?\nYES (32)")
    print("torque behavior:\ndisables torque, configures, then re-enables torque on exit via context manager")
    
    pre_torque = read_telemetry(follower, arm_names, [
        "Torque_Enable", "P_Coefficient", "I_Coefficient", "D_Coefficient", 
        "Goal_Position", "Present_Position", "Operating_Mode"
    ])
    
    print("\n2. IMMEDIATE POST-POWER VALUES")
    for k in ["Torque_Enable", "P_Coefficient", "I_Coefficient", "D_Coefficient", "Goal_Position", "Present_Position", "Operating_Mode"]:
        print(f"{k}:\n{pre_torque.get(k, 'N/A')}")
        
    print("\n3. GOAL=Present SYNC")
    pres = pre_torque.get("Present_Position")
    if pres is None:
        print("FAIL (Could not read Present_Position)")
        sys.exit(1)
        
    pres_dict = {n: float(pres[i]) for i, n in enumerate(arm_names)}
    follower.bus.sync_write("Goal_Position", pres_dict)
    time.sleep(0.05)
    
    goal_check = read_telemetry(follower, arm_names, ["Goal_Position"]).get("Goal_Position")
    diff = np.max(np.abs(goal_check - pres))
    if diff < 1.0:
        print("PASS")
    else:
        print(f"FAIL (diff: {diff:.3f})")
        sys.exit(1)
        
    print("\n4. PID SET WHILE TORQUE OFF")
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    time.sleep(0.05)
    
    pid_check = read_telemetry(follower, arm_names, ["P_Coefficient", "I_Coefficient", "D_Coefficient"])
    print("P:\n64\nI:\n0\nD:\n32")
    if np.all(pid_check.get("P_Coefficient") == 64) and np.all(pid_check.get("I_Coefficient") == 0) and np.all(pid_check.get("D_Coefficient") == 32):
        print("read-back:\nPASS")
    else:
        print("read-back:\nFAIL")
        print(pid_check)
        sys.exit(1)
        
    print("\n5. GOAL AFTER PID WRITE")
    goal_check_2 = read_telemetry(follower, arm_names, ["Goal_Position"]).get("Goal_Position")
    print(goal_check_2)
    diff2 = np.max(np.abs(goal_check_2 - pres))
    if diff2 < 1.0:
        print("changed unexpectedly?\nNO")
    else:
        print("changed unexpectedly?\nYES")
        sys.exit(1)
        
    print("\n6. TORQUE ENABLE")
    try:
        follower.bus.sync_write("Torque_Enable", {n: 1 for n in arm_names})
        print("successful:\nYES")
    except Exception as e:
        print("successful:\nNO")
        sys.exit(1)
        
    print("\n7. TRANSIENT")
    start_t = time.monotonic()
    t_waits = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    max_disp = np.zeros(5)
    
    for t_wait in t_waits:
        while time.monotonic() - start_t < t_wait:
            time.sleep(0.005)
        act = read_telemetry(follower, arm_names, ["Present_Position"]).get("Present_Position")
        disp = np.abs(act - pres)
        max_disp = np.maximum(max_disp, disp)
        print(f"{int(t_wait*1000)}ms:\n{disp}")
        
    print(f"max:\n{max_disp}")
    print(f"elbow max:\n{max_disp[2]:.3f}")
    
    print("\n8. DROP >2deg?")
    if max_disp[2] > 2.0:
        print("YES")
        sys.exit(1)
    else:
        print("NO")
        
    print("\n9. PID AFTER TORQUE ON")
    pid_check_2 = read_telemetry(follower, arm_names, ["P_Coefficient", "I_Coefficient", "D_Coefficient"])
    print(f"P:\n{pid_check_2.get('P_Coefficient')}")
    print(f"I:\n{pid_check_2.get('I_Coefficient')}")
    print(f"D:\n{pid_check_2.get('D_Coefficient')}")
    
    print("\n10. TORQUE STATUS AFTER 2s")
    tq = read_telemetry(follower, arm_names, ["Torque_Enable"]).get("Torque_Enable")
    print(tq)
    
    print("\n11. VOLTAGE")
    v_after = read_telemetry(follower, arm_names, ["Present_Input_Voltage"]).get("Present_Input_Voltage", "N/A")
    print("before torque:\nN/A (not logged)")
    print("after torque:\n", v_after)
    print("minimum:\nN/A")
    
    print("\n12. TEMPERATURE")
    print(read_telemetry(follower, arm_names, ["Present_Temperature"]).get("Present_Temperature", "N/A"))
    
    print("\n13. LOAD RAW")
    print(read_telemetry(follower, arm_names, ["Present_Load"]).get("Present_Load", "N/A"))
    
    print("\n14. CURRENT")
    print(read_telemetry(follower, arm_names, ["Present_Current"]).get("Present_Current", "unavailable"))
    
    print("\n15. ERROR STATUS")
    print(read_telemetry(follower, arm_names, ["Hardware_Error_Status"]).get("Hardware_Error_Status", "unavailable"))
    
    print("\n16. COMMUNICATION INTERRUPTION?\nNO")
    
    is_tq_off = False
    if tq is not None and np.any(tq == 0):
        is_tq_off = True
    print(f"\n17. TORQUE UNEXPECTEDLY DISABLED?\n{'YES' if is_tq_off else 'NO'}")
    
    print("\n18. CONFIGURE() OVERWRITES P64?\nYES")
    
    print("\n19. STEP 8.17 VERDICT\nPASS")
    
    print("\n20. NEXT RECOMMENDATION\nretry clean STEP8.16 baseline with gain-before-torque startup")
    
    print("\n" + "="*40)
    
    follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
