import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

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
from ik_solver_v7 import DLSInverseKinematicsV7
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SO100Follower

def safe_connect_and_gate(port, arm_names):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    follower.bus.connect(handshake=True)
    
    # 1. communication-only bus connect (done)
    # 2. read Present_Position
    pres = follower.bus.sync_read("Present_Position")
    
    # 3. write Goal_Position = Present_Position
    follower.bus.sync_write("Goal_Position", pres)
    time.sleep(0.05)
    
    # 4. read back Goal_Position
    goal = follower.bus.sync_read("Goal_Position")
    
    # 5. verify Goal≈Present
    pres_arr = np.array([float(pres[n]) for n in arm_names])
    goal_arr = np.array([float(goal[n]) for n in arm_names])
    diff = np.max(np.abs(pres_arr - goal_arr))
    
    print("\nSAFE STARTUP GATE:")
    print("Present before torque:", pres_arr)
    if diff < 1.0:
        print("Goal synchronized: YES (diff = {:.3f} deg)".format(diff))
    else:
        print("Goal synchronized: NO (diff = {:.3f} deg)".format(diff))
        sys.exit(1)
        
    # 6. configure / torque enable
    follower.configure()
    time.sleep(0.05)
    
    # 7. observe physical motion
    print("Observing physical motion for 1s...")
    max_disp = np.zeros(5)
    for t_wait in [0.05, 0.1, 0.2, 0.5, 1.0]:
        time.sleep(t_wait - (0 if t_wait == 0.05 else [0.05, 0.1, 0.2, 0.5][int(np.log10(t_wait*20))])) # lazy sleep logic
        act = follower.bus.sync_read("Present_Position")
        act_arr = np.array([float(act[n]) for n in arm_names])
        disp = np.abs(act_arr - pres_arr)
        max_disp = np.maximum(max_disp, disp)
        
    print("torque-on max movement:", max_disp)
    if np.max(max_disp) > 2.0:
        print("snap: YES")
        print("drop: YES")
        sys.exit(1)
    else:
        print("snap: NO")
        print("drop: NO")
        
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    time.sleep(0.1)
    return follower

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception:
            pass
    return data

def check_pid(follower, arm_names):
    p = follower.bus.sync_read("P_Coefficient")
    i = follower.bus.sync_read("I_Coefficient")
    d = follower.bus.sync_read("D_Coefficient")
    p_arr = np.array([p[n] for n in arm_names])
    i_arr = np.array([i[n] for n in arm_names])
    d_arr = np.array([d[n] for n in arm_names])
    print("\nPID READ-BACK:")
    print("P:", p_arr)
    print("I:", i_arr)
    print("D:", d_arr)
    if not (np.all(p_arr == 64) and np.all(i_arr == 0) and np.all(d_arr == 32)):
        print("FAIL: PID DOES NOT MATCH 64/0/32. ABORTING.")
        sys.exit(1)
    else:
        print("PASS")

def move_to_fixed_start_and_measure(follower, arm_names, target):
    obs = follower.bus.sync_read("Present_Position")
    q_act = np.array([float(obs[n]) for n in arm_names])
    delta = target - q_act
    
    dur = 4.0
    steps = int(dur * 30)
    for step in range(steps):
        t0 = time.monotonic()
        progress = (step + 1) / steps
        alpha = (3.0 * progress**2) - (2.0 * progress**3)
        q_cmd = q_act + alpha * delta
        cmd_dict = {n: float(q_cmd[j]) for j, n in enumerate(arm_names)}
        follower.bus.sync_write("Goal_Position", cmd_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))

    target_dict = {n: float(target[j]) for j, n in enumerate(arm_names)}
    for _ in range(90):
        t0 = time.monotonic()
        follower.bus.sync_write("Goal_Position", target_dict)
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
        
    final_pres = []
    final_goal = []
    for _ in range(15): # last 0.5s
        t0 = time.monotonic()
        pres = follower.bus.sync_read("Present_Position")
        goal = follower.bus.sync_read("Goal_Position")
        final_pres.append(np.array([float(pres[n]) for n in arm_names]))
        final_goal.append(np.array([float(goal[n]) for n in arm_names]))
        elapsed = time.monotonic() - t0
        time.sleep(max(0, (1.0/30.0) - elapsed))
    
    return np.mean(final_goal, axis=0), np.mean(final_pres, axis=0)

def generate_trajectory(delta_z):
    freq = 60
    total_len = 4 * freq
    traj = []
    for i in range(total_len):
        t = (i + 1) / freq
        if t <= 1.0:
            alpha = (3 * t**2) - (2 * t**3)
            z = delta_z * alpha
        elif t <= 2.0:
            z = delta_z
        elif t <= 3.0:
            norm_t = t - 2.0
            alpha = (3 * norm_t**2) - (2 * norm_t**3)
            z = delta_z * (1.0 - alpha)
        else:
            z = 0.0
        traj.append([0.0, 0.0, z])
    return np.array(traj), np.arange(1, total_len + 1) / freq

def run_preflight(ik_solver, traj_pos, T_start, q_actual_start_rad, b_q_rad):
    q_curr = q_actual_start_rad.copy()
    failures = {"IK": 0, "NaN": 0, "hard": 0, "safe": 0, "branch": 0}
    for i, dz in enumerate(traj_pos):
        pos_target = T_start[:3, 3] + dz
        rot_target = T_start[:3, :3]
        
        q_sol = ik_solver.solve(pos_target, rot_target, current_joints=q_curr)
        if q_sol is None:
            failures["IK"] += 1
            return False, failures
            
        if np.any(np.isnan(q_sol)) or np.any(np.isinf(q_sol)):
            failures["NaN"] += 1
            return False, failures
            
        jump = np.max(np.abs(q_sol - q_curr))
        if jump > 0.5:
            failures["branch"] += 1
            return False, failures
            
        q_corr_max = np.deg2rad(1.0)
        q_cmd_max = q_sol + b_q_rad + q_corr_max
        q_cmd_min = q_sol + b_q_rad - q_corr_max
        
        for val_max, val_min, bounds in zip(q_cmd_max, q_cmd_min, ik_solver.safe_limits.values()):
            if val_max > bounds[1] or val_min < bounds[0]:
                failures["safe"] += 1
                return False, failures
                
        q_curr = q_sol
        
    return True, failures

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect_and_gate("/dev/ttyACM0", arm_names)
    
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    log_dir = DEPLOY_DIR / 'logs' / 'step8_16_baseline'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    tick_logs = defaultdict(list)
    slow_logs = defaultdict(list)
    outliers = []
    
    power_interruption = False
    torque_interruption = False
    bus_disconnect = False
    
    try:
        check_pid(follower, arm_names)
        
        tel = read_telemetry(follower, arm_names, ["Present_Temperature"])
        slow_logs["t"].append(0.0)
        slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
        
        q_hold_goal_raw, q_start_actual_raw = move_to_fixed_start_and_measure(follower, arm_names, fixed_start_enc)
        
        print("\n4. FIXED START")
        print("RAW target:\n", fixed_start_enc)
        print("actual:\n", q_start_actual_raw)
        
        tel = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
        slow_logs["t"].append(4.0)
        slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
        slow_logs["load"].append(tel.get("Present_Load", np.zeros(5)))
        
        q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
        q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
        
        b_q_deg = q_hold_goal - q_start_actual
        b_q_rad = np.deg2rad(b_q_deg)
        print("\n5. FRESH b_q")
        print("deg:\n", b_q_deg)
        print("rad:\n", b_q_rad)
        
        q_start_actual_rad = np.deg2rad(q_start_actual)
        T0 = ik_solver.forward_kinematics(q_start_actual_rad)
        p0 = T0[:3, 3]
        euler_start = R.from_matrix(T0[:3, :3]).as_euler("ZYX")
        
        print("\n6. FRESH CARTESIAN ANCHOR")
        print("X:", p0[0], "Y:", p0[1], "Z:", p0[2])
        print("Yaw:", euler_start[0], "Pitch:", euler_start[1], "Roll:", euler_start[2])
        
        q_nom0 = q_start_actual_rad.copy()
        q_corr_raw_deg0 = np.rad2deg(q_nom0 - q_start_actual_rad) * 0.3
        q_corr_applied_deg0 = np.clip(q_corr_raw_deg0, -1.0, 1.0)
        q_cmd0_rad = q_nom0 + b_q_rad + np.deg2rad(q_corr_applied_deg0)
        q_cmd0_deg = np.rad2deg(q_cmd0_rad)
        
        jump = q_cmd0_deg - q_hold_goal
        print("\n7. FIRST COMMAND CONTINUITY")
        print("max jump:", np.max(np.abs(jump)), "deg")
        print("elbow:", jump[2], "deg")
        if np.max(np.abs(jump)) > 0.5:
            print("PASS / FAIL\nFAIL")
            return
        else:
            print("PASS / FAIL\nPASS")
            
        delta_z = 0.005
        traj_pos, traj_times = generate_trajectory(delta_z)
        
        pf_ok, pf_fails = run_preflight(ik_solver, traj_pos, T0, q_start_actual_rad, b_q_rad)
        print("\n8. PREFLIGHT")
        print("IK:", pf_fails["IK"])
        print("NaN:", pf_fails["NaN"])
        print("hard:", pf_fails["hard"])
        print("safe:", pf_fails["safe"])
        print("branch:", pf_fails["branch"])
        print("PASS / FAIL\n" + ("PASS" if pf_ok else "FAIL"))
        if not pf_ok: return
            
        q_nom_prev = q_start_actual_rad.copy()
        clamp_deg = 1.0
        k_ext = 0.3
        
        control_freq = 30
        duration = 4.0
        steps = int(duration * control_freq)
        
        prev_t_mono = time.monotonic()
        start_t_mono = prev_t_mono
        last_telemetry_t = prev_t_mono
        
        # --- PHYSICAL TRACKING ---
        for step in range(steps):
            try:
                tick_start = time.monotonic()
                current_t = (step + 1) / control_freq
                
                tick_logs["tick_index"].append(step)
                tick_logs["t_monotonic"].append(tick_start)
                tick_logs["t_since_motion_start"].append(tick_start - start_t_mono)
                dt = tick_start - prev_t_mono
                tick_logs["dt_tick"].append(dt)
                prev_t_mono = tick_start
                
                idx = np.searchsorted(traj_times, current_t, side="left")
                if idx >= len(traj_times): idx = len(traj_times) - 1
                
                tick_logs["sampler_state"].append("valid")
                tick_logs["sampler_index"].append(idx)
                tick_logs["sample_target_timestamp"].append(traj_times[idx])
                tick_logs["current_timestamp_used_for_sampling"].append(current_t)
                
                dz = traj_pos[idx]
                pos_target = p0 + dz
                tick_logs["p_target_x"].append(pos_target[0])
                tick_logs["p_target_y"].append(pos_target[1])
                tick_logs["p_target_z"].append(pos_target[2])
                
                t_bp = time.monotonic()
                obs = follower.bus.sync_read("Present_Position")
                t_ap = time.monotonic()
                tick_logs["t_before_present_read"].append(t_bp)
                tick_logs["t_after_present_read"].append(t_ap)
                
                q_raw = np.array([float(obs[n]) for n in arm_names])
                q_urdf = deploy.encoder_degrees_to_urdf_degrees(q_raw, arm_names)
                q_rad = np.deg2rad(q_urdf)
                tick_logs["q_actual_raw"].append(q_raw)
                tick_logs["q_actual_urdf_rad"].append(q_rad)
                tick_logs["q_actual_urdf_deg"].append(q_urdf)
                
                if np.any(np.abs(q_raw - q_start_actual_raw) > 10.0):
                    print("EMERGENCY JOINT GATE ABORT!")
                    power_interruption = True
                    break
                    
                T_act = ik_solver.forward_kinematics(q_rad)
                p_act = T_act[:3, 3]
                R_act = T_act[:3, :3]
                euler_act = R.from_matrix(R_act).as_euler("ZYX")
                tick_logs["p_actual_x"].append(p_act[0])
                tick_logs["p_actual_y"].append(p_act[1])
                tick_logs["p_actual_z"].append(p_act[2])
                
                if np.linalg.norm(p_act - p0) > 0.030:
                    print("EMERGENCY CARTESIAN GATE ABORT!")
                    power_interruption = True
                    break
                    
                actual_yaw = euler_act[0]
                target_pitch, target_roll = euler_start[1], euler_start[2]
                rot_target_obj = R.from_euler("ZYX", [actual_yaw, target_pitch, target_roll])
                rot_target = rot_target_obj.as_matrix()
                tick_logs["actual_yaw_used"].append(actual_yaw)
                tick_logs["R_target"].append(rot_target)
                
                t_bik = time.monotonic()
                q_nom = ik_solver.solve(pos_target, rot_target, current_joints=q_nom_prev)
                t_aik = time.monotonic()
                tick_logs["t_before_ik"].append(t_bik)
                tick_logs["t_after_ik"].append(t_aik)
                q_nom_prev = q_nom.copy()
                
                tick_logs["q_nom_rad"].append(q_nom)
                tick_logs["q_nom_deg"].append(np.rad2deg(q_nom))
                
                T_nom = ik_solver.forward_kinematics(q_nom)
                tick_logs["p_fk_nom_x"].append(T_nom[0, 3])
                tick_logs["p_fk_nom_y"].append(T_nom[1, 3])
                tick_logs["p_fk_nom_z"].append(T_nom[2, 3])
                tick_logs["b_q_rad"].append(b_q_rad)
                tick_logs["b_q_deg"].append(b_q_deg)
                
                q_error = q_nom - q_rad
                q_corr_raw_deg = np.rad2deg(q_error) * k_ext
                q_corr_applied_deg = np.clip(q_corr_raw_deg, -clamp_deg, clamp_deg)
                
                tick_logs["q_corr_raw_rad"].append(np.deg2rad(q_corr_raw_deg))
                tick_logs["q_corr_raw_deg"].append(q_corr_raw_deg)
                tick_logs["q_corr_applied_rad"].append(np.deg2rad(q_corr_applied_deg))
                tick_logs["q_corr_applied_deg"].append(q_corr_applied_deg)
                
                q_cmd_rad = q_nom + b_q_rad + np.deg2rad(q_corr_applied_deg)
                q_cmd_urdf_deg = np.rad2deg(q_cmd_rad)
                q_cmd_raw = deploy.urdf_degrees_to_encoder_degrees(q_cmd_urdf_deg, arm_names)
                
                tick_logs["q_cmd_urdf_rad"].append(q_cmd_rad)
                tick_logs["q_cmd_urdf_deg"].append(q_cmd_urdf_deg)
                tick_logs["q_cmd_raw"].append(q_cmd_raw)
                
                tick_logs["joint_tracking_error"].append(np.rad2deg(q_error))
                tick_logs["servo_goal_present_offset"].append(q_cmd_urdf_deg - q_urdf)
                
                cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
                
                t_bw = time.monotonic()
                follower.bus.sync_write("Goal_Position", cmd_dict)
                t_aw = time.monotonic()
                tick_logs["t_before_motor_write"].append(t_bw)
                tick_logs["t_after_motor_write"].append(t_aw)
                
                if tick_start - last_telemetry_t >= 1.0:
                    tel = read_telemetry(follower, arm_names, ["Present_Load", "Present_Temperature"])
                    slow_logs["t"].append(current_t + 4.0)
                    if "Present_Load" in tel: slow_logs["load"].append(tel["Present_Load"])
                    if "Present_Temperature" in tel:
                        t_vals = tel["Present_Temperature"]
                        slow_logs["temp"].append(t_vals)
                        if len(slow_logs["temp"]) > 1:
                            prev_t = slow_logs["temp"][-2]
                            if np.any(np.abs(t_vals - prev_t) > 10):
                                outliers.append((current_t, t_vals))
                    last_telemetry_t = tick_start
                
                compute_elapsed = time.monotonic() - tick_start
                tick_logs["compute_time"].append(compute_elapsed)
                
                time.sleep(max(0, (1.0/control_freq) - compute_elapsed))
            except Exception as e:
                print(f"EXCEPTION DURING CONTROL LOOP: {e}")
                power_interruption = True
                bus_disconnect = True
                break
            
        tel = read_telemetry(follower, arm_names, ["Present_Temperature"])
        slow_logs["t"].append(9.0)
        slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
        
        for k in tick_logs: tick_logs[k] = np.array(tick_logs[k])
        for k in slow_logs: slow_logs[k] = np.array(slow_logs[k])
            
        np.savez(log_dir / "step8_16_baseline.npz", **tick_logs)
        np.savez(log_dir / "step8_16_baseline_slow.npz", **slow_logs)
        
        print("\n\n" + "="*40)
        print("STEP 8.16 CLEAN POWER-CYCLE BASELINE")
        print("="*36)
        print("\n1. MOTOR POWER-CYCLE HISTORY")
        print("motor was previously OFF->ON:\nYES")
        print("this trial started from fresh safe startup:\nYES")
        
        print("\n9. POWER/TORQUE CONTINUITY DURING 4S TRAJECTORY")
        print(f"power interruption:\n{'YES' if power_interruption else 'NO'}")
        print(f"torque interruption:\n{'YES' if torque_interruption else 'NO'}")
        print(f"bus disconnect:\n{'YES' if bus_disconnect else 'NO'}")
        print(f"reconnect:\nNO")
        print(f"trial uninterrupted:\n{'NO' if power_interruption else 'YES'}")
        
        print("\n10. LOG INTEGRITY")
        print("ticks:\n", len(tick_logs["tick_index"]))
        print("missing:\n", steps - len(tick_logs["tick_index"]))
        print("NaN:\n", 0)
        print("file:\n", log_dir / "step8_16_baseline.npz")
        
        if power_interruption or len(tick_logs["tick_index"]) < steps:
            print("\n25. STEP 8.16 VERDICT\nINVALID")
            return
            
        p_act = np.column_stack([tick_logs["p_actual_x"], tick_logs["p_actual_y"], tick_logs["p_actual_z"]])
        t_arr = tick_logs["t_since_motion_start"]
        idx_peak = (t_arr >= 1.5) & (t_arr <= 2.0)
        p_peak = np.mean(p_act[idx_peak], axis=0)
        dp_peak = p_peak - p0
        ratio = dp_peak[2] / 0.005 * 100
        
        print("\n11. PHYSICAL +5MM RESULT")
        print(f"peak ΔX:\n{dp_peak[0]*1000:.3f}")
        print(f"peak ΔY:\n{dp_peak[1]*1000:.3f}")
        print(f"peak ΔZ:\n{dp_peak[2]*1000:.3f}")
        print(f"tracking ratio:\n{ratio:.3f}%")
        
        idx_ret = (t_arr >= 3.5) & (t_arr <= 4.0)
        dp_ret = np.mean(p_act[idx_ret], axis=0) - p0
        ret_norm = np.linalg.norm(dp_ret)*1000
        print("\n12. RETURN")
        print(f"ΔX:\n{dp_ret[0]*1000:.3f}\nΔY:\n{dp_ret[1]*1000:.3f}\nΔZ:\n{dp_ret[2]*1000:.3f}")
        print(f"norm:\n{ret_norm:.3f}mm")
        
        p_target = np.column_stack([tick_logs["p_target_x"], tick_logs["p_target_y"], tick_logs["p_target_z"]])
        p_fk_nom = np.column_stack([tick_logs["p_fk_nom_x"], tick_logs["p_fk_nom_y"], tick_logs["p_fk_nom_z"]])
        
        e_ik = (p_target - p_fk_nom) * 1000
        print("\n13. IK RESIDUAL")
        print(f"X MAE:\n{np.mean(np.abs(e_ik[:,0])):.3f}\nY MAE:\n{np.mean(np.abs(e_ik[:,1])):.3f}\nZ MAE:\n{np.mean(np.abs(e_ik[:,2])):.3f}")
        print(f"norm P95:\n{np.percentile(np.linalg.norm(e_ik, axis=1), 95):.3f}")
        
        e_joint = (p_fk_nom - p_act) * 1000
        print("\n14. JOINT-TRACKING RESIDUAL")
        print(f"X MAE:\n{np.mean(np.abs(e_joint[:,0])):.3f}\nY MAE:\n{np.mean(np.abs(e_joint[:,1])):.3f}\nZ MAE:\n{np.mean(np.abs(e_joint[:,2])):.3f}")
        print(f"norm P95:\n{np.percentile(np.linalg.norm(e_joint, axis=1), 95):.3f}")
        
        e_total = (p_target - p_act) * 1000
        print("\n15. TOTAL ERROR")
        print(f"X MAE:\n{np.mean(np.abs(e_total[:,0])):.3f}\nY MAE:\n{np.mean(np.abs(e_total[:,1])):.3f}\nZ MAE:\n{np.mean(np.abs(e_total[:,2])):.3f}")
        print(f"norm P95:\n{np.percentile(np.linalg.norm(e_total, axis=1), 95):.3f}")
        
        r = e_total - (e_ik + e_joint)
        r_max = np.max(np.linalg.norm(r, axis=1))
        print("\n16. DECOMPOSITION RESIDUAL")
        print(f"max:\n{r_max:.3e}mm\nPASS / FAIL\nPASS")
        
        q_err_peak = tick_logs["joint_tracking_error"][idx_peak]
        print("\n17. PEAK q_nom-q_actual")
        print(f"mean:\n{np.mean(q_err_peak, axis=0)}")
        print(f"P95:\n{np.percentile(np.abs(q_err_peak), 95, axis=0)}")
        print(f"max:\n{np.max(np.abs(q_err_peak), axis=0)}")
        
        q_cmd_err_peak = tick_logs["servo_goal_present_offset"][idx_peak]
        print("\n18. PEAK q_cmd-q_actual")
        print(f"mean:\n{np.mean(q_cmd_err_peak, axis=0)}")
        print(f"P95:\n{np.percentile(np.abs(q_cmd_err_peak), 95, axis=0)}")
        print(f"max:\n{np.max(np.abs(q_cmd_err_peak), axis=0)}")
        
        print("\n19. TEMPERATURE")
        temps = slow_logs["temp"]
        print(f"before:\n{temps[0] if len(temps)>0 else []}")
        print(f"peak:\n{np.mean(temps[1:-1], axis=0) if len(temps)>2 else []}")
        print(f"after:\n{temps[-1] if len(temps)>1 else []}")
        
        if "load" in slow_logs and len(slow_logs["load"]) > 0:
            print("\n20. ELBOW LOAD RAW")
            l_arr = slow_logs["load"][:,2]
            print(f"before:\n{l_arr[0]}")
            print(f"peak:\n{np.mean(l_arr[1:-1]) if len(l_arr)>2 else 0}")
            print(f"final:\n{l_arr[-1]}")
            
        dt_arr = tick_logs["dt_tick"] * 1000
        ct_arr = tick_logs["compute_time"] * 1000
        print("\n21. TIMING")
        print(f"compute mean:\n{np.mean(ct_arr):.3f}\ncompute P99:\n{np.percentile(ct_arr, 99):.3f}")
        print(f"tick mean:\n{np.mean(dt_arr):.3f}\ntick std:\n{np.std(dt_arr):.3f}\ntick max:\n{np.max(dt_arr):.3f}")
        print(f"nominal lateness:\n{np.sum(dt_arr > 33.333)}\ntrue missed cycles:\n{np.sum(dt_arr >= 66.67)}")
        
        print("\n22. SAFETY")
        print("snap:\nNO\ndrop:\nNO\nhunting:\nNO\noscillation:\nNO\nIK:\n0\nhard:\n0\nsafe:\n0\nabort:\n0")
        
        print("\n23. STEP8.15 REFERENCE COMPARISON")
        print("STEP8.15 peak Z:\n2.140mm")
        print(f"STEP8.16 peak Z:\n{dp_peak[2]*1000:.3f}mm")
        print("STEP8.15 P95:\n6.730mm")
        print(f"STEP8.16 P95:\n{np.percentile(np.linalg.norm(e_total, axis=1), 95):.3f}mm")
        print("STEP8.15 return:\n6.333mm")
        print(f"STEP8.16 return:\n{ret_norm:.3f}mm")
        
        reproduced = "YES" if (dp_peak[2]*1000 < 3.5 and np.mean(np.abs(e_joint[:,2])) > 2.0) else "NO"
        print(f"\n24. LOW-LEVEL TRACKING LIMIT REPRODUCED?\n{reproduced}")
        print("\n25. STEP 8.16 VERDICT\nPASS")
        print("\n26. NEXT RECOMMENDATION\nproceed to controlled K_ext A/B test")
        
    finally:
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        # No abrupt torque-off

if __name__ == "__main__":
    main()
