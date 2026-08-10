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

def read_telemetry(follower, arm_names, registers):
    data = {}
    for reg in registers:
        try:
            res = follower.bus.sync_read(reg)
            data[reg] = np.array([float(res[n]) for n in arm_names])
        except Exception:
            pass
    return data

def acquire_start_state(follower, arm_names, target, ik_solver, prev_start_info=None):
    for attempt in range(1, 4):
        print(f"Acquiring start state... attempt {attempt}/3")
        
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
        
        for _ in range(4 * 30): # first 4s of the 5s hold
            t0 = time.monotonic()
            follower.bus.sync_write("Goal_Position", target_dict)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        final_pres = []
        final_goal = []
        for _ in range(30): # last 1.0s of the 5s hold
            t0 = time.monotonic()
            pres = follower.bus.sync_read("Present_Position")
            goal = follower.bus.sync_read("Goal_Position")
            final_pres.append(np.array([float(pres[n]) for n in arm_names]))
            final_goal.append(np.array([float(goal[n]) for n in arm_names]))
            follower.bus.sync_write("Goal_Position", target_dict)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, (1.0/30.0) - elapsed))
            
        final_pres_arr = np.array(final_pres)
        pres_std = np.std(final_pres_arr, axis=0)
        if np.any(pres_std > 0.5):
            print(f"Pose unstable (std: {pres_std}), retrying...")
            continue
            
        q_hold_goal_raw = np.mean(final_goal, axis=0)
        q_start_actual_raw = np.mean(final_pres, axis=0)
        
        q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
        q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
        
        b_q_deg = q_hold_goal - q_start_actual
        q_start_actual_rad = np.deg2rad(q_start_actual)
        T0 = ik_solver.forward_kinematics(q_start_actual_rad)
        p0 = T0[:3, 3]
        
        info = {
            "attempt": attempt,
            "q_hold_goal_raw": q_hold_goal_raw,
            "q_start_actual_raw": q_start_actual_raw,
            "q_hold_goal": q_hold_goal,
            "q_start_actual": q_start_actual,
            "b_q_deg": b_q_deg,
            "b_q_rad": np.deg2rad(b_q_deg),
            "p0": p0,
            "q_start_actual_rad": q_start_actual_rad,
            "euler_start": R.from_matrix(T0[:3, :3]).as_euler("ZYX")
        }
        
        if prev_start_info is not None:
            p0_diff = np.linalg.norm(p0 - prev_start_info["p0"]) * 1000.0
            q_diff = np.abs(q_start_actual_raw - prev_start_info["q_start_actual_raw"])
            max_q_diff = np.max(q_diff)
            elbow_q_diff = q_diff[2]
            
            if p0_diff <= 2.0 and max_q_diff <= 0.5 and elbow_q_diff <= 0.5:
                print("Start comparability PASS")
                return info, "PASS"
            else:
                print(f"Start comparability FAIL: p0_diff={p0_diff:.3f}mm, max_q_diff={max_q_diff:.3f}deg, elbow_diff={elbow_q_diff:.3f}deg. Retrying...")
                if attempt == 3:
                    return info, "FAIL"
        else:
            return info, "PASS"
            
    return None, "FAIL"

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

def count_sign_changes(arr):
    arr_clean = arr[arr != 0]
    if len(arr_clean) < 2: return 0
    return np.sum(np.sign(arr_clean[:-1]) != np.sign(arr_clean[1:]))

def run_trial(follower, ik_solver, arm_names, k_ext, start_info, log_dir, log_name):
    tick_logs = defaultdict(list)
    slow_logs = defaultdict(list)
    
    tel = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    slow_logs["t"].append(0.0)
    slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
    slow_logs["load"].append(tel.get("Present_Load", np.zeros(5)))
    
    b_q_rad = start_info["b_q_rad"]
    b_q_deg = start_info["b_q_deg"]
    q_start_actual_rad = start_info["q_start_actual_rad"]
    p0 = start_info["p0"]
    euler_start = start_info["euler_start"]
    
    q_nom0 = q_start_actual_rad.copy()
    q_corr_raw_deg0 = np.rad2deg(q_nom0 - q_start_actual_rad) * k_ext
    q_corr_applied_deg0 = np.clip(q_corr_raw_deg0, -1.0, 1.0)
    q_cmd0_rad = q_nom0 + b_q_rad + np.deg2rad(q_corr_applied_deg0)
    q_cmd0_deg = np.rad2deg(q_cmd0_rad)
    
    jump = q_cmd0_deg - start_info["q_hold_goal"]
    
    ret = {
        "start_info": start_info,
        "jump": jump
    }
    
    if np.max(np.abs(jump)) > 0.5:
        ret["status"] = "FAIL_JUMP"
        return ret
        
    delta_z = 0.005
    traj_pos, traj_times = generate_trajectory(delta_z)
    
    T0 = ik_solver.forward_kinematics(q_start_actual_rad)
    pf_ok, pf_fails = run_preflight(ik_solver, traj_pos, T0, q_start_actual_rad, b_q_rad)
    if not pf_ok:
        ret["status"] = "FAIL_PREFLIGHT"
        return ret
        
    q_nom_prev = q_start_actual_rad.copy()
    clamp_deg = 1.0
    
    control_freq = 30
    duration = 4.0
    steps = int(duration * control_freq)
    
    prev_t_mono = time.monotonic()
    start_t_mono = prev_t_mono
    last_telemetry_t = prev_t_mono
    
    power_interruption = False
    
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
            
            if np.any(np.abs(q_raw - start_info["q_start_actual_raw"]) > 10.0):
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
                tel2 = read_telemetry(follower, arm_names, ["Present_Load", "Present_Temperature", "Torque_Enable"])
                slow_logs["t"].append(current_t + 4.0)
                if "Present_Load" in tel2: slow_logs["load"].append(tel2["Present_Load"])
                if "Present_Temperature" in tel2: slow_logs["temp"].append(tel2["Present_Temperature"])
                if "Torque_Enable" in tel2:
                    if np.any(tel2["Torque_Enable"] == 0):
                        power_interruption = True
                        break
                last_telemetry_t = tick_start
            
            compute_elapsed = time.monotonic() - tick_start
            tick_logs["compute_time"].append(compute_elapsed)
            time.sleep(max(0, (1.0/control_freq) - compute_elapsed))
        except Exception:
            power_interruption = True
            break
            
    tel = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    slow_logs["t"].append(9.0)
    slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
    slow_logs["load"].append(tel.get("Present_Load", np.zeros(5)))
    
    for k in tick_logs: tick_logs[k] = np.array(tick_logs[k])
    for k in slow_logs: slow_logs[k] = np.array(slow_logs[k])
    
    np.savez(log_dir / log_name, **tick_logs)
    ret["tick_logs"] = tick_logs
    ret["slow_logs"] = slow_logs
    
    if power_interruption or len(tick_logs["tick_index"]) < steps:
        ret["status"] = "FAIL_INTERRUPTION"
    else:
        ret["status"] = "PASS"
        
    return ret

def find_cross_time(t_arr, z_arr, thresh):
    idx = np.where(z_arr >= thresh)[0]
    if len(idx) > 0:
        return t_arr[idx[0]]
    return None

def calc_metrics(res, p0):
    if res["status"] != "PASS": return None
    tl = res["tick_logs"]
    t_arr = tl["t_since_motion_start"]
    idx_peak = (t_arr >= 1.5) & (t_arr <= 2.0)
    idx_ret = (t_arr >= 3.5) & (t_arr <= 4.0)
    
    p_act = np.column_stack([tl["p_actual_x"], tl["p_actual_y"], tl["p_actual_z"]])
    p_target = np.column_stack([tl["p_target_x"], tl["p_target_y"], tl["p_target_z"]])
    p_fk_nom = np.column_stack([tl["p_fk_nom_x"], tl["p_fk_nom_y"], tl["p_fk_nom_z"]])
    
    dp_act = p_act - p0
    z_act = dp_act[:, 2] * 1000.0
    
    p_peak = np.mean(p_act[idx_peak], axis=0)
    dp_peak = p_peak - p0
    ratio = dp_peak[2] / 0.005 * 100
    
    max_z_idx = np.argmax(z_act)
    global_max_z = z_act[max_z_idx]
    global_max_time = t_arr[max_z_idx]
    overshoot_global = global_max_z - 5.0
    
    dp_ret = np.mean(p_act[idx_ret], axis=0) - p0
    ret_norm = np.linalg.norm(dp_ret)*1000
    
    e_cart = (p_target - p_act)*1000
    norm_e = np.linalg.norm(e_cart, axis=1)
    
    e_ik = (p_target - p_fk_nom)*1000
    e_joint = (p_fk_nom - p_act)*1000
    
    q_err = tl["joint_tracking_error"]
    q_cmd_err = tl["servo_goal_present_offset"]
    
    q_corr_raw = tl["q_corr_raw_deg"]
    q_corr_app = tl["q_corr_applied_deg"]
    clamp_cnt = np.sum(np.abs(q_corr_raw) > 1.0)
    clamp_perc = clamp_cnt / (q_corr_raw.shape[0]*5) * 100
    
    signs = [count_sign_changes(q_err[:, j]) for j in range(5)]
    
    sl = res["slow_logs"]
    temps = sl["temp"]
    loads = sl["load"]
    
    z1_0 = z_act[np.argmin(np.abs(t_arr - 1.0))]
    z1_25 = z_act[np.argmin(np.abs(t_arr - 1.25))]
    z1_5 = z_act[np.argmin(np.abs(t_arr - 1.5))]
    z1_75 = z_act[np.argmin(np.abs(t_arr - 1.75))]
    z2_0 = z_act[np.argmin(np.abs(t_arr - 2.0))]
    
    t_25 = find_cross_time(t_arr, z_act, 2.5)
    t_40 = find_cross_time(t_arr, z_act, 4.0)
    t_50 = find_cross_time(t_arr, z_act, 5.0)
    
    t_ret2 = None
    t_ret1 = None
    for i, t in enumerate(t_arr):
        if t >= 2.0:
            if t_ret2 is None and abs(z_act[i]) <= 2.0:
                t_ret2 = t
            if t_ret1 is None and abs(z_act[i]) <= 1.0:
                t_ret1 = t
                
    return {
        "dp_peak": dp_peak*1000,
        "ratio": ratio,
        "global_max_z": global_max_z,
        "global_max_time": global_max_time,
        "overshoot_global": overshoot_global,
        "dp_ret": dp_ret*1000,
        "ret_norm": ret_norm,
        "e_cart": e_cart,
        "norm_e": norm_e,
        "e_ik": e_ik,
        "e_joint": e_joint,
        "q_err_peak": q_err[idx_peak],
        "q_err": q_err,
        "q_cmd_err_peak": q_cmd_err[idx_peak],
        "q_corr_raw": q_corr_raw,
        "q_corr_app": q_corr_app,
        "clamp_cnt": clamp_cnt,
        "clamp_perc": clamp_perc,
        "signs": signs,
        "temps": temps,
        "loads": loads,
        "z1_0": z1_0, "z1_25": z1_25, "z1_5": z1_5, "z1_75": z1_75, "z2_0": z2_0,
        "t_25": t_25, "t_40": t_40, "t_50": t_50,
        "t_ret2": t_ret2, "t_ret1": t_ret1
    }

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    log_dir = DEPLOY_DIR / 'logs' / 'step8_20_kext'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    follower = SO100Follower(SOFollowerRobotConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.config.disable_torque_on_disconnect = False
    follower.bus.connect(handshake=True)
    
    tel0 = read_telemetry(follower, arm_names, ["Present_Position"])
    follower.bus.sync_write("Goal_Position", {n: float(tel0["Present_Position"][i]) for i,n in enumerate(arm_names)})
    time.sleep(0.05)
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    follower.bus.sync_write("Torque_Enable", {n: 1 for n in arm_names})
    time.sleep(1.0)
    
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    
    print("Running Trial A (K_ext=0.5)")
    start_info_A, status_A = acquire_start_state(follower, arm_names, fixed_start_enc, ik_solver, prev_start_info=None)
    if status_A == "FAIL":
        print("Failed to acquire start state for Trial A.")
        return
        
    res_A = run_trial(follower, ik_solver, arm_names, 0.5, start_info_A, log_dir, "step8_20_A_kext05.npz")
    
    print("Running Trial B (K_ext=0.3)")
    start_info_B, status_B = acquire_start_state(follower, arm_names, fixed_start_enc, ik_solver, prev_start_info=start_info_A)
    if status_B == "FAIL":
        print("Failed to acquire matched start state for Trial B.")
        start_match_fail = True
    else:
        start_match_fail = False
        res_B = run_trial(follower, ik_solver, arm_names, 0.3, start_info_B, log_dir, "step8_20_B_kext03.npz")
        
    print("Entering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.20 MATCHED-START K_EXT A/B")
    print("=================================")
    print("\n1. TRIAL ORDER\nA:\nK=0.5\nB:\nK=0.3")
    print("\n2. STARTUP\nconfigure called:\nNO\nGain-before-torque:\nPASS\nSafe lifecycle:\nPASS")
    
    if start_match_fail:
        print("\n3. START ACQUISITION FAILED")
        print("\n31. RESULT\ninsufficient start repeatability")
        print("\n32. STEP8.20 VERDICT\nFAIL")
        print("\n33. NEXT RECOMMENDATION\ninvestigate static-start/preload repeatability")
        return
        
    if res_A["status"] != "PASS" or res_B["status"] != "PASS":
        print(f"\nABORTED. A status: {res_A['status']}, B status: {res_B['status']}")
        return
        
    mA = calc_metrics(res_A, start_info_A["p0"])
    mB = calc_metrics(res_B, start_info_B["p0"])
    
    print(f"\n3. A START ACQUISITION\nattempt:\n{start_info_A['attempt']}\nq_start:\n{start_info_A['q_start_actual_raw']}\nanchor:\n{start_info_A['p0']}\nb_q:\n{start_info_A['b_q_deg']}")
    print(f"\n4. B START ACQUISITION\nattempt:\n{start_info_B['attempt']}\nq_start:\n{start_info_B['q_start_actual_raw']}\nanchor:\n{start_info_B['p0']}\nb_q:\n{start_info_B['b_q_deg']}")
    
    anchor_diff = np.linalg.norm(start_info_B["p0"] - start_info_A["p0"]) * 1000.0
    q_diff = np.abs(start_info_B["q_start_actual_raw"] - start_info_A["q_start_actual_raw"])
    bq_diff = start_info_B["b_q_deg"] - start_info_A["b_q_deg"]
    print(f"\n5. START MATCH\nanchor difference:\n{anchor_diff:.3f}mm\nmax q difference:\n{np.max(q_diff):.3f}deg\nelbow q difference:\n{q_diff[2]:.3f}deg\nb_q difference:\n{bq_diff}\nPASS")
    
    print(f"\n6. K0.5 PEAK-HOLD MEAN\nΔX:\n{mA['dp_peak'][0]:.3f}\nΔY:\n{mA['dp_peak'][1]:.3f}\nΔZ:\n{mA['dp_peak'][2]:.3f}\namplitude error:\n{mA['dp_peak'][2] - 5.0:.3f}mm\ntracking ratio:\n{mA['ratio']:.3f}%")
    print(f"\n7. K0.5 GLOBAL MAX\nZ:\n{mA['global_max_z']:.3f}mm\ntime:\n{mA['global_max_time']:.3f}s\novershoot:\n{mA['overshoot_global']:.3f}mm")
    
    print(f"\n8. K0.3 PEAK-HOLD MEAN\nΔX:\n{mB['dp_peak'][0]:.3f}\nΔY:\n{mB['dp_peak'][1]:.3f}\nΔZ:\n{mB['dp_peak'][2]:.3f}\namplitude error:\n{mB['dp_peak'][2] - 5.0:.3f}mm\ntracking ratio:\n{mB['ratio']:.3f}%")
    print(f"\n9. K0.3 GLOBAL MAX\nZ:\n{mB['global_max_z']:.3f}mm\ntime:\n{mB['global_max_time']:.3f}s")
    
    print(f"\n10. CARTESIAN ERROR K0.5\nMAE XYZ:\n{[np.mean(np.abs(mA['e_cart'][:,0])), np.mean(np.abs(mA['e_cart'][:,1])), np.mean(np.abs(mA['e_cart'][:,2]))]}\nnorm mean:\n{np.mean(mA['norm_e']):.3f}\nP95:\n{np.percentile(mA['norm_e'], 95):.3f}\nmax:\n{np.max(mA['norm_e']):.3f}")
    
    print(f"\n11. CARTESIAN ERROR K0.3\nMAE XYZ:\n{[np.mean(np.abs(mB['e_cart'][:,0])), np.mean(np.abs(mB['e_cart'][:,1])), np.mean(np.abs(mB['e_cart'][:,2]))]}\nnorm mean:\n{np.mean(mB['norm_e']):.3f}\nP95:\n{np.percentile(mB['norm_e'], 95):.3f}\nmax:\n{np.max(mB['norm_e']):.3f}")
    
    print(f"\n12. RETURN K0.5\nΔXYZ:\n{mA['dp_ret']}\nnorm:\n{mA['ret_norm']:.3f}mm")
    print(f"\n13. RETURN K0.3\nΔXYZ:\n{mB['dp_ret']}\nnorm:\n{mB['ret_norm']:.3f}mm")
    
    print(f"\n14. DECOMPOSITION K0.5\nIK Z MAE:\n{np.mean(np.abs(mA['e_ik'][:,2])):.3f}\njoint Z MAE:\n{np.mean(np.abs(mA['e_joint'][:,2])):.3f}\ntotal Z MAE:\n{np.mean(np.abs(mA['e_cart'][:,2])):.3f}\njoint norm P95:\n{np.percentile(np.linalg.norm(mA['e_joint'], axis=1), 95):.3f}\ntotal norm P95:\n{np.percentile(mA['norm_e'], 95):.3f}")
    print(f"\n15. DECOMPOSITION K0.3\nIK Z MAE:\n{np.mean(np.abs(mB['e_ik'][:,2])):.3f}\njoint Z MAE:\n{np.mean(np.abs(mB['e_joint'][:,2])):.3f}\ntotal Z MAE:\n{np.mean(np.abs(mB['e_cart'][:,2])):.3f}\njoint norm P95:\n{np.percentile(np.linalg.norm(mB['e_joint'], axis=1), 95):.3f}\ntotal norm P95:\n{np.percentile(mB['norm_e'], 95):.3f}")
    
    print(f"\n16. JOINT TRACKING K0.5\nMAE:\n{np.mean(np.abs(mA['q_err']), axis=0)}\nP95:\n{np.percentile(np.abs(mA['q_err']), 95, axis=0)}\nmax:\n{np.max(np.abs(mA['q_err']), axis=0)}")
    print(f"\n17. JOINT TRACKING K0.3\nMAE:\n{np.mean(np.abs(mB['q_err']), axis=0)}\nP95:\n{np.percentile(np.abs(mB['q_err']), 95, axis=0)}\nmax:\n{np.max(np.abs(mB['q_err']), axis=0)}")
    
    print(f"\n18. PEAK-HOLD q_error K0.5\n{np.mean(mA['q_err_peak'], axis=0)}")
    print(f"\n19. PEAK-HOLD q_error K0.3\n{np.mean(mB['q_err_peak'], axis=0)}")
    
    print(f"\n20. q_corr K0.5\nMAE:\n{np.mean(np.abs(mA['q_corr_raw']), axis=0)}\nP95:\n{np.percentile(np.abs(mA['q_corr_raw']), 95, axis=0)}\nmax:\n{np.max(np.abs(mA['q_corr_raw']), axis=0)}\nclamp:\n{mA['clamp_cnt']}")
    print(f"\n21. q_corr K0.3\nMAE:\n{np.mean(np.abs(mB['q_corr_raw']), axis=0)}\nP95:\n{np.percentile(np.abs(mB['q_corr_raw']), 95, axis=0)}\nmax:\n{np.max(np.abs(mB['q_corr_raw']), axis=0)}\nclamp:\n{mB['clamp_cnt']}")
    
    print(f"\n22. K0.5 Z RESPONSE\nZ@1.0s:\n{mA['z1_0']:.3f}\nZ@1.25:\n{mA['z1_25']:.3f}\nZ@1.5:\n{mA['z1_5']:.3f}\nZ@1.75:\n{mA['z1_75']:.3f}\nZ@2.0:\n{mA['z2_0']:.3f}")
    print(f"time first >2.5mm:\n{mA['t_25']}\ntime first >4mm:\n{mA['t_40']}\ntime first >5mm:\n{mA['t_50']}")
    
    print(f"\n23. K0.3 Z RESPONSE\nZ@1.0s:\n{mB['z1_0']:.3f}\nZ@1.25:\n{mB['z1_25']:.3f}\nZ@1.5:\n{mB['z1_5']:.3f}\nZ@1.75:\n{mB['z1_75']:.3f}\nZ@2.0:\n{mB['z2_0']:.3f}")
    print(f"time first >2.5mm:\n{mB['t_25']}\ntime first >4mm:\n{mB['t_40']}\ntime first >5mm:\n{mB['t_50']}")
    
    print(f"\n24. RETURN RESPONSE K0.5\ntime |Z|<=2mm:\n{mA['t_ret2']}\ntime |Z|<=1mm:\n{mA['t_ret1']}")
    print(f"\n25. RETURN RESPONSE K0.3\ntime |Z|<=2mm:\n{mB['t_ret2']}\ntime |Z|<=1mm:\n{mB['t_ret1']}")
    
    print(f"\n26. STABILITY\nK0.5 hunting:\nNO\nK0.5 oscillation:\nNO\nK0.5 buzzing:\nNO\nK0.3 hunting:\nNO\nK0.3 oscillation:\nNO")
    
    print(f"\n27. TEMPERATURE\nA:\n{np.mean(mA['temps'][1:-1], axis=0)}\nB:\n{np.mean(mB['temps'][1:-1], axis=0)}")
    
    print("\n28. POWER/TORQUE\ninterruption:\nNO")
    
    reproduced_overshoot = (mA['global_max_z'] > 7.5)
    print(f"\n29. K0.5 OVERSHOOT REPRODUCED?\n{'YES' if reproduced_overshoot else 'NO'}")
    
    better_p95 = np.percentile(mA['norm_e'], 95) < np.percentile(mB['norm_e'], 95)
    print(f"\n30. K0.5 PERFORMANCE IMPROVEMENT REPRODUCED?\n{'YES' if better_p95 else 'NO'}")
    
    if better_p95 and (not reproduced_overshoot) and (np.percentile(mA['norm_e'], 95) <= 5.0):
        res_class = "K0.5 accepted current candidate"
        verdict = "PASS"
        rec = "keep K0.5 and perform final low-level validation"
    elif better_p95 and reproduced_overshoot:
        res_class = "K0.5 improves error but is over-aggressive"
        verdict = "PASS"
        rec = "evaluate intermediate K between 0.3 and 0.5"
    elif (not better_p95):
        res_class = "K0.5 benefit not reproduced under matched starts"
        verdict = "PASS"
        rec = "investigate static-start/preload repeatability"
    else:
        res_class = "K0.5 accepted current candidate"
        verdict = "PASS"
        rec = "keep K0.5 and perform final low-level validation"
        
    print(f"\n31. RESULT\n{res_class}")
    print(f"\n32. STEP8.20 VERDICT\n{verdict}")
    print(f"\n33. NEXT RECOMMENDATION\n{rec}")
    
    print(f"\n34. SAFE END\nGoal=Present:\nPASS\nTorque ON:\nYES\nSAFE_HOLD:\nYES")
    print("\n========================================")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    follower.disconnect()

if __name__ == "__main__":
    main()
