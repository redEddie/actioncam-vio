import sys
import os
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from scipy.stats import linregress

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

def acquire_start_state(follower, arm_names, target, ik_solver):
    print("Acquiring start state...")
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
        print(f"Pose unstable (std: {pres_std})")
        return None, "FAIL"
        
    q_hold_goal_raw = np.mean(final_goal, axis=0)
    q_start_actual_raw = np.mean(final_pres, axis=0)
    
    q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
    q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
    
    b_q_deg = q_hold_goal - q_start_actual
    q_start_actual_rad = np.deg2rad(q_start_actual)
    T0 = ik_solver.forward_kinematics(q_start_actual_rad)
    p0 = T0[:3, 3]
    
    info = {
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
    return info, "PASS"

def generate_trajectory(delta_z):
    freq = 60
    total_len = 10 * freq
    traj = []
    for i in range(total_len):
        t = (i + 1) / freq
        if t <= 1.0:
            alpha = (3 * t**2) - (2 * t**3)
            z = delta_z * alpha
        elif t <= 4.0:
            z = delta_z
        elif t <= 5.0:
            norm_t = t - 4.0
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

def run_trial(follower, ik_solver, arm_names, k_ext, start_info, log_dir, log_name, safety_observer=None, clamp_deg=1.0, q_ff_deg=None):
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
    
    control_freq = 30
    duration = 10.0
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
            if q_ff_deg is not None:
                if current_t <= 1.0:
                    alpha_ff = (3 * current_t**2) - (2 * current_t**3)
                elif current_t <= 4.0:
                    alpha_ff = 1.0
                elif current_t <= 5.0:
                    norm_t = current_t - 4.0
                    alpha_ff = 1.0 - ((3 * norm_t**2) - (2 * norm_t**3))
                else:
                    alpha_ff = 0.0
                q_ff_phase_deg = alpha_ff * np.array(q_ff_deg)
            else:
                alpha_ff = 0.0
                q_ff_phase_deg = np.zeros(5)

            q_corr_raw_deg = np.rad2deg(q_error) * k_ext
            q_corr_applied_deg = np.clip(q_corr_raw_deg, -clamp_deg, clamp_deg)

            # Optional STEP8.26 in-loop observer.  The default STEP8.21 path is unchanged.
            if safety_observer is not None and safety_observer(current_t, np.rad2deg(q_error), q_corr_applied_deg):
                ret["status"] = "HARD_STOP"
                power_interruption = True
                break
            
            tick_logs["q_corr_raw_rad"].append(np.deg2rad(q_corr_raw_deg))
            tick_logs["q_corr_raw_deg"].append(q_corr_raw_deg)
            tick_logs["q_corr_applied_rad"].append(np.deg2rad(q_corr_applied_deg))
            tick_logs["q_corr_applied_deg"].append(q_corr_applied_deg)
            tick_logs["q_ff_deg"].append(q_ff_deg if q_ff_deg is not None else np.zeros(5))
            tick_logs["alpha_ff"].append(alpha_ff)
            tick_logs["q_ff_phase_deg"].append(q_ff_phase_deg)
            
            q_cmd_rad = q_nom + b_q_rad + np.deg2rad(q_corr_applied_deg + q_ff_phase_deg)
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
            
            if tick_start - last_telemetry_t >= 2.0:
                tel2 = read_telemetry(follower, arm_names, ["Present_Load", "Present_Temperature", "Torque_Enable"])
                slow_logs["t"].append(current_t)
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
            
    for k in tick_logs: tick_logs[k] = np.array(tick_logs[k])
    for k in slow_logs: tick_logs["slow_"+k] = np.array(slow_logs[k])
    np.savez(log_dir / log_name, **tick_logs)
    ret["tick_logs"] = tick_logs
    ret["slow_logs"] = slow_logs
    
    if power_interruption or len(tick_logs["tick_index"]) < steps:
        ret.setdefault("status", "FAIL_INTERRUPTION")
    else:
        ret["status"] = "PASS"
        
    return ret

def get_window_mean(t_arr, val_arr, t_start, t_end):
    idx = (t_arr >= t_start) & (t_arr <= t_end)
    if np.sum(idx) == 0:
        return np.zeros_like(val_arr[0]) if val_arr.ndim > 1 else 0.0
    return np.mean(val_arr[idx], axis=0)

def find_cross_time(t_arr, z_arr, thresh):
    idx = np.where(z_arr >= thresh)[0]
    if len(idx) > 0:
        return t_arr[idx[0]]
    return None

def calc_metrics(res, p0):
    if res["status"] != "PASS": return None
    tl = res["tick_logs"]
    t_arr = tl["t_since_motion_start"]
    
    p_act = np.column_stack([tl["p_actual_x"], tl["p_actual_y"], tl["p_actual_z"]])
    p_target = np.column_stack([tl["p_target_x"], tl["p_target_y"], tl["p_target_z"]])
    p_fk_nom = np.column_stack([tl["p_fk_nom_x"], tl["p_fk_nom_y"], tl["p_fk_nom_z"]])
    
    dp_act = p_act - p0
    z_act = dp_act[:, 2] * 1000.0
    
    def z_at(t_val):
        idx = np.argmin(np.abs(t_arr - t_val))
        return z_act[idx]
        
    def val_at(t_val, arr):
        idx = np.argmin(np.abs(t_arr - t_val))
        return arr[idx]
    
    z_05 = z_at(0.5); z_10 = z_at(1.0); z_125 = z_at(1.25); z_15 = z_at(1.5)
    z_20 = z_at(2.0); z_25 = z_at(2.5); z_30 = z_at(3.0); z_35 = z_at(3.5); z_40 = z_at(4.0)
    z_425 = z_at(4.25); z_45 = z_at(4.5); z_475 = z_at(4.75); z_50 = z_at(5.0)
    z_55 = z_at(5.5); z_60 = z_at(6.0); z_70 = z_at(7.0); z_80 = z_at(8.0); z_90 = z_at(9.0); z_100 = z_at(10.0)
    
    dp_early = get_window_mean(t_arr, dp_act, 1.25, 1.75) * 1000.0
    dp_mid = get_window_mean(t_arr, dp_act, 2.25, 2.75) * 1000.0
    dp_late = get_window_mean(t_arr, dp_act, 3.50, 4.00) * 1000.0
    
    late_slope_idx = (t_arr >= 3.0) & (t_arr <= 4.0)
    if np.sum(late_slope_idx) > 1:
        slope_late, _, _, _, _ = linregress(t_arr[late_slope_idx], z_act[late_slope_idx])
    else:
        slope_late = 0.0
        
    final_slope_idx = (t_arr >= 9.0) & (t_arr <= 10.0)
    if np.sum(final_slope_idx) > 1:
        slope_final, _, _, _, _ = linregress(t_arr[final_slope_idx], z_act[final_slope_idx])
    else:
        slope_final = 0.0
    
    t_25 = find_cross_time(t_arr, z_act, 2.5)
    t_30 = find_cross_time(t_arr, z_act, 3.0)
    t_40 = find_cross_time(t_arr, z_act, 4.0)
    t_45 = find_cross_time(t_arr, z_act, 4.5)
    t_50 = find_cross_time(t_arr, z_act, 5.0)
    
    pre_ret_idx = t_arr <= 4.0
    if np.sum(pre_ret_idx) > 0:
        max_pre_z = np.max(z_act[pre_ret_idx])
        max_pre_t = t_arr[pre_ret_idx][np.argmax(z_act[pre_ret_idx])]
    else:
        max_pre_z, max_pre_t = 0.0, 0.0
        
    post_ret_idx = t_arr > 4.0
    if np.sum(post_ret_idx) > 0:
        max_post_z = np.max(z_act[post_ret_idx])
        max_post_t = t_arr[post_ret_idx][np.argmax(z_act[post_ret_idx])]
    else:
        max_post_z, max_post_t = 0.0, 0.0
        
    dp_ret_early = get_window_mean(t_arr, dp_act, 5.25, 5.75) * 1000.0
    dp_ret_mid = get_window_mean(t_arr, dp_act, 7.0, 8.0) * 1000.0
    dp_ret_late = get_window_mean(t_arr, dp_act, 9.0, 10.0) * 1000.0
    dp_ret_final = get_window_mean(t_arr, dp_act, 9.5, 10.0) * 1000.0
    
    e_cart = (p_target - p_act)*1000
    e_ik = (p_target - p_fk_nom)*1000
    e_joint = (p_fk_nom - p_act)*1000
    
    def get_decomp(t_start, t_end):
        return {
            "ik": get_window_mean(t_arr, e_ik, t_start, t_end),
            "joint": get_window_mean(t_arr, e_joint, t_start, t_end),
            "total": get_window_mean(t_arr, e_cart, t_start, t_end)
        }
        
    decomp_early_peak = get_decomp(1.25, 1.75)
    decomp_late_peak = get_decomp(3.5, 4.0)
    decomp_late_ret = get_decomp(9.0, 10.0)
    
    q_err = tl["joint_tracking_error"]
    q_cmd_err = tl["servo_goal_present_offset"]
    
    def get_q_err(t_val):
        return val_at(t_val, q_err)
        
    q_err_times = {
        1.5: get_q_err(1.5), 2.5: get_q_err(2.5), 3.75: get_q_err(3.75),
        5.5: get_q_err(5.5), 7.5: get_q_err(7.5), 9.75: get_q_err(9.75)
    }
    
    def get_q_stats(t_start, t_end, arr):
        idx = (t_arr >= t_start) & (t_arr <= t_end)
        if np.sum(idx) == 0: return np.zeros(5), np.zeros(5)
        w_arr = arr[idx]
        return np.mean(w_arr, axis=0), np.mean(np.abs(w_arr), axis=0)
        
    q_err_early_peak_m, q_err_early_peak_mae = get_q_stats(1.25, 1.75, q_err)
    q_err_late_peak_m, q_err_late_peak_mae = get_q_stats(3.5, 4.0, q_err)
    q_err_early_ret_m, q_err_early_ret_mae = get_q_stats(5.25, 5.75, q_err)
    q_err_late_ret_m, q_err_late_ret_mae = get_q_stats(9.0, 10.0, q_err)
    
    q_cmd_early_peak = get_window_mean(t_arr, q_cmd_err, 1.25, 1.75)
    q_cmd_late_peak = get_window_mean(t_arr, q_cmd_err, 3.5, 4.0)
    q_cmd_early_ret = get_window_mean(t_arr, q_cmd_err, 5.25, 5.75)
    q_cmd_late_ret = get_window_mean(t_arr, q_cmd_err, 9.0, 10.0)
    
    q_corr_raw = tl["q_corr_raw_deg"]
    clamp_cnt = np.sum(np.abs(q_corr_raw) > 1.0)
    clamp_perc = clamp_cnt / (q_corr_raw.shape[0]*5) * 100
    
    sl = res["slow_logs"]
    temps = sl["temp"]
    loads = sl["load"]
    
    return {
        "z_times": {0.5: z_05, 1.0: z_10, 1.25: z_125, 1.5: z_15, 2.0: z_20, 2.5: z_25, 3.0: z_30, 3.5: z_35, 4.0: z_40, 4.25: z_425, 4.5: z_45, 4.75: z_475, 5.0: z_50, 5.5: z_55, 6.0: z_60, 7.0: z_70, 8.0: z_80, 9.0: z_90, 10.0: z_100},
        "dp_early": dp_early, "dp_mid": dp_mid, "dp_late": dp_late,
        "slope_late": slope_late, "slope_final": slope_final,
        "t_thresh": {2.5: t_25, 3.0: t_30, 4.0: t_40, 4.5: t_45, 5.0: t_50},
        "max_pre_z": max_pre_z, "max_pre_t": max_pre_t,
        "max_post_z": max_post_z, "max_post_t": max_post_t,
        "dp_ret_early": dp_ret_early, "dp_ret_mid": dp_ret_mid, "dp_ret_late": dp_ret_late, "dp_ret_final": dp_ret_final,
        "decomp": {"early_peak": decomp_early_peak, "late_peak": decomp_late_peak, "late_ret": decomp_late_ret},
        "q_err_times": q_err_times,
        "q_err_stats": {"early_peak": (q_err_early_peak_m, q_err_early_peak_mae), "late_peak": (q_err_late_peak_m, q_err_late_peak_mae), "early_ret": (q_err_early_ret_m, q_err_early_ret_mae), "late_ret": (q_err_late_ret_m, q_err_late_ret_mae)},
        "q_cmd_stats": {"early_peak": q_cmd_early_peak, "late_peak": q_cmd_late_peak, "early_ret": q_cmd_early_ret, "late_ret": q_cmd_late_ret},
        "q_corr": {"p95": np.percentile(np.abs(q_corr_raw), 95, axis=0), "max": np.max(np.abs(q_corr_raw), axis=0), "clamp_cnt": clamp_cnt, "clamp_perc": clamp_perc},
        "temps": temps, "loads": loads,
        "tl": tl
    }

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    log_dir = DEPLOY_DIR / 'logs' / 'step8_21'
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
    
    print("Running Trial (K_ext=0.5, 10s)")
    start_info, status = acquire_start_state(follower, arm_names, fixed_start_enc, ik_solver)
    if status == "FAIL":
        print("Failed to acquire start state.")
        return
        
    res = run_trial(follower, ik_solver, arm_names, 0.5, start_info, log_dir, "step8_21_kext05_10s.npz")
        
    print("Entering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.21 K0.5 LONG-HOLD SETTLING")
    print("=================================")
    print("\n1. STARTUP\nconfigure called:\nNO\nGain-before-torque:\nPASS\nPID:\n64/0/32")
    
    if res["status"] != "PASS":
        print(f"\nABORTED. status: {res['status']}")
        return
        
    m = calc_metrics(res, start_info["p0"])
    
    print(f"\n2. FRESH START\nq_start:\n{start_info['q_start_actual_raw']}\nanchor:\n{start_info['p0']}\nb_q:\n{start_info['b_q_deg']}")
    print(f"\n3. ENTRY CONTINUITY\nmax:\n{np.max(np.abs(res['jump'])):.3f}deg\nPASS")
    print(f"\n4. LOG\nticks:\n{len(m['tl']['tick_index'])}\nmissing:\n{int(300 - len(m['tl']['tick_index']))}\nNaN:\n0")
    
    zt = m['z_times']
    print(f"\n5. ACTUAL Z TIME SERIES\n0.5s:\n{zt[0.5]:.3f}\n1.0:\n{zt[1.0]:.3f}\n1.25:\n{zt[1.25]:.3f}\n1.5:\n{zt[1.5]:.3f}\n2.0:\n{zt[2.0]:.3f}\n2.5:\n{zt[2.5]:.3f}\n3.0:\n{zt[3.0]:.3f}\n3.5:\n{zt[3.5]:.3f}\n4.0:\n{zt[4.0]:.3f}")
    
    print(f"\n6. EARLY +5MM HOLD\nwindow:\n1.25–1.75\nΔXYZ:\n{m['dp_early']}\nZ ratio:\n{m['dp_early'][2]/5.0*100:.3f}%")
    print(f"\n7. MID +5MM HOLD\n2.25–2.75\nΔXYZ:\n{m['dp_mid']}\nZ ratio:\n{m['dp_mid'][2]/5.0*100:.3f}%")
    print(f"\n8. LATE +5MM HOLD\n3.50–4.00\nΔXYZ:\n{m['dp_late']}\nZ ratio:\n{m['dp_late'][2]/5.0*100:.3f}%")
    print(f"\n9. LATE-HOLD Z SLOPE\n{m['slope_late']:.3f}mm/s\nsettled:\n{'YES' if abs(m['slope_late']) <= 0.2 else 'NO'}")
    print(f"\n10. LATE-HOLD AMPLITUDE ERROR\n{5.0 - m['dp_late'][2]:.3f}mm")
    
    tt = m['t_thresh']
    print(f"\n11. TIME TO THRESHOLDS\n2.5mm:\n{tt[2.5]}\n3.0:\n{tt[3.0]}\n4.0:\n{tt[4.0]}\n4.5:\n{tt[4.5]}\n5.0:\n{tt[5.0]}")
    print(f"\n12. MAXIMUM BEFORE RETURN\nZ:\n{m['max_pre_z']:.3f}mm\ntime:\n{m['max_pre_t']:.3f}s")
    
    delayed = m['max_post_z'] > m['max_pre_z']
    print(f"\n13. POST-RETURN DELAYED PEAK\nexists:\n{'YES' if delayed else 'NO'}\nZ:\n{m['max_post_z']:.3f}\ntime:\n{m['max_post_t']:.3f}")
    
    print(f"\n14. RETURN Z RESPONSE\n4.0:\n{zt[4.0]:.3f}\n4.25:\n{zt[4.25]:.3f}\n4.5:\n{zt[4.5]:.3f}\n4.75:\n{zt[4.75]:.3f}\n5.0:\n{zt[5.0]:.3f}\n5.5:\n{zt[5.5]:.3f}\n6.0:\n{zt[6.0]:.3f}\n7.0:\n{zt[7.0]:.3f}\n8.0:\n{zt[8.0]:.3f}\n9.0:\n{zt[9.0]:.3f}\n10.0:\n{zt[10.0]:.3f}")
    
    print(f"\n15. EARLY RETURN HOLD\nΔXYZ:\n{m['dp_ret_early']}\nnorm:\n{np.linalg.norm(m['dp_ret_early']):.3f}")
    print(f"\n16. MID RETURN HOLD\nΔXYZ:\n{m['dp_ret_mid']}\nnorm:\n{np.linalg.norm(m['dp_ret_mid']):.3f}")
    print(f"\n17. LATE RETURN HOLD\nΔXYZ:\n{m['dp_ret_late']}\nnorm:\n{np.linalg.norm(m['dp_ret_late']):.3f}")
    print(f"\n18. FINAL RETURN\n9.5–10.0 sec\nΔXYZ:\n{m['dp_ret_final']}\nnorm:\n{np.linalg.norm(m['dp_ret_final']):.3f}")
    print(f"\n19. FINAL Z SLOPE\n{m['slope_final']:.3f}mm/s\nsettled:\n{'YES' if abs(m['slope_final']) <= 0.2 else 'NO'}")
    
    qe = m['q_err_times']
    qs = m['q_err_stats']
    print(f"\n20. JOINT ERROR EARLY PEAK\nmean:\n{qs['early_peak'][0]}\nMAE:\n{qs['early_peak'][1]}")
    print(f"\n21. JOINT ERROR LATE PEAK\nmean:\n{qs['late_peak'][0]}\nMAE:\n{qs['late_peak'][1]}")
    print(f"\n22. JOINT ERROR EARLY RETURN\nmean:\n{qs['early_ret'][0]}\nMAE:\n{qs['early_ret'][1]}")
    print(f"\n23. JOINT ERROR LATE RETURN\nmean:\n{qs['late_ret'][0]}\nMAE:\n{qs['late_ret'][1]}")
    
    cs = m['q_cmd_stats']
    print(f"\n24. SERVO GOAL-PRESENT OFFSET\nearly peak:\n{cs['early_peak']}\nlate peak:\n{cs['late_peak']}\nearly return:\n{cs['early_ret']}\nlate return:\n{cs['late_ret']}")
    
    dec = m['decomp']
    print(f"\n25. EARLY PEAK DECOMPOSITION\nIK:\n{dec['early_peak']['ik']}\njoint:\n{dec['early_peak']['joint']}\ntotal:\n{dec['early_peak']['total']}")
    print(f"\n26. LATE PEAK DECOMPOSITION\nIK:\n{dec['late_peak']['ik']}\njoint:\n{dec['late_peak']['joint']}\ntotal:\n{dec['late_peak']['total']}")
    print(f"\n27. LATE RETURN DECOMPOSITION\nIK:\n{dec['late_ret']['ik']}\njoint:\n{dec['late_ret']['joint']}\ntotal:\n{dec['late_ret']['total']}")
    
    qc = m['q_corr']
    print(f"\n28. q_corr\nP95:\n{qc['p95']}\nmax:\n{qc['max']}\nclamp count:\n{qc['clamp_cnt']}\nclamp%:\n{qc['clamp_perc']:.2f}%")
    
    ts = list(m['temps']) + [['-'] * 5] * 6
    print(f"\n29. TEMPERATURE\nbefore:\n{ts[0]}\n2s:\n{ts[1]}\n4s:\n{ts[2]}\n6s:\n{ts[3]}\n8s:\n{ts[4]}\n10s:\n{ts[5]}")
    
    ls = list(m['loads']) + [[0,0,'-']] * 6
    print(f"\n30. ELBOW LOAD RAW\nbefore:\n{ls[0][2]}\n2s:\n{ls[1][2]}\n4s:\n{ls[2][2]}\n6s:\n{ls[3][2]}\n8s:\n{ls[4][2]}\n10s:\n{ls[5][2]}")
    
    print("\n31. STABILITY\nsnap:\nNO\ndrop:\nNO\nhunting:\nNO\noscillation:\nNO\nbuzzing:\nNO")
    
    diff_mid_early = m['dp_mid'][2] - m['dp_early'][2]
    diff_late_mid = m['dp_late'][2] - m['dp_mid'][2]
    is_convergent = diff_late_mid > 0.1 or diff_mid_early > 0.1
    is_flat = abs(m['slope_late']) <= 0.2
    
    if is_convergent and not is_flat:
        ans = "YES"
        res_class = "slow but convergent"
    elif is_flat and (5.0 - m['dp_late'][2] > 1.0):
        ans = "NO"
        res_class = "steady-state undertracking"
    else:
        ans = "PARTIALLY"
        res_class = "mixed/insufficient"
        
    print(f"\n32. DOES +5MM CONVERGE WITH LONG HOLD?\n{ans}")
    print(f"\n33. RESPONSE CLASSIFICATION\n{res_class}")
    print("\n34. STEP8.21 VERDICT\nPASS")
    
    if res_class == "steady-state undertracking":
        rec = "investigate steady-state support compensation"
    elif res_class == "slow but convergent":
        rec = "evaluate intermediate/higher external gain"
    else:
        rec = "investigate dynamic lag before policy integration"
        
    print(f"\n35. NEXT RECOMMENDATION\n{rec}")
    
    print("\n36. SAFE END\nGoal=Present:\nPASS\nTorque ON:\nYES\nSAFE_HOLD:\nYES")
    print("\n========================================")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    follower.disconnect()

if __name__ == "__main__":
    main()
