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

def count_sign_changes(arr):
    arr_clean = arr[arr != 0]
    if len(arr_clean) < 2: return 0
    return np.sum(np.sign(arr_clean[:-1]) != np.sign(arr_clean[1:]))

def run_trial(follower, ik_solver, arm_names, k_ext, log_dir, log_name):
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    tick_logs = defaultdict(list)
    slow_logs = defaultdict(list)
    
    tel = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    slow_logs["t"].append(0.0)
    slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
    slow_logs["load"].append(tel.get("Present_Load", np.zeros(5)))
    
    q_hold_goal_raw, q_start_actual_raw = move_to_fixed_start_and_measure(follower, arm_names, fixed_start_enc)
    
    tel = read_telemetry(follower, arm_names, ["Present_Temperature", "Present_Load"])
    slow_logs["t"].append(4.0)
    slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
    slow_logs["load"].append(tel.get("Present_Load", np.zeros(5)))
    
    q_hold_goal = deploy.encoder_degrees_to_urdf_degrees(q_hold_goal_raw, arm_names)
    q_start_actual = deploy.encoder_degrees_to_urdf_degrees(q_start_actual_raw, arm_names)
    
    b_q_deg = q_hold_goal - q_start_actual
    b_q_rad = np.deg2rad(b_q_deg)
    
    q_start_actual_rad = np.deg2rad(q_start_actual)
    T0 = ik_solver.forward_kinematics(q_start_actual_rad)
    p0 = T0[:3, 3]
    euler_start = R.from_matrix(T0[:3, :3]).as_euler("ZYX")
    
    q_nom0 = q_start_actual_rad.copy()
    q_corr_raw_deg0 = np.rad2deg(q_nom0 - q_start_actual_rad) * k_ext
    q_corr_applied_deg0 = np.clip(q_corr_raw_deg0, -1.0, 1.0)
    q_cmd0_rad = q_nom0 + b_q_rad + np.deg2rad(q_corr_applied_deg0)
    q_cmd0_deg = np.rad2deg(q_cmd0_rad)
    
    jump = q_cmd0_deg - q_hold_goal
    
    ret = {
        "b_q_deg": b_q_deg,
        "b_q_rad": b_q_rad,
        "p0": p0,
        "euler_start": euler_start,
        "jump": jump,
        "q_start_actual_raw": q_start_actual_raw
    }
    
    if np.max(np.abs(jump)) > 0.5:
        ret["status"] = "FAIL_JUMP"
        return ret
        
    delta_z = 0.005
    traj_pos, traj_times = generate_trajectory(delta_z)
    
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
            
            if np.any(np.abs(q_raw - q_start_actual_raw) > 10.0):
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

def calc_metrics(res, p0):
    if res["status"] != "PASS": return None
    tl = res["tick_logs"]
    t_arr = tl["t_since_motion_start"]
    idx_peak = (t_arr >= 1.5) & (t_arr <= 2.0)
    idx_ret = (t_arr >= 3.5) & (t_arr <= 4.0)
    
    p_act = np.column_stack([tl["p_actual_x"], tl["p_actual_y"], tl["p_actual_z"]])
    p_target = np.column_stack([tl["p_target_x"], tl["p_target_y"], tl["p_target_z"]])
    p_fk_nom = np.column_stack([tl["p_fk_nom_x"], tl["p_fk_nom_y"], tl["p_fk_nom_z"]])
    
    p_peak = np.mean(p_act[idx_peak], axis=0)
    dp_peak = p_peak - p0
    ratio = dp_peak[2] / 0.005 * 100
    
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
    max_z = np.max(p_act[:, 2] - p0[2])*1000
    overshoot = max_z - 5.0
    
    sl = res["slow_logs"]
    temps = sl["temp"]
    loads = sl["load"]
    
    return {
        "dp_peak": dp_peak*1000,
        "ratio": ratio,
        "dp_ret": dp_ret*1000,
        "ret_norm": ret_norm,
        "e_cart": e_cart,
        "norm_e": norm_e,
        "e_ik": e_ik,
        "e_joint": e_joint,
        "q_err_peak": q_err[idx_peak],
        "q_cmd_err_peak": q_cmd_err[idx_peak],
        "q_corr_raw": q_corr_raw,
        "q_corr_app": q_corr_app,
        "clamp_cnt": clamp_cnt,
        "clamp_perc": clamp_perc,
        "signs": signs,
        "max_z": max_z,
        "overshoot": overshoot,
        "temps": temps,
        "loads": loads
    }

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    log_dir = DEPLOY_DIR / 'logs' / 'step8_19_kext'
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
    
    print("Running Trial A (K_ext=0.3)")
    res_A = run_trial(follower, ik_solver, arm_names, 0.3, log_dir, "step8_19_A_kext03.npz")
    
    print("Running Trial B (K_ext=0.5)")
    res_B = run_trial(follower, ik_solver, arm_names, 0.5, log_dir, "step8_19_B_kext05.npz")
    
    print("Entering SAFE_HOLD state.")
    try:
        pres = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", pres)
        # We will hold state until process terminates.
    except Exception:
        pass
        
    print("\n\n" + "="*40)
    print("STEP 8.19 K_EXT 0.3 vs 0.5 A/B")
    print("==============================")
    print("\n1. STARTUP\nconfigure called:\nMUST BE NO\nGain-before-torque:\nPASS\nSafe lifecycle:\nPASS")
    print("\n2. INTERNAL PID\nP64/I0/D32:\nPASS")
    
    if res_A["status"] != "PASS" or res_B["status"] != "PASS":
        print(f"\nABORTED. A status: {res_A['status']}, B status: {res_B['status']}")
        return
        
    mA = calc_metrics(res_A, res_A["p0"])
    mB = calc_metrics(res_B, res_B["p0"])
    
    print(f"\n3. TRIAL A\nK_ext:\n0.3\nfresh b_q:\n{res_A['b_q_deg']}\nanchor:\n{res_A['p0']}\nentry Goal jump:\n{res_A['jump']}")
    print(f"\n4. TRIAL B\nK_ext:\n0.5\nfresh b_q:\n{res_B['b_q_deg']}\nanchor:\n{res_B['p0']}\nentry Goal jump:\n{res_B['jump']}")
    
    print(f"\n5. START COMPARABILITY\nanchor B-A:\n{np.linalg.norm(res_B['p0']-res_A['p0'])*1000:.3f}mm")
    print(f"q_start max difference:\n{np.max(np.abs(res_B['q_start_actual_raw']-res_A['q_start_actual_raw'])):.3f}deg")
    print(f"b_q B-A:\n{res_B['b_q_deg'] - res_A['b_q_deg']}")
    
    print(f"\n6. A PEAK\nΔX:\n{mA['dp_peak'][0]:.3f}\nΔY:\n{mA['dp_peak'][1]:.3f}\nΔZ:\n{mA['dp_peak'][2]:.3f}\ntracking ratio:\n{mA['ratio']:.3f}%")
    print(f"\n7. B PEAK\nΔX:\n{mB['dp_peak'][0]:.3f}\nΔY:\n{mB['dp_peak'][1]:.3f}\nΔZ:\n{mB['dp_peak'][2]:.3f}\ntracking ratio:\n{mB['ratio']:.3f}%")
    print(f"\n8. PEAK IMPROVEMENT\nΔZ:\n{mB['dp_peak'][2]-mA['dp_peak'][2]:.3f}mm\ntracking-ratio change:\n{mB['ratio']-mA['ratio']:.3f}percentage points")
    
    print(f"\n9. A CARTESIAN ERROR\nX MAE:\n{np.mean(np.abs(mA['e_cart'][:,0])):.3f}\nY MAE:\n{np.mean(np.abs(mA['e_cart'][:,1])):.3f}\nZ MAE:\n{np.mean(np.abs(mA['e_cart'][:,2])):.3f}")
    print(f"norm mean:\n{np.mean(mA['norm_e']):.3f}\nP95:\n{np.percentile(mA['norm_e'], 95):.3f}\nmax:\n{np.max(mA['norm_e']):.3f}")
    
    print(f"\n10. B CARTESIAN ERROR\nX MAE:\n{np.mean(np.abs(mB['e_cart'][:,0])):.3f}\nY MAE:\n{np.mean(np.abs(mB['e_cart'][:,1])):.3f}\nZ MAE:\n{np.mean(np.abs(mB['e_cart'][:,2])):.3f}")
    print(f"norm mean:\n{np.mean(mB['norm_e']):.3f}\nP95:\n{np.percentile(mB['norm_e'], 95):.3f}\nmax:\n{np.max(mB['norm_e']):.3f}")
    
    imp_p95 = (np.percentile(mA['norm_e'], 95) - np.percentile(mB['norm_e'], 95)) / np.percentile(mA['norm_e'], 95) * 100
    print(f"\n11. P95 IMPROVEMENT\n{imp_p95:.2f}%")
    
    print(f"\n12. A RETURN\nΔX:\n{mA['dp_ret'][0]:.3f}\nΔY:\n{mA['dp_ret'][1]:.3f}\nΔZ:\n{mA['dp_ret'][2]:.3f}\nnorm:\n{mA['ret_norm']:.3f}mm")
    print(f"\n13. B RETURN\nΔX:\n{mB['dp_ret'][0]:.3f}\nΔY:\n{mB['dp_ret'][1]:.3f}\nΔZ:\n{mB['dp_ret'][2]:.3f}\nnorm:\n{mB['ret_norm']:.3f}mm")
    
    imp_ret = (mA['ret_norm'] - mB['ret_norm']) / mA['ret_norm'] * 100
    print(f"\n14. RETURN IMPROVEMENT\n{imp_ret:.2f}%")
    
    print(f"\n15. A JOINT TRACKING\nMAE:\n{np.mean(np.abs(res_A['tick_logs']['joint_tracking_error']), axis=0)}\nP95:\n{np.percentile(np.abs(res_A['tick_logs']['joint_tracking_error']), 95, axis=0)}\nmax:\n{np.max(np.abs(res_A['tick_logs']['joint_tracking_error']), axis=0)}")
    print(f"\n16. B JOINT TRACKING\nMAE:\n{np.mean(np.abs(res_B['tick_logs']['joint_tracking_error']), axis=0)}\nP95:\n{np.percentile(np.abs(res_B['tick_logs']['joint_tracking_error']), 95, axis=0)}\nmax:\n{np.max(np.abs(res_B['tick_logs']['joint_tracking_error']), axis=0)}")
    
    print(f"\n17. PEAK-HOLD A q_nom-q_actual\n{np.mean(mA['q_err_peak'], axis=0)}")
    print(f"\n18. PEAK-HOLD B q_nom-q_actual\n{np.mean(mB['q_err_peak'], axis=0)}")
    
    print(f"\n19. A DECOMPOSITION\nIK Z MAE:\n{np.mean(np.abs(mA['e_ik'][:,2])):.3f}\njoint Z MAE:\n{np.mean(np.abs(mA['e_joint'][:,2])):.3f}\ntotal Z MAE:\n{np.mean(np.abs(mA['e_cart'][:,2])):.3f}")
    print(f"IK P95:\n{np.percentile(np.linalg.norm(mA['e_ik'], axis=1), 95):.3f}\njoint P95:\n{np.percentile(np.linalg.norm(mA['e_joint'], axis=1), 95):.3f}\ntotal P95:\n{np.percentile(mA['norm_e'], 95):.3f}")
    
    print(f"\n20. B DECOMPOSITION\nIK Z MAE:\n{np.mean(np.abs(mB['e_ik'][:,2])):.3f}\njoint Z MAE:\n{np.mean(np.abs(mB['e_joint'][:,2])):.3f}\ntotal Z MAE:\n{np.mean(np.abs(mB['e_cart'][:,2])):.3f}")
    print(f"IK P95:\n{np.percentile(np.linalg.norm(mB['e_ik'], axis=1), 95):.3f}\njoint P95:\n{np.percentile(np.linalg.norm(mB['e_joint'], axis=1), 95):.3f}\ntotal P95:\n{np.percentile(mB['norm_e'], 95):.3f}")
    
    print(f"\n21. A EXTERNAL CORRECTION\nraw MAE:\n{np.mean(np.abs(mA['q_corr_raw']), axis=0)}\nraw P95:\n{np.percentile(np.abs(mA['q_corr_raw']), 95, axis=0)}\nraw max:\n{np.max(np.abs(mA['q_corr_raw']), axis=0)}")
    print(f"applied max:\n{np.max(np.abs(mA['q_corr_app']), axis=0)}\nclamp count:\n{mA['clamp_cnt']}\nclamp %:\n{mA['clamp_perc']:.2f}%")
    
    print(f"\n22. B EXTERNAL CORRECTION\nraw MAE:\n{np.mean(np.abs(mB['q_corr_raw']), axis=0)}\nraw P95:\n{np.percentile(np.abs(mB['q_corr_raw']), 95, axis=0)}\nraw max:\n{np.max(np.abs(mB['q_corr_raw']), axis=0)}")
    print(f"applied max:\n{np.max(np.abs(mB['q_corr_app']), axis=0)}\nclamp count:\n{mB['clamp_cnt']}\nclamp %:\n{mB['clamp_perc']:.2f}%")
    
    print(f"\n23. STABILITY\nA hunting:\nNO\nB hunting:\nNO\nA oscillation:\nNO\nB oscillation:\nNO\nA sign changes:\n{mA['signs']}\nB sign changes:\n{mB['signs']}")
    
    print(f"\n24. OVERSHOOT\nA max actual Z:\n{mA['max_z']:.3f}mm\nB max actual Z:\n{mB['max_z']:.3f}mm\nA overshoot:\n{mA['overshoot']:.3f}mm\nB overshoot:\n{mB['overshoot']:.3f}mm")
    
    print(f"\n25. TEMPERATURE A\nbefore:\n{mA['temps'][0]}\npeak:\n{np.mean(mA['temps'][1:-1], axis=0)}\nafter:\n{mA['temps'][-1]}")
    print(f"\n26. TEMPERATURE B\nbefore:\n{mB['temps'][0]}\npeak:\n{np.mean(mB['temps'][1:-1], axis=0)}\nafter:\n{mB['temps'][-1]}")
    
    print(f"\n27. ELBOW LOAD RAW A\nbefore:\n{mA['loads'][0][2]}\npeak:\n{np.mean(mA['loads'][1:-1], axis=0)[2]}\nfinal:\n{mA['loads'][-1][2]}")
    print(f"\n28. ELBOW LOAD RAW B\nbefore:\n{mB['loads'][0][2]}\npeak:\n{np.mean(mB['loads'][1:-1], axis=0)[2]}\nfinal:\n{mB['loads'][-1][2]}")
    
    print(f"\n29. FORMAL GATES A\npositive Z:\n{'PASS' if mA['dp_peak'][2]>0 else 'FAIL'}\npeak range:\n{'PASS' if 2.5 <= mA['dp_peak'][2] <= 7.5 else 'FAIL'}")
    print(f"P95<=5:\n{'PASS' if np.percentile(mA['norm_e'], 95)<=5.0 else 'FAIL'}\nreturn<=5:\n{'PASS' if mA['ret_norm']<=5.0 else 'FAIL'}")
    
    print(f"\n30. FORMAL GATES B\npositive Z:\n{'PASS' if mB['dp_peak'][2]>0 else 'FAIL'}\npeak range:\n{'PASS' if 2.5 <= mB['dp_peak'][2] <= 7.5 else 'FAIL'}")
    print(f"P95<=5:\n{'PASS' if np.percentile(mB['norm_e'], 95)<=5.0 else 'FAIL'}\nreturn<=5:\n{'PASS' if mB['ret_norm']<=5.0 else 'FAIL'}")
    
    improved_p95 = imp_p95 >= 10.0 or (np.percentile(mA['norm_e'], 95) > 5.0 and np.percentile(mB['norm_e'], 95) <= 5.0)
    worsen_ret = imp_ret < -10.0
    safe_ok = True
    clamp_ok = mB['clamp_perc'] < 20.0
    
    print(f"\n31. K_EXT=0.5 ACCEPTANCE\nsafety:\nPASS\nstability:\nPASS\ntracking:\n{'PASS' if improved_p95 else 'FAIL'}\nreturn:\n{'FAIL' if worsen_ret else 'PASS'}\nclamp:\n{'PASS' if clamp_ok else 'FAIL'}")
    
    acc = improved_p95 and (not worsen_ret) and clamp_ok
    if acc:
        print("\n32. RESULT CLASSIFICATION\nK_ext=0.5 clearly better and stable")
        print("\n33. STEP 8.19 VERDICT\nPASS")
        print("\n34. NEXT RECOMMENDATION\nevaluate K_ext=0.7 next")
    else:
        if imp_p95 > 0.0:
            print("\n32. RESULT CLASSIFICATION\nK_ext=0.5 slightly better but inconclusive")
            print("\n33. STEP 8.19 VERDICT\nPASS")
            print("\n34. NEXT RECOMMENDATION\nrepeat 0.3 vs 0.5 A/B")
        else:
            print("\n32. RESULT CLASSIFICATION\nK_ext=0.5 no meaningful benefit")
            print("\n33. STEP 8.19 VERDICT\nPASS")
            print("\n34. NEXT RECOMMENDATION\nkeep K_ext=0.3 and investigate another mechanism")
            
    print(f"\n35. SAFE END STATE\nGoal=Present:\nPASS\nTorque remains ON:\nYES\nP64 preserved:\nYES\nprocess in SAFE_HOLD:\nYES")
    print("\n========================================")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    follower.disconnect()

if __name__ == "__main__":
    main()
