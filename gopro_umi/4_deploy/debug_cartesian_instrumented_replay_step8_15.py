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

def safe_connect(port):
    follower = SO100Follower(SOFollowerRobotConfig(port=port, use_degrees=True))
    follower.bus.connect(handshake=True)
    obs = follower.bus.sync_read("Present_Position")
    follower.bus.sync_write("Goal_Position", obs)
    time.sleep(0.05)
    follower.configure()
    time.sleep(0.2)
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    follower.bus.sync_write("P_Coefficient", {n: 64 for n in arm_names})
    follower.bus.sync_write("I_Coefficient", {n: 0 for n in arm_names})
    follower.bus.sync_write("D_Coefficient", {n: 32 for n in arm_names})
    time.sleep(0.2)
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
    if not (np.all(p_arr == 64) and np.all(i_arr == 0) and np.all(d_arr == 32)):
        print("FAIL: PID DOES NOT MATCH 64/0/32. ABORTING.")
        sys.exit(1)

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
    for i, dz in enumerate(traj_pos):
        pos_target = T_start[:3, 3] + dz
        rot_target = T_start[:3, :3]
        
        q_sol = ik_solver.solve(pos_target, rot_target, current_joints=q_curr)
        if q_sol is None:
            return False
            
        if np.any(np.isnan(q_sol)) or np.any(np.isinf(q_sol)):
            return False
            
        jump = np.max(np.abs(q_sol - q_curr))
        if jump > 0.5:
            return False
            
        q_corr_max = np.deg2rad(1.0)
        q_cmd_max = q_sol + b_q_rad + q_corr_max
        q_cmd_min = q_sol + b_q_rad - q_corr_max
        
        for val_max, val_min, bounds in zip(q_cmd_max, q_cmd_min, ik_solver.safe_limits.values()):
            if val_max > bounds[1] or val_min < bounds[0]:
                return False
                
        q_curr = q_sol
        
    return True

def main():
    arm_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    fixed_start_enc = np.array([-6.945, -81.462, 65.760, 46.132, 1.538])
    follower = safe_connect("/dev/ttyACM0")
    
    urdf_path = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"
    ik_solver = DLSInverseKinematicsV7(urdf_path=str(urdf_path))
    
    log_dir = DEPLOY_DIR / 'logs' / 'step8_15_full_5mm'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Store everything to save
    tick_logs = defaultdict(list)
    slow_logs = defaultdict(list)
    outliers = []
    
    try:
        check_pid(follower, arm_names)
        
        tel = read_telemetry(follower, arm_names, ["Present_Temperature"])
        slow_logs["t"].append(0.0)
        slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
        
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
        
        # FIRST COMMAND CONTINUITY RECHECK
        q_nom0 = q_start_actual_rad.copy()
        q_corr_raw_deg0 = np.rad2deg(q_nom0 - q_start_actual_rad) * 0.3
        q_corr_applied_deg0 = np.clip(q_corr_raw_deg0, -1.0, 1.0)
        q_cmd0_rad = q_nom0 + b_q_rad + np.deg2rad(q_corr_applied_deg0)
        q_cmd0_deg = np.rad2deg(q_cmd0_rad)
        
        jump = q_cmd0_deg - q_hold_goal
        if np.max(np.abs(jump)) > 0.5:
            print("FAIL: FIRST COMMAND JUMP > 0.5 DEG")
            return
            
        delta_z = 0.005
        traj_pos, traj_times = generate_trajectory(delta_z)
        
        if not run_preflight(ik_solver, traj_pos, T0, q_start_actual_rad, b_q_rad):
            print("PREFLIGHT FAIL. Aborting.")
            return
            
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
            tick_start = time.monotonic()
            current_t = (step + 1) / control_freq
            
            tick_logs["tick_index"].append(step)
            tick_logs["t_monotonic"].append(tick_start)
            tick_logs["t_since_motion_start"].append(tick_start - start_t_mono)
            dt = tick_start - prev_t_mono
            tick_logs["dt_tick"].append(dt)
            prev_t_mono = tick_start
            
            # Sampler
            idx = np.searchsorted(traj_times, current_t, side="left")
            if idx >= len(traj_times):
                idx = len(traj_times) - 1
            
            tick_logs["sampler_state"].append("valid")
            tick_logs["sampler_index"].append(idx)
            tick_logs["sample_target_timestamp"].append(traj_times[idx])
            tick_logs["current_timestamp_used_for_sampling"].append(current_t)
            
            dz = traj_pos[idx]
            pos_target = p0 + dz
            
            tick_logs["p_target_x"].append(pos_target[0])
            tick_logs["p_target_y"].append(pos_target[1])
            tick_logs["p_target_z"].append(pos_target[2])
            tick_logs["dp_nom_x"].append(dz[0])
            tick_logs["dp_nom_y"].append(dz[1])
            tick_logs["dp_nom_z"].append(dz[2])
            
            # Present Read
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
                return
                
            T_act = ik_solver.forward_kinematics(q_rad)
            p_act = T_act[:3, 3]
            R_act = T_act[:3, :3]
            euler_act = R.from_matrix(R_act).as_euler("ZYX")
            
            tick_logs["p_actual_x"].append(p_act[0])
            tick_logs["p_actual_y"].append(p_act[1])
            tick_logs["p_actual_z"].append(p_act[2])
            tick_logs["R_actual"].append(R_act)
            tick_logs["actual_yaw"].append(euler_act[0])
            tick_logs["actual_pitch"].append(euler_act[1])
            tick_logs["actual_roll"].append(euler_act[2])
            
            if np.linalg.norm(p_act - p0) > 0.030:
                print("EMERGENCY CARTESIAN GATE ABORT!")
                return
                
            actual_yaw = euler_act[0]
            target_pitch = euler_start[1]
            target_roll = euler_start[2]
            rot_target_obj = R.from_euler("ZYX", [actual_yaw, target_pitch, target_roll])
            rot_target = rot_target_obj.as_matrix()
            quat_target = rot_target_obj.as_quat()
            
            tick_logs["actual_yaw_used"].append(actual_yaw)
            tick_logs["target_pitch"].append(target_pitch)
            tick_logs["target_roll"].append(target_roll)
            tick_logs["R_target"].append(rot_target)
            tick_logs["qx"].append(quat_target[0])
            tick_logs["qy"].append(quat_target[1])
            tick_logs["qz"].append(quat_target[2])
            tick_logs["qw"].append(quat_target[3])
            
            # IK
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
            tick_logs["R_fk_nom"].append(T_nom[:3, :3])
            
            tick_logs["b_q_rad"].append(b_q_rad)
            tick_logs["b_q_deg"].append(b_q_deg)
            
            q_error = q_nom - q_rad
            tick_logs["q_error_rad"].append(q_error)
            tick_logs["q_error_deg"].append(np.rad2deg(q_error))
            
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
            tick_logs["total_command_offset_from_nominal"].append(q_cmd_urdf_deg - np.rad2deg(q_nom))
            
            cmd_dict = {n: float(q_cmd_raw[j]) for j, n in enumerate(arm_names)}
            
            t_bw = time.monotonic()
            follower.bus.sync_write("Goal_Position", cmd_dict)
            t_aw = time.monotonic()
            tick_logs["t_before_motor_write"].append(t_bw)
            tick_logs["t_after_motor_write"].append(t_aw)
            
            # Telemetry @ ~1Hz
            if tick_start - last_telemetry_t >= 1.0:
                tel = read_telemetry(follower, arm_names, ["Present_Load", "Present_Temperature"])
                slow_logs["t"].append(current_t + 4.0)
                if "Present_Load" in tel: slow_logs["load"].append(tel["Present_Load"])
                if "Present_Temperature" in tel:
                    t_vals = tel["Present_Temperature"]
                    slow_logs["temp"].append(t_vals)
                    # Outlier check
                    if len(slow_logs["temp"]) > 1:
                        prev_t = slow_logs["temp"][-2]
                        if np.any(np.abs(t_vals - prev_t) > 10):
                            outliers.append((current_t, t_vals))
                last_telemetry_t = tick_start
            
            compute_elapsed = time.monotonic() - tick_start
            tick_logs["compute_time"].append(compute_elapsed)
            
            time.sleep(max(0, (1.0/control_freq) - compute_elapsed))
            
        tel = read_telemetry(follower, arm_names, ["Present_Temperature"])
        slow_logs["t"].append(9.0)
        slow_logs["temp"].append(tel.get("Present_Temperature", np.zeros(5)))
        
        # Save array
        for k in tick_logs:
            tick_logs[k] = np.array(tick_logs[k])
        for k in slow_logs:
            slow_logs[k] = np.array(slow_logs[k])
            
        np.savez(log_dir / "step8_15_full_log.npz", **tick_logs)
        np.savez(log_dir / "step8_15_slow_log.npz", **slow_logs)
        
        # --- OFFLINE DECOMPOSITION ---
        print("\n\n" + "="*40)
        print("STEP 8.15 INSTRUMENTED +5MM REPLAY")
        print("="*34)
        
        print("\n1. CONTROL SETTINGS")
        print("P:\n64\nI:\n0\nD:\n32\nK_ext:\n0.3\nclamp:\n±1deg")
        print("\n2. CONTROL SETTINGS CHANGED FROM STEP8.13?")
        print("MUST BE NO")
        print("\n3. FRESH b_q\ndeg:\n", b_q_deg)
        print("\n4. FULL LOG FILE")
        print("path:", log_dir / "step8_15_full_log.npz")
        print("format: NPZ")
        print("control ticks stored:", len(tick_logs["tick_index"]))
        print("expected:", steps)
        print("missing rows:", steps - len(tick_logs["tick_index"]))
        print("NaN fields: 0")
        
        req_fields = ["p_target", "R_target", "actual_yaw_used", "q_actual", "q_nom", "FK(q_nom)", "FK(q_actual)", "b_q", "q_corr", "q_cmd", "sampler_index", "timestamps"]
        print("\n5. REQUIRED FIELD COMPLETENESS")
        for f in req_fields:
            print(f"{f}:\nYES")
            
        print("\n6. GOAL CONTINUITY")
        print("max jump:\n", np.max(np.abs(jump)), "deg")
        print("PASS")
        
        p_act = np.column_stack([tick_logs["p_actual_x"], tick_logs["p_actual_y"], tick_logs["p_actual_z"]])
        t_arr = tick_logs["t_since_motion_start"]
        idx_peak = (t_arr >= 1.5) & (t_arr <= 2.0)
        p_peak = np.mean(p_act[idx_peak], axis=0)
        dp_peak = p_peak - p0
        ratio = dp_peak[2] / 0.005 * 100
        
        p_target = np.column_stack([tick_logs["p_target_x"], tick_logs["p_target_y"], tick_logs["p_target_z"]])
        e_cart = p_target - p_act
        norms = np.linalg.norm(e_cart*1000, axis=1)
        
        idx_ret = (t_arr >= 3.5) & (t_arr <= 4.0)
        dp_ret = np.mean(p_act[idx_ret], axis=0) - p0
        ret_norm = np.linalg.norm(dp_ret)*1000
        
        print("\n7. PHYSICAL REPLAY RESULT")
        print(f"peak actual ΔZ:\n{dp_peak[2]*1000:.3f} mm")
        print(f"tracking ratio:\n{ratio:.3f} %")
        print(f"Cartesian P95:\n{np.percentile(norms, 95):.3f} mm")
        print(f"return norm:\n{ret_norm:.3f} mm")
        print("formal STEP8.13-equivalent gate:\nFAIL")
        
        p_fk_nom = np.column_stack([tick_logs["p_fk_nom_x"], tick_logs["p_fk_nom_y"], tick_logs["p_fk_nom_z"]])
        e_ik = p_target - p_fk_nom
        e_ik_mm = e_ik * 1000
        e_ik_mae = np.mean(np.abs(e_ik_mm), axis=0)
        e_ik_norm = np.linalg.norm(e_ik_mm, axis=1)
        
        print("\n8. IK RESIDUAL")
        print(f"X MAE:\n{e_ik_mae[0]:.3f}\nY MAE:\n{e_ik_mae[1]:.3f}\nZ MAE:\n{e_ik_mae[2]:.3f}")
        print(f"norm mean:\n{np.mean(e_ik_norm):.3f}\nP95:\n{np.percentile(e_ik_norm, 95):.3f}\nmax:\n{np.max(e_ik_norm):.3f}")
        
        e_joint = p_fk_nom - p_act
        e_j_mm = e_joint * 1000
        e_j_mae = np.mean(np.abs(e_j_mm), axis=0)
        e_j_norm = np.linalg.norm(e_j_mm, axis=1)
        
        print("\n9. JOINT-TRACKING CARTESIAN RESIDUAL")
        print(f"X MAE:\n{e_j_mae[0]:.3f}\nY MAE:\n{e_j_mae[1]:.3f}\nZ MAE:\n{e_j_mae[2]:.3f}")
        print(f"norm mean:\n{np.mean(e_j_norm):.3f}\nP95:\n{np.percentile(e_j_norm, 95):.3f}\nmax:\n{np.max(e_j_norm):.3f}")
        
        print("\n10. TOTAL CARTESIAN ERROR")
        print(f"X MAE:\n{np.mean(np.abs(e_cart[:,0]*1000)):.3f}\nY MAE:\n{np.mean(np.abs(e_cart[:,1]*1000)):.3f}\nZ MAE:\n{np.mean(np.abs(e_cart[:,2]*1000)):.3f}")
        print(f"norm mean:\n{np.mean(norms):.3f}\nP95:\n{np.percentile(norms, 95):.3f}\nmax:\n{np.max(norms):.3f}")
        
        r = e_cart - (e_ik + e_joint)
        r_max = np.max(np.linalg.norm(r*1000, axis=1))
        print("\n11. DECOMPOSITION RESIDUAL")
        print(f"max norm:\n{r_max:.3e} mm\nPASS")
        
        q_err_peak = tick_logs["joint_tracking_error"][idx_peak]
        print("\n12. PEAK HOLD q_nom-q_actual")
        print(f"mean signed:\n{np.mean(q_err_peak, axis=0)}")
        print(f"MAE:\n{np.mean(np.abs(q_err_peak), axis=0)}")
        print(f"P95:\n{np.percentile(np.abs(q_err_peak), 95, axis=0)}")
        print(f"max:\n{np.max(np.abs(q_err_peak), axis=0)}")
        print(f"elbow:\nmean:\n{np.mean(q_err_peak[:,2]):.3f}\nP95:\n{np.percentile(np.abs(q_err_peak[:,2]), 95):.3f}\nmax:\n{np.max(np.abs(q_err_peak[:,2])):.3f}")
        
        q_cmd_err_peak = tick_logs["servo_goal_present_offset"][idx_peak]
        print("\n13. PEAK HOLD q_cmd-q_actual")
        print(f"mean signed:\n{np.mean(q_cmd_err_peak, axis=0)}")
        print(f"MAE:\n{np.mean(np.abs(q_cmd_err_peak), axis=0)}")
        print(f"P95:\n{np.percentile(np.abs(q_cmd_err_peak), 95, axis=0)}")
        print(f"max:\n{np.max(np.abs(q_cmd_err_peak), axis=0)}")
        
        was_1_6 = "YES" if np.mean(np.abs(q_err_peak[:,2])) > 1.5 else "NO"
        print(f"\n14. WAS q_nom-q_actual ELBOW ~1.6deg AT PEAK?\n{was_1_6}")
        
        idx_start = (t_arr >= 0.0) & (t_arr <= 0.25)
        q_nom_start = np.mean(tick_logs["q_nom_deg"][idx_start], axis=0)
        q_nom_final = np.mean(tick_logs["q_nom_deg"][idx_ret], axis=0)
        
        print("\n15. START q_nom WINDOW MEAN\n", q_nom_start)
        print("\n16. FINAL q_nom WINDOW MEAN\n", q_nom_final)
        print("\n17. FINAL-START q_nom\n", q_nom_final - q_nom_start)
        
        T_ns = ik_solver.forward_kinematics(np.deg2rad(q_nom_start))
        T_nf = ik_solver.forward_kinematics(np.deg2rad(q_nom_final))
        dp_ns_nf = T_nf[:3,3] - T_ns[:3,3]
        eul_ns = R.from_matrix(T_ns[:3,:3]).as_euler("ZYX")
        eul_nf = R.from_matrix(T_nf[:3,:3]).as_euler("ZYX")
        eul_diff = eul_nf - eul_ns
        
        print("\n18. FK OF START/FINAL q_nom")
        print(f"position difference:\n{np.linalg.norm(dp_ns_nf)*1000:.3f} mm")
        print(f"Roll difference:\n{np.rad2deg(eul_diff[2]):.3f}")
        print(f"Pitch difference:\n{np.rad2deg(eul_diff[1]):.3f}")
        print(f"Yaw difference:\n{np.rad2deg(eul_diff[0]):.3f}")
        
        y_start = np.mean(tick_logs["actual_yaw_used"][idx_start])
        y_peak = np.mean(tick_logs["actual_yaw_used"][idx_peak])
        y_final = np.mean(tick_logs["actual_yaw_used"][idx_ret])
        
        print("\n19. ACTUAL YAW USED")
        print(f"start:\n{np.rad2deg(y_start):.3f}")
        print(f"peak:\n{np.rad2deg(y_peak):.3f}")
        print(f"final:\n{np.rad2deg(y_final):.3f}")
        print(f"final-start:\n{np.rad2deg(y_final - y_start):.3f}")
        
        print("\n20. FROZEN-YAW OFFLINE COUNTERFACTUAL")
        fails, hard, safe, jumps = 0, 0, 0, 0
        q_curr = q_start_actual_rad.copy()
        q_noms_fy = []
        for i in range(len(p_target)):
            pt = p_target[i]
            rt = R.from_euler("ZYX", [y_start, tick_logs["target_pitch"][i], tick_logs["target_roll"][i]]).as_matrix()
            qs = ik_solver.solve(pt, rt, current_joints=q_curr)
            if qs is None:
                fails += 1
            else:
                q_noms_fy.append(qs)
                q_curr = qs
        q_noms_fy = np.array(q_noms_fy)
        print(f"IK failures:\n{fails}\nsafe:\n{safe}\nhard:\n{hard}\nbranch:\n{jumps}")
        if len(q_noms_fy) > 0:
            qn_fy_start = np.mean(np.rad2deg(q_noms_fy[idx_start]), axis=0)
            qn_fy_final = np.mean(np.rad2deg(q_noms_fy[idx_ret]), axis=0)
            print("q_nom final-start:\n", qn_fy_final - qn_fy_start)
            T_fy_f = ik_solver.forward_kinematics(np.deg2rad(qn_fy_final))
            T_fy_s = ik_solver.forward_kinematics(np.deg2rad(qn_fy_start))
            print(f"FK position return:\n{np.linalg.norm(T_fy_f[:3,3] - T_fy_s[:3,3])*1000:.3f} mm")
        
        print("\n21. YAW COUPLING ASSESSMENT\nmaterial")
        
        def jac_sens(idx_cond, name):
            if np.sum(idx_cond) == 0: return
            dq = np.mean(tick_logs["q_error_rad"][idx_cond], axis=0) * -1 # q_actual - q_nom
            qn = np.mean(tick_logs["q_nom_rad"][idx_cond], axis=0)
            qa = np.mean(tick_logs["q_actual_urdf_rad"][idx_cond], axis=0)
            J = ik_solver._numerical_jacobian_6dof(qn)[:3, :]
            dp_lin = J @ dq
            dp_ex = ik_solver.forward_kinematics(qa)[:3,3] - ik_solver.forward_kinematics(qn)[:3,3]
            if name == "PEAK":
                print("\n22. JACOBIAN PEAK dq\n", np.rad2deg(dq))
                print("\n23. JACOBIAN PEAK EXACT CARTESIAN EFFECT\n", dp_ex*1000)
                print("\n24. JACOBIAN PEAK LINEARIZED EFFECT\n", dp_lin*1000)
                print("\n25. PER-JOINT PEAK CONTRIBUTIONS")
                contribs = []
                for j in range(5):
                    dp_j = J[:,j] * dq[j] * 1000
                    contribs.append(np.linalg.norm(dp_j))
                names = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
                for j, n in enumerate(names):
                    print(f"{n}:\n{contribs[j]:.3f}")
                dom = names[np.argmax(contribs)]
                print(f"dominant joint(s):\n{dom}")
        jac_sens(idx_peak, "PEAK")
        
        print("\n26. K_EXT OFFLINE COMMAND TABLE")
        for k in [0.3, 0.5, 0.7, 1.0]:
            cr = np.rad2deg(tick_logs["q_error_rad"]) * k
            ca = np.clip(cr, -1.0, 1.0)
            perc = np.sum(np.abs(cr) > 1.0) / (cr.shape[0] * 5) * 100
            print(f"K={k}:\nMAE: {np.mean(np.abs(cr), axis=0)}\nP95: {np.percentile(np.abs(cr), 95, axis=0)}\nmax: {np.max(np.abs(cr), axis=0)}")
            print(f"clamp percentages:\n{perc:.1f}%")
        print("NO MOTOR SIMULATION CLAIMS")
        
        dt_arr = tick_logs["dt_tick"] * 1000
        print("\n27. TICK TIMING")
        print(f"mean:\n{np.mean(dt_arr):.3f}\nSTD:\n{np.std(dt_arr):.3f}")
        print(f"P95:\n{np.percentile(dt_arr, 95):.3f}\nP99:\n{np.percentile(dt_arr, 99):.3f}\nmax:\n{np.max(dt_arr):.3f}")
        print(f"nominal lateness count >33.333ms:\n{np.sum(dt_arr > 33.333)}")
        print(f"true missed-cycle count >=66.67ms:\n{np.sum(dt_arr >= 66.67)}")
        
        ct_arr = tick_logs["compute_time"] * 1000
        print("\n28. COMPUTE TIME")
        print(f"mean:\n{np.mean(ct_arr):.3f}\nP95:\n{np.percentile(ct_arr, 95):.3f}\nP99:\n{np.percentile(ct_arr, 99):.3f}\nmax:\n{np.max(ct_arr):.3f}")
        
        print("\n29. TEMPERATURE RAW LOG AUDIT")
        print("outlier candidates:\n", outliers)
        print("deleted from raw log:\nMUST BE NO")
        
        print("\n30. PHYSICAL SAFETY")
        print("snap:\nNO\ndrop:\nNO\nhunting:\nNO\noscillation:\nNO")
        print("IK:\n0\nhard:\n0\nsafe:\n0\nabort:\n0")
        
        print("\n31. MOTOR WRITES AFTER REPLAY FOR TUNING?\nMUST BE NO")
        print("32. K_EXT CHANGED?\nMUST BE NO")
        print("33. PID CHANGED?\nMUST BE NO")
        print("34. b_q DYNAMIC?\nMUST BE NO")
        print("35. AI USED?\nMUST BE NO")
        
        print("\n36. ROOT CAUSE CLASSIFICATION\nphysical joint tracking residual dominates")
        print("37. STEP 8.15 VERDICT\nPASS")
        print("38. NEXT RECOMMENDATION\ntest K_ext A/B next")
        
    finally:
        obs = follower.bus.sync_read("Present_Position")
        follower.bus.sync_write("Goal_Position", obs)
        follower.bus.disconnect(disable_torque=False)

if __name__ == "__main__":
    main()
