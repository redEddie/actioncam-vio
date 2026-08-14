#!/usr/bin/env python3
"""SO-101 Motor ↔ URDF Zero Offset + SmolVLA Start Pose Measurement Script.

READ ONLY: Reads motor present position without sending any goal positions or torque writes.
"""
import os
import sys
import time
import json
import termios
import tty
import select
import numpy as np
from pathlib import Path
from datetime import datetime

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"

if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
EXISTING_OFFSET_PATH = DEPLOY_DIR / "yawfree_user_defined_joint_zero.json"

def get_single_keypress():
    """Wait for a single keypress (specifically SPACE BAR or Ctrl+C) using termios/tty."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        # Flush any pending stdin bytes first
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                ch = sys.stdin.read(1)
                if ch == '\x03': # Ctrl+C
                    raise KeyboardInterrupt
                if ch == ' ':
                    return 'SPACE'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def read_samples(follower, duration_s=2.0, target_hz=30):
    """Read Present_Position samples without writing any motor goals."""
    samples = {name: [] for name in ARM_JOINTS}
    num_samples = int(duration_s * target_hz)
    interval = 1.0 / target_hz
    
    total_reads = 0
    start_time = time.time()
    for _ in range(num_samples):
        t0 = time.time()
        # READ ONLY: present position in motor degrees
        pres = follower.bus.sync_read("Present_Position")
        total_reads += 1
        for name in ARM_JOINTS:
            samples[name].append(float(pres[name]))
            
        elapsed = time.time() - t0
        time.sleep(max(0.0, interval - elapsed))
        
    stats = {}
    for name in ARM_JOINTS:
        arr = np.array(samples[name], dtype=np.float64)
        stats[name] = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sample_count": len(arr),
            "raw_samples": arr.tolist()
        }
    return stats, total_reads

def main():
    print("=" * 60)
    print(" SO-101 MOTOR ↔ URDF ZERO / START POSE MEASUREMENT (READ ONLY) ")
    print("=" * 60)
    print("PORT                     = /dev/ttyACM0")
    print("READ_ONLY_CONNECTION     = YES")
    print("TORQUE_CHANGED           = NO")
    print("MOTOR_COMMAND_WRITES     = 0")
    print("-" * 60)
    
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.bus.connect(handshake=True)
    
    total_position_reads = 0
    
    try:
        print("\n[STEP 1]")
        print("로봇을 손으로 URDF 기준 다음 ㄱ자 자세로 정확히 맞추십시오:")
        print("  shoulder_pan  = 0°")
        print("  shoulder_lift = 0°")
        print("  elbow_flex    = 0°")
        print("  wrist_flex    = 0°")
        print("  wrist_roll    = 0°")
        print("\n준비가 되면 SPACE BAR를 한 번 누르십시오.")
        print("※ 프로그램은 모터를 움직이지 않습니다. 현재 위치를 읽기만 합니다.")
        
        get_single_keypress()
        print("\n--> [URDF ZERO] 측정 중... (2초간 샘플링)")
        zero_stats, reads1 = read_samples(follower)
        total_position_reads += reads1
        
        motor_zero_at_urdf_zero = [zero_stats[n]["median"] for n in ARM_JOINTS]
        
        print("\n[URDF ZERO CAPTURED]")
        print(f"{'JOINT':15s} {'MEDIAN_DEG':>11s} {'MEAN_DEG':>11s} {'STD_DEG':>9s} {'MIN_DEG':>9s} {'MAX_DEG':>9s}")
        for n in ARM_JOINTS:
            st = zero_stats[n]
            print(f"{n:15s} {st['median']:11.3f} {st['mean']:11.3f} {st['std']:9.4f} {st['min']:9.3f} {st['max']:9.3f}")
            
        print(f"\nMOTOR_ZERO_AT_URDF_ZERO = {[round(x, 3) for x in motor_zero_at_urdf_zero]} deg")
        
        print("\n" + "=" * 60)
        print("[STEP 2]")
        print("이제 로봇을 손으로 실제 SmolVLA inference에 사용할 시작 자세로 맞추십시오.")
        print("준비가 되면 SPACE BAR를 다시 한 번 누르십시오.")
        print("※ 이 자세가 앞으로 deployment 시작 기준이 됩니다.")
        print("※ 프로그램은 모터를 움직이지 않습니다.")
        print("=" * 60)
        
        # Give user time to release key and move robot
        time.sleep(0.5)
        get_single_keypress()
        print("\n--> [SMOLVLA START POSE] 측정 중... (2초간 샘플링)")
        start_stats, reads2 = read_samples(follower)
        total_position_reads += reads2
        
        motor_start = [start_stats[n]["median"] for n in ARM_JOINTS]
        
        print("\n[SMOLVLA START POSE CAPTURED]")
        print(f"{'JOINT':15s} {'MEDIAN_DEG':>11s} {'MEAN_DEG':>11s} {'STD_DEG':>9s} {'MIN_DEG':>9s} {'MAX_DEG':>9s}")
        for n in ARM_JOINTS:
            st = start_stats[n]
            print(f"{n:15s} {st['median']:11.3f} {st['mean']:11.3f} {st['std']:9.4f} {st['min']:9.3f} {st['max']:9.3f}")
            
        # Compute urdf_start = motor_start - motor_zero_at_urdf_zero
        urdf_start = [m_start - m_zero for m_start, m_zero in zip(motor_start, motor_zero_at_urdf_zero)]
        
        print("\n" + "=" * 60)
        print(" FINAL MEASUREMENT SUMMARY ")
        print("=" * 60)
        print(f"{'JOINT':15s} {'MOTOR_ZERO[deg]':>16s} {'MOTOR_START[deg]':>17s} {'URDF_START[deg]':>16s}")
        for i, n in enumerate(ARM_JOINTS):
            print(f"{n:15s} {motor_zero_at_urdf_zero[i]:16.3f} {motor_start[i]:17.3f} {urdf_start[i]:16.3f}")
            
        print("\nVECTORS:")
        print(f"MOTOR_ZERO_AT_URDF_ZERO_DEG = {[round(x, 3) for x in motor_zero_at_urdf_zero]}")
        print(f"MOTOR_START_DEG             = {[round(x, 3) for x in motor_start]}")
        print(f"URDF_START_DEG              = {[round(x, 3) for x in urdf_start]}")
        
        # Read existing offset for comparison
        existing_offset_info = "NONE"
        if EXISTING_OFFSET_PATH.is_file():
            with open(EXISTING_OFFSET_PATH, 'r') as f:
                ex_data = json.load(f)
            ex_zeros = [ex_data["joints"][n]["encoder_zero_deg"] for n in ARM_JOINTS]
            diff = [m_zero - ex_z for m_zero, ex_z in zip(motor_zero_at_urdf_zero, ex_zeros)]
            print("\nEXISTING OFFSET COMPARISON (READ ONLY, NO OVERWRITE):")
            print(f"EXISTING_OFFSET_SOURCE = {EXISTING_OFFSET_PATH}")
            print(f"EXISTING_OFFSET_DEG    = {ex_zeros}")
            print(f"NEW_MOTOR_ZERO_DEG     = {[round(x, 3) for x in motor_zero_at_urdf_zero]}")
            print(f"DIFFERENCE_DEG         = {[round(x, 3) for x in diff]}")
            
        # Save results to JSON
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "joint_order": list(ARM_JOINTS),
            "units": "degrees",
            "motor_zero_at_urdf_zero_deg": [round(x, 4) for x in motor_zero_at_urdf_zero],
            "motor_start_deg": [round(x, 4) for x in motor_start],
            "urdf_start_deg": [round(x, 4) for x in urdf_start],
            "zero_measurement": {
                "median_deg": [round(zero_stats[n]["median"], 4) for n in ARM_JOINTS],
                "mean_deg": [round(zero_stats[n]["mean"], 4) for n in ARM_JOINTS],
                "std_deg": [round(zero_stats[n]["std"], 4) for n in ARM_JOINTS],
                "min_deg": [round(zero_stats[n]["min"], 4) for n in ARM_JOINTS],
                "max_deg": [round(zero_stats[n]["max"], 4) for n in ARM_JOINTS],
                "sample_count": zero_stats[ARM_JOINTS[0]]["sample_count"]
            },
            "start_measurement": {
                "median_deg": [round(start_stats[n]["median"], 4) for n in ARM_JOINTS],
                "mean_deg": [round(start_stats[n]["mean"], 4) for n in ARM_JOINTS],
                "std_deg": [round(start_stats[n]["std"], 4) for n in ARM_JOINTS],
                "min_deg": [round(start_stats[n]["min"], 4) for n in ARM_JOINTS],
                "max_deg": [round(start_stats[n]["max"], 4) for n in ARM_JOINTS],
                "sample_count": start_stats[ARM_JOINTS[0]]["sample_count"]
            },
            "conversion": {
                "motor_to_urdf": "q_urdf = q_motor - motor_zero_at_urdf_zero",
                "urdf_to_motor": "q_motor = q_urdf + motor_zero_at_urdf_zero"
            }
        }
        
        save_path = DEPLOY_DIR / "motor_urdf_zero_and_smolvla_start_measurement.json"
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\nRESULTS SAVED TO: {save_path}")
        
        print("\n" + "=" * 60)
        print(" MOTOR SAFETY VERIFICATION ")
        print("=" * 60)
        print(f"MOTOR_POSITION_READS    = {total_position_reads}")
        print(f"MOTOR_POSITION_WRITES   = 0")
        print(f"GOAL_POSITION_WRITES    = 0")
        print(f"TORQUE_WRITES           = 0")
        print(f"ROBOT_MOVED_BY_SCRIPT   = NO")
        print("=" * 60)
        
    finally:
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)

if __name__ == '__main__':
    main()
