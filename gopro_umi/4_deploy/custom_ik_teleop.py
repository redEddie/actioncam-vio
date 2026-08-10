"""
SO-100 / SO-101 Follower Arm - Simple UMI Final Engine Teleoperation
==============================================================================
[기술 설명: simple_umi_final teleop.py 스케일 매핑 및 1 Euro Filter 엔진 이식]
1. Simple UMI Final Scale & IK Matrix Engine:
   - SCALE_PAN = -4.4
   - IK_MATRIX: lift_for_frontback=-4.4, elbow_for_frontback=4.4, lift_for_updown=4.4
   - JOINT_LIMITS 안전 구속 (80% 안전 마진적용)
2. 1 Euro Filter (떨림 방지 및 미끄러지듯 부드러운 관절 보간):
   - MIN_CUTOFF = 0.6, BETA = 0.07, DCUTOFF = 1.0
3. Real-Time Unbuffered Keyboard (엔터 없이 눌렀을 때 즉시 실시간 반응).
==============================================================================
"""

import time
import sys
import os
import tty
import termios
import select
import math
import numpy as np
from pathlib import Path

# Teleop/lerobot/src 경로 자동 추가
lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction

# ============================================================================
# simple_umi_final 설정 및 1 Euro Filter 클래스 이식
# ============================================================================
JOINT_LIMITS = {
    "shoulder_pan":  (-95.0,  95.0), # 95% 관절 한계 확장
    "shoulder_lift": (-95.0,  95.0),
    "elbow_flex":    (-95.0,  95.0),
    "wrist_flex":    (-75.0,  75.0),
    "wrist_roll":    (-170.0, 170.0),
    "gripper":       (  0.0, 100.0),
}

SCALE_PAN = -4.4

IK_MATRIX = {
    "lift_for_frontback":  -4.4,  # 어깨 2번 관절
    "elbow_for_frontback":  4.4,  # 팔꿈치 3번 관절
    "lift_for_updown":   4.4,     # 어깨 2번 관절
    "elbow_for_updown":  0.0,
}

class OneEuroFilter:
    """simple_umi_final 전용 1 Euro Filter"""
    def __init__(self, mincutoff=0.6, beta=0.07, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def smoothing_factor(self, te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        te = t - self.t_prev
        if te <= 0.0:
            return self.x_prev
            
        self.t_prev = t
        dx = (x - self.x_prev) / te
        edx = self.dx_prev + self.smoothing_factor(te, self.dcutoff) * (dx - self.dx_prev)
        self.dx_prev = edx

        cutoff = self.mincutoff + self.beta * abs(edx)
        x_hat = self.x_prev + self.smoothing_factor(te, cutoff) * (x - self.x_prev)
        self.x_prev = x_hat
        return x_hat

class DirectUnbufferedKeyboard:
    """엔터 키 필요 없이 즉시 반응하는 무버퍼 키보드 매니저"""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def get_char(self):
        if select.select([self.fd], [], [], 0.001) == ([self.fd], [], []):
            try:
                raw_bytes = os.read(self.fd, 1)
                return raw_bytes.decode('utf-8', errors='ignore')
            except Exception:
                return None
        return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def calc_targets(dx_cm, dy_cm, dz_cm, home_joints):
    """simple_umi_final 원본 스케일 매핑 수식"""
    t = dict(home_joints)
    
    # 1번 관절 (Pan)
    t["shoulder_pan"] = home_joints["shoulder_pan"] + dx_cm * SCALE_PAN
    
    # 2번(Lift), 3번(Elbow) 관절 연동 제어 매트릭스
    t["shoulder_lift"] = home_joints["shoulder_lift"] + (dz_cm * IK_MATRIX["lift_for_frontback"]) + (dy_cm * IK_MATRIX["lift_for_updown"])
    t["elbow_flex"]    = home_joints["elbow_flex"]    + (dz_cm * IK_MATRIX["elbow_for_frontback"]) + (dy_cm * IK_MATRIX["elbow_for_updown"])
    
    # 관절 한계 범위 Clamping
    for j, (lo, hi) in JOINT_LIMITS.items():
        if j in t:
            t[j] = clamp(t[j], lo, hi)
    return t

def main():
    print("🤖 SO-100 팔로워 암 연결 및 Simple UMI Final 엔진 초기화 중...")
    follower_config = SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True)
    follower = SO100Follower(follower_config)
    follower.connect()

    arm_motors = [m for m in follower.bus.motors.keys() if "gripper" not in m]

    # 홈 관절 각도 세팅
    home_joints = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
    }

    # 관절별 1 Euro Filter 생성
    filters = {m: OneEuroFilter(mincutoff=0.6, beta=0.07, dcutoff=1.0) for m in arm_motors}

    current_gripper = 0.0
    dx_cm, dy_cm, dz_cm = 0.0, 0.0, 0.0

    print("\n🚀 관절을 중앙(0도) 포지션으로 부드럽게 초기 정렬합니다...")
    current_obs = follower.get_observation()
    for step in range(1, 61):
        action_dict = {}
        for motor_name in follower.bus.motors.keys():
            key = f"{motor_name}.pos"
            if "gripper" in motor_name:
                action_dict[key] = float(current_gripper)
            else:
                start_val = float(current_obs[key])
                action_dict[key] = start_val + (0.0 - start_val) * (step / 60.0)
        follower.send_action(RobotAction(action_dict))
        time.sleep(1.0 / 30.0)

    print("\n⚡ [Simple UMI Final 엔진 실시간 반응형 키보드 텔레옵 시작!]")
    print("  [w/s] : 앞/뒤 전진 후퇴 (dz)")
    print("  [a/d] : 좌/우 이동     (dx)")
    print("  [r/f] : 위/아래 이동   (dy)")
    print("  [z]   : 그리퍼 토글 (0% <-> 100%)")
    print("  [ESC] : 안전 종료\n")

    with DirectUnbufferedKeyboard() as kbd:
        while True:
            t0 = time.perf_counter()
            ch = kbd.get_char()
            
            if ch == '\x1b':  # ESC 키
                break

            key_pressed = False
            if ch:
                ch = ch.lower()
                if ch == 'w': dz_cm -= 0.5; key_pressed = True     # 전진 (물리적 전진 방향 반전 교정)
                elif ch in ('s', 'x'): dz_cm += 0.5; key_pressed = True # 후퇴 (물리적 후퇴 방향 반전 교정)
                elif ch == 'a': dx_cm += 0.5; key_pressed = True    # 좌측 +0.5cm
                elif ch == 'd': dx_cm -= 0.5; key_pressed = True    # 우측 -0.5cm
                elif ch == 'r': dy_cm += 0.5; key_pressed = True    # 위로 +0.5cm
                elif ch == 'f': dy_cm -= 0.5; key_pressed = True    # 아래로 -0.5cm
                elif ch == 'z':
                    current_gripper = 100.0 if current_gripper <= 50.0 else 0.0
                    key_pressed = True

            if key_pressed:
                # 1. simple_umi_final 스케일 매핑 수식으로 관절목표각 계산
                raw_target_joints = calc_targets(dx_cm, dy_cm, dz_cm, home_joints)

                # 2. 1 Euro Filter 부드러운 필터링 적용
                curr_t = time.perf_counter()
                filtered_action = {}
                for m in arm_motors:
                    val = raw_target_joints[m]
                    filtered_val = filters[m](val, curr_t)
                    filtered_action[f"{m}.pos"] = float(filtered_val)
                
                filtered_action["gripper.pos"] = float(current_gripper)

                # 3. 실물 로봇 전송
                follower.send_action(RobotAction(filtered_action))

                sys.stdout.write(f"\r🔥 [UMI 매핑] dX:{dx_cm:+4.1f}cm dY:{dy_cm:+4.1f}cm dZ:{dz_cm:+4.1f}cm | 그리퍼:{current_gripper}%   ")
                sys.stdout.flush()

            time.sleep(max(1.0 / 30.0 - (time.perf_counter() - t0), 0.0))

    print("\n\n종료합니다. 로봇 힘 해제.")
    follower.disconnect()

if __name__ == "__main__":
    main()
