"""
IK Solver v7 자동 검증 — 6-DOF (Dynamic Yaw Replacement) + 90% Safe Limits
========================================================================
v7: 회전(Orientation) 오차까지 완벽하게 고려하되, 
    Yaw는 로봇의 현재 상태로 대체하여 5-DOF 한계를 극복함.

사용법: python3 test_ik_auto_v7.py
"""

import time, sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

lerobot_src = (
    Path(__file__).resolve().parents[4] / "etc" / "third_party" / "lerobot" / "src"
)
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction
from ik.ik_solver_v7 import DLSInverseKinematicsV7

URDF_PATH = str(Path(__file__).resolve().parents[3] / "ik" / "urdf" / "so_arm_with_gopro_final.urdf")
STEP_M = 0.001  # 2mm -> 1mm 로 속도 절반으로 감소
STEPS_PER_DIR = 30  # 스텝 수 증가 (총 이동거리 유지)
LOOP_HZ = 15  # 1초에 15번 전송 (부드럽고 느리게)


def deg2rad(obs, motors):
    return np.array([np.deg2rad(float(obs[f"{m}.pos"])) for m in motors])


def rad2deg_dict(q, motors):
    return {f"{m}.pos": float(np.rad2deg(q[i])) for i, m in enumerate(motors)}


def main():
    print("=" * 70)
    print("🧪 IK v7: 6-DOF (Dynamic Yaw Replacement) + 90% Safe Limits")
    print("=" * 70)

    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    all_m = list(follower.bus.motors.keys())

    ik = DLSInverseKinematicsV7(urdf_path=URDF_PATH, limit_margin_ratio=0.90)

    obs = follower.get_observation()
    cur = deg2rad(obs, arm_m)

    print(f"\n📐 시작 관절각:")
    for i, m in enumerate(arm_m):
        lo, hi = ik.safe_limits[m]
        val = cur[i]
        near = (val >= hi - 0.05 or val <= lo + 0.05)
        print(f"  {m:16s}: {np.rad2deg(val):+7.1f}°  "
              f"[{'⚠️ 한계' if near else '✅'}]")

    # 1. 90% 안전 한계 클램핑 (시작 시)
    print("\n🔄 90% 안전 한계로 초기 클램핑 검사...")
    clamped = ik._clamp_joints_safe(cur)
    any_clamped = False
    for i, m in enumerate(arm_m):
        d = clamped[i] - cur[i]
        if abs(d) > 1e-6:
            print(f"  {m}: {np.rad2deg(cur[i]):+.1f}° → {np.rad2deg(clamped[i]):+.1f}°")
            any_clamped = True

    if any_clamped:
        print("  ⚠️ 관절이 90% 안전 한계 밖에 있어 하드코딩으로 강제 이동합니다...")
        obs_now = follower.get_observation()
        for step in range(1, 21):
            action = {}
            for i, m in enumerate(arm_m):
                s = float(obs_now[f"{m}.pos"])
                t = float(np.rad2deg(clamped[i]))
                action[f"{m}.pos"] = s + (t - s) * (step / 20.0)
            for m in all_m:
                if "gripper" in m:
                    action[f"{m}.pos"] = float(obs_now[f"{m}.pos"])
            follower.send_action(RobotAction(action))
            time.sleep(1.0 / 30.0)
        cur = clamped.copy()
        print("  ✅ 강제 이동 완료.")
    else:
        print("  ✅ 모든 관절이 90% 안전 구역 내에 있습니다.")

    init_tcp_pos = ik.get_tcp_position(cur)
    init_tcp_rot = ik.get_tcp_rotation(cur)
    euler = R.from_matrix(init_tcp_rot).as_euler('zyx', degrees=True)
    
    print(f"\n📍 초기 TCP 위치: X={init_tcp_pos[0]:.4f}  Y={init_tcp_pos[1]:.4f}  Z={init_tcp_pos[2]:.4f}")
    print(f"📐 초기 TCP 자세(Z-Y-X): Yaw={euler[0]:.1f}° Pitch={euler[1]:.1f}° Roll={euler[2]:.1f}°")

    def joint_status(q):
        parts = []
        for i, m in enumerate(arm_m):
            lo, hi = ik.safe_limits[m]
            val = q[i]
            pct = (val - lo) / (hi - lo) * 100
            parts.append(f"{pct:.0f}%")
        return " ".join(parts)

    # 사용자 요청: 10cm씩 앞, 위, 오, 왼 이동 테스트
    # 10cm 이동이므로 스텝 수를 100번으로 늘려서 부드럽게 이동시킵니다. (1mm * 100 = 10cm)
    path_tests = [
        ("전진 10cm (글로벌 +X)", np.array([1.0, 0.0, 0.0]), 100, False),  # 앞 (X축)
        ("상승 10cm (글로벌 +Z)", np.array([0.0, 0.0, 1.0]), 100, False),  # 위 (Z축)
        ("우회전 10cm (글로벌 -Y)", np.array([0.0, -1.0, 0.0]), 100, False), # 오 (오른쪽은 -Y)
        ("좌회전 10cm (글로벌 +Y)", np.array([0.0, 1.0, 0.0]), 100, False),  # 왼 (왼쪽은 +Y)
    ]

    target_pos = init_tcp_pos.copy()
    target_rot = init_tcp_rot.copy()
    results = []

    print(f"\n{'='*70}")
    print(f"🚀 v7: 연속 이동 경로 테스트 (글로벌 Y축 찾기)")
    print(f"   관절범위: [pan lift elbow wrist_f wrist_r] (0~100% = 안전한계 내부)")
    print(f"{'='*70}")

    try:
        for name, move_dir, steps, is_local in path_tests:
            print(f"\n──── {name} ────")
            errs = []
            
            # 이전 테스트에서 물리적 한계에 부딪혀 도달하지 못한 불가능한 목표값이 
            # 누적되지 않도록, 현재 실제 달성한 위치를 새로운 출발점으로 리셋합니다.
            target_pos = ik.get_tcp_position(cur)

            for step in range(steps):
                if is_local:
                    global_dir = target_rot @ move_dir
                else:
                    global_dir = move_dir
                    
                target_pos += global_dir * STEP_M

                t0 = time.perf_counter()
                new_q = ik.solve(target_pos, target_rot, cur)
                ik_ms = (time.perf_counter() - t0) * 1000

                achieved_pos = ik.get_tcp_position(new_q)
                achieved_rot = ik.get_tcp_rotation(new_q)
                
                err_p = np.linalg.norm(target_pos - achieved_pos) * 1000
                
                tgt_euler = R.from_matrix(target_rot).as_euler('zyx', degrees=True)
                ach_euler = R.from_matrix(achieved_rot).as_euler('zyx', degrees=True)
                
                delta = np.rad2deg(new_q - cur)
                errs.append(err_p)

                print(f"  [{step+1:2d}] 🎯 목표위치: X={target_pos[0]:.4f} Y={target_pos[1]:.4f} Z={target_pos[2]:.4f}")
                print(f"       🤖 실제위치: X={achieved_pos[0]:.4f} Y={achieved_pos[1]:.4f} Z={achieved_pos[2]:.4f} (오차: {err_p:.2f}mm)")
                print(f"       🎯 목표자세: Yaw={tgt_euler[0]:.1f}° Pitch={tgt_euler[1]:.1f}° Roll={tgt_euler[2]:.1f}°")
                print(f"       🤖 실제자세: Yaw={ach_euler[0]:.1f}° Pitch={ach_euler[1]:.1f}° Roll={ach_euler[2]:.1f}°")
                print(f"       ⚙️  Δq(관절): [{', '.join(f'{d:+.1f}' for d in delta)}]  안전여유: [{joint_status(new_q)}]")
                print(f"       ⏱️  IK연산: {ik_ms:.1f}ms")
                print(f"       ----------------------------------------------------")

                action = rad2deg_dict(new_q, arm_m)
                try:
                    obs_now = follower.get_observation()
                    for m in all_m:
                        if "gripper" in m:
                            action[f"{m}.pos"] = float(obs_now[f"{m}.pos"])
                except Exception:
                    pass
                follower.send_action(RobotAction(action))
                cur = new_q.copy()
                time.sleep(1.0 / LOOP_HZ)

            avg = np.mean(errs)
            final = errs[-1]
            st = "✅ PASS" if final < 5 else ("⚠️ WARN" if final < 10 else "❌ FAIL")
            results.append((name, avg, final, st))
            print(f"  {st} {name}: 평균={avg:.2f}mm  최종={final:.2f}mm")

    except KeyboardInterrupt:
        print("\n⚠️ 중단")

    follower.disconnect()
    print("\n종료.")

if __name__ == "__main__":
    main()
