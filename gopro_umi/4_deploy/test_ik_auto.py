"""
IK Solver v6 자동 검증 — v1 복원 + Soft Repulsion
===================================================
v6은 v1처럼 공격적이지만, URDF 한계 근처에서만 감속.
settle 프레임 없이 빠른 테스트.

사용법: python3 test_ik_auto.py
"""

import time, sys
import numpy as np
from pathlib import Path

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction
from ik_solver import DLSInverseKinematics

URDF_PATH = "URDF/so_arm_with_gopro_final.urdf"
STEP_M = 0.002
STEPS_PER_DIR = 15
LOOP_HZ = 10


def deg2rad(obs, motors):
    return np.array([np.deg2rad(float(obs[f"{m}.pos"])) for m in motors])


def rad2deg_dict(q, motors):
    return {f"{m}.pos": float(np.rad2deg(q[i])) for i, m in enumerate(motors)}


def main():
    print("=" * 65)
    print("🧪 IK v6: v1 복원 + 좁은 Soft Repulsion (5.7°)")
    print("=" * 65)

    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    all_m = list(follower.bus.motors.keys())

    ik = DLSInverseKinematics(urdf_path=URDF_PATH)

    obs = follower.get_observation()
    cur = deg2rad(obs, arm_m)

    print(f"\n📐 시작 관절각:")
    for i, m in enumerate(arm_m):
        lo, hi = ik.physical_limits[m]
        val = cur[i]
        near = (val >= hi - 0.1 or val <= lo + 0.1)
        print(f"  {m:16s}: {np.rad2deg(val):+7.1f}°  "
              f"[{'⚠️ 한계' if near else '✅'}]")

    # Micro-Pull
    print("\n🔄 Micro-Pull...")
    pulled = ik.micro_pull(cur)
    any_pulled = False
    for i, m in enumerate(arm_m):
        d = pulled[i] - cur[i]
        if abs(d) > 1e-6:
            print(f"  {m}: {np.rad2deg(cur[i]):+.1f}° → {np.rad2deg(pulled[i]):+.1f}°")
            any_pulled = True

    if any_pulled:
        obs_now = follower.get_observation()
        for step in range(1, 21):
            action = {}
            for i, m in enumerate(arm_m):
                s = float(obs_now[f"{m}.pos"])
                t = float(np.rad2deg(pulled[i]))
                action[f"{m}.pos"] = s + (t - s) * (step / 20.0)
            for m in all_m:
                if "gripper" in m:
                    action[f"{m}.pos"] = float(obs_now[f"{m}.pos"])
            follower.send_action(RobotAction(action))
            time.sleep(1.0 / 30.0)
        cur = pulled.copy()
        print("  ✅ 완료.")

    init_tcp = ik.get_tcp_position(cur)
    print(f"\n📍 초기 TCP: X={init_tcp[0]:.4f}  Y={init_tcp[1]:.4f}  Z={init_tcp[2]:.4f}")

    # 관절 상태 모니터링 함수
    def joint_status(q):
        parts = []
        for i, m in enumerate(arm_m):
            lo, hi = ik.physical_limits[m]
            val = q[i]
            pct = (val - lo) / (hi - lo) * 100  # 0%=하한, 100%=상한
            parts.append(f"{pct:.0f}%")
        return " ".join(parts)

    tests = [
        ("전진 (X+)", 0, +1),
        ("후퇴 (X-)", 0, -1),
        ("좌측 (Y+)", 1, +1),
        ("우측 (Y-)", 1, -1),
        ("상승 (Z+)", 2, +1),
        ("하강 (Z-)", 2, -1),
    ]

    target = init_tcp.copy()
    results = []

    print(f"\n{'='*65}")
    print(f"🚀 v6: λ=0.01, 100iter, max_step=0.15, repulsion=5.7°")
    print(f"   관절범위: [pan lift elbow wrist_f wrist_r] (0%=하한, 100%=상한)")
    print(f"{'='*65}")

    try:
        for name, axis, sign in tests:
            print(f"\n──── {name} ────")
            errs = []

            for step in range(STEPS_PER_DIR):
                target[axis] += sign * STEP_M

                t0 = time.perf_counter()
                new_q = ik.solve(target, cur)
                ik_ms = (time.perf_counter() - t0) * 1000

                achieved = ik.get_tcp_position(new_q)
                err = np.linalg.norm(target - achieved) * 1000
                move = np.linalg.norm(achieved - ik.get_tcp_position(cur)) * 1000
                delta = np.rad2deg(new_q - cur)
                errs.append(err)

                print(f"  [{step+1:2d}] "
                      f"오차={err:.2f}mm  이동={move:.2f}mm  "
                      f"IK={ik_ms:.1f}ms  "
                      f"Δq=[{', '.join(f'{d:+.1f}' for d in delta)}]  "
                      f"범위=[{joint_status(new_q)}]")

                # 로봇 전송
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

            # 복귀
            print(f"  ← 복귀...")
            for step in range(STEPS_PER_DIR):
                target[axis] -= sign * STEP_M
                new_q = ik.solve(target, cur)
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

    print(f"\n{'='*65}")
    print("📊 결과")
    print(f"{'='*65}")
    for n, a, f, s in results:
        print(f"  {n:12s}: 평균={a:.2f}mm  최종={f:.2f}mm  {s}")
    pc = sum(1 for _, _, f, _ in results if f < 5)
    print(f"\n  통과: {pc}/{len(results)}")

    follower.disconnect()
    print("종료.")


if __name__ == "__main__":
    main()
