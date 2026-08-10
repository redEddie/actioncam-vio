"""
SO-100/SO-101 DLS IK Solver v6 — v1 복원 + 좁은 Soft Repulsion
================================================================
v1~v5 교훈:
  v1 (λ=0.01): 첫 3~7스텝 ✅ 작동 → 반대쪽 한계 충돌로 데드락
  v4 (λ=0.1):  너무 보수적 → 0.06mm/step밖에 못 움직임
  v5 (70%한계): 시작점이 한계 밖 → Soft Repulsion이 전부 죽임

v6 전략:
  v1의 공격적 파라미터를 되살리되,
  URDF 한계에서 5°만 좁은 Soft Repulsion만 추가.
  → 첫 3~7스텝의 좋은 성능 유지
  → 한계 근처에서만 부드럽게 감속 (급정지 대신)
  → Target limiter 제거 (솔버가 자유롭게 풀게)
"""

import numpy as np
from pathlib import Path
import sys

_lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if _lerobot_src.exists() and str(_lerobot_src) not in sys.path:
    sys.path.insert(0, str(_lerobot_src))

from lerobot.model.kinematics import RobotKinematics


class DLSInverseKinematics:
    """
    v1 기반 DLS IK + 좁은 Soft Repulsion.
    """

    # URDF 원본 한계
    PHYSICAL_JOINT_LIMITS_RAD = {
        "shoulder_pan":  (-1.91986, 1.91986),
        "shoulder_lift": (-1.74533, 1.74533),
        "elbow_flex":    (-1.69,    1.69),
        "wrist_flex":    (-1.65806, 1.65806),
        "wrist_roll":    (-2.74385, 2.84121),
    }

    def __init__(
        self,
        urdf_path: str,
        tcp_frame: str = "tcp_link",
        joint_names: list = None,
        damping: float = 0.01,            # v1과 동일: 공격적
        repulsion_margin: float = 0.1,    # ~5.7° — URDF 한계에서만 감속
    ):
        if joint_names is None:
            joint_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll",
            ]

        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.damping = damping
        self.repulsion_margin = repulsion_margin
        self.physical_limits = dict(self.PHYSICAL_JOINT_LIMITS_RAD)

        self._fk = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=tcp_frame,
            joint_names=joint_names,
        )

    def forward_kinematics(self, joints_rad: np.ndarray) -> np.ndarray:
        return self._fk.forward_kinematics(joints_rad)

    def get_tcp_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)[:3, 3].copy()

    def micro_pull(self, joints_rad: np.ndarray, pull_margin: float = 0.05) -> np.ndarray:
        """URDF 한계에서 ~3° 안쪽으로 미세 후퇴."""
        pulled = joints_rad.astype(np.float64).copy()
        for i, name in enumerate(self.joint_names):
            lo, hi = self.physical_limits[name]
            if pulled[i] > hi - pull_margin:
                pulled[i] = hi - pull_margin
            elif pulled[i] < lo + pull_margin:
                pulled[i] = lo + pull_margin
        return pulled

    def _numerical_jacobian_pos(self, joints_rad: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        n = self.n_joints
        J = np.zeros((3, n))
        p0 = self.forward_kinematics(joints_rad)[:3, 3]
        for i in range(n):
            q_plus = joints_rad.copy()
            q_plus[i] += eps
            p_plus = self.forward_kinematics(q_plus)[:3, 3]
            J[:, i] = (p_plus - p0) / eps
        return J

    def _clamp_joints(self, joints_rad: np.ndarray) -> np.ndarray:
        clamped = joints_rad.copy()
        for i, name in enumerate(self.joint_names):
            lo, hi = self.physical_limits[name]
            clamped[i] = np.clip(clamped[i], lo, hi)
        return clamped

    def solve(
        self,
        target_pos: np.ndarray,
        current_joints: np.ndarray,
        max_iter: int = 100,        # v1과 동일
        pos_tol: float = 5e-4,
        max_step_rad: float = 0.15, # v1의 0.3에서 약간 줄임 (8.6°/iter)
    ) -> np.ndarray:
        """
        v1 기반 DLS IK + 좁은 Soft Repulsion.

        URDF 한계에서 5.7° 이내에서만 점진적 감속.
        나머지 범위에서는 v1처럼 자유롭게 움직임.
        """
        q = self._clamp_joints(current_joints.astype(np.float64).copy())
        lam = self.damping
        margin = self.repulsion_margin

        for _ in range(max_iter):
            p_curr = self.forward_kinematics(q)[:3, 3]
            e = target_pos - p_curr
            err_norm = np.linalg.norm(e)

            if err_norm < pos_tol:
                break

            J = self._numerical_jacobian_pos(q)
            lam_adaptive = lam + 0.001 * err_norm

            JJT = J @ J.T
            A = JJT + (lam_adaptive ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, e)

            # ▶ 좁은 Soft Repulsion: URDF 한계 근처에서만 감속
            for i, name in enumerate(self.joint_names):
                lo, hi = self.physical_limits[name]

                # 상한 근처: 증가 방향만 감속
                if q[i] > hi - margin and dq[i] > 0:
                    dist = hi - q[i]
                    scale = np.clip(dist / margin, 0.0, 1.0)
                    dq[i] *= scale

                # 하한 근처: 감소 방향만 감속
                if q[i] < lo + margin and dq[i] < 0:
                    dist = q[i] - lo
                    scale = np.clip(dist / margin, 0.0, 1.0)
                    dq[i] *= scale

            step_norm = np.linalg.norm(dq)
            if step_norm > max_step_rad:
                dq *= max_step_rad / step_norm

            q = self._clamp_joints(q + dq)

        return q

    def solve_from_degrees(self, target_pos, current_joints_deg, **kw):
        q_rad = np.deg2rad(current_joints_deg.astype(np.float64))
        return np.rad2deg(self.solve(target_pos, q_rad, **kw))