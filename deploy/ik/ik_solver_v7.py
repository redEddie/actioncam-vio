"""
SO-100/SO-101 DLS IK Solver v8 — 5D task (XYZ + Roll/Pitch) + free Yaw
===================================================================================
특징:
1. 5D task (X, Y, Z, Roll, Pitch)를 추종하는 5x5 task Jacobian.
2. Robot-base +Z 방향 Yaw는 자유도로 남겨 두고 task에서 제외한다.
3. 90% 안전 관절 한계: 물리적 한계의 90% 범위만 사용. 
   시작 시 이 범위를 벗어나면 즉시 가장 가까운 안전 한계로 하드코딩(클램핑)하여 시작.
"""

import numpy as np
from pathlib import Path
import sys
from scipy.spatial.transform import Rotation as R

_lerobot_src = (
    Path(__file__).resolve().parents[2] / "etc" / "third_party" / "lerobot" / "src"
)
if _lerobot_src.exists() and str(_lerobot_src) not in sys.path:
    sys.path.insert(0, str(_lerobot_src))

from lerobot.model.kinematics import RobotKinematics


class DLSInverseKinematicsV7:
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
        damping: float = 0.01,
        limit_margin_ratio: float = 0.98, # 98% 한계 사용
    ):
        if joint_names is None:
            joint_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll",
            ]

        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.damping = damping

        # 물리적 한계의 90%를 안전 한계로 설정
        self.safe_limits = {}
        for k, (lo, hi) in self.PHYSICAL_JOINT_LIMITS_RAD.items():
            center = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            safe_half = half_range * limit_margin_ratio
            self.safe_limits[k] = (center - safe_half, center + safe_half)

        self._fk = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=tcp_frame,
            joint_names=joint_names,
        )
        self.last_position_error = float("inf")
        self.last_orientation_error = float("inf")
        self.last_iterations = 0
        self.last_converged = False

    def forward_kinematics(self, joints_rad: np.ndarray) -> np.ndarray:
        # LEROBOT 버그/특징 해결: _fk.forward_kinematics는 무조건 '도(degree)'를 입력으로 받아서 내부에서 라디안으로 바꿉니다!
        # 우리가 라디안을 그냥 주면 라디안->라디안으로 두 번 변환되어서 모든 각도가 1/57로 줄어들어 0도 근처로 고정되는 치명적 버그가 있었습니다.
        return self._fk.forward_kinematics(np.rad2deg(joints_rad))

    def get_tcp_pose(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)

    def get_tcp_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)[:3, 3].copy()
        
    def get_tcp_rotation(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)[:3, :3].copy()

    def _clamp_joints_safe(self, joints_rad: np.ndarray) -> np.ndarray:
        """90% 안전 한계 내로 강제 클램핑"""
        clamped = joints_rad.copy()
        for i, name in enumerate(self.joint_names):
            lo, hi = self.safe_limits[name]
            clamped[i] = np.clip(clamped[i], lo, hi)
        return clamped

    def _numerical_jacobian_6dof(self, joints_rad: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """6x5 Jacobian 매트릭스 계산 (Position 3 + Rotation 3)"""
        n = self.n_joints
        J = np.zeros((6, n))

        T0 = self.forward_kinematics(joints_rad)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        for i in range(n):
            q_plus = joints_rad.copy()
            q_plus[i] += eps
            T_plus = self.forward_kinematics(q_plus)
            p_plus = T_plus[:3, 3]
            R_plus = T_plus[:3, :3]

            # Position derivative
            J[:3, i] = (p_plus - p0) / eps

            # Orientation derivative (Angular velocity vector)
            R_delta = R_plus @ R0.T
            w = R.from_matrix(R_delta).as_rotvec() / eps
            J[3:, i] = w

        return J

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot_matrix: np.ndarray,
        current_joints: np.ndarray,
        max_iter: int = 100,
        pos_tol: float = 5e-4,
        rot_tol: float = 1e-3,
        max_step_rad: float = 0.15,
    ) -> np.ndarray:
        """Solve the 5D task ``XYZ + Roll + Pitch`` while leaving Yaw free.

        The relative rotation vector is expressed in the robot-base/world
        frame. Its X/Y components are constrained; the Z/Yaw component is
        omitted from both the error and Jacobian task.
        """
        # 1. 시작 시점에서 관절이 한계를 넘었다면 90% 안전 한계로 즉시 클램핑 (하드코딩)
        q = self._clamp_joints_safe(current_joints.astype(np.float64).copy())
        lam = self.damping
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot_matrix = np.asarray(target_rot_matrix, dtype=np.float64).reshape(3, 3)
        self.last_converged = False

        for iteration in range(1, max_iter + 1):
            T_curr = self.forward_kinematics(q)
            p_curr = T_curr[:3, 3]
            R_curr = T_curr[:3, :3]

            e_pos = target_pos - p_curr

            # 회전 에러 행렬: 타겟과 현재 자세의 회전 차이 (Global Frame)
            R_err = target_rot_matrix @ R_curr.T
            
            e_rot = R.from_matrix(R_err).as_rotvec()
            # 5D task: XYZ + Roll/Pitch. Global Z/Yaw is free.
            e = np.concatenate([e_pos, e_rot[:2]])
            err_norm_pos = np.linalg.norm(e_pos)
            err_norm_rot = np.linalg.norm(e_rot[:2])
            self.last_position_error = float(err_norm_pos)
            self.last_orientation_error = float(err_norm_rot)
            self.last_iterations = iteration

            if err_norm_pos < pos_tol and err_norm_rot < rot_tol:
                self.last_converged = True
                break

            J = self._numerical_jacobian_6dof(q)
            J_task = np.vstack([J[:3, :], J[3:5, :]])

            lam_adaptive = lam + 0.001 * err_norm_pos
            JJT = J_task @ J_task.T
            A = JJT + (lam_adaptive ** 2) * np.eye(5)

            dq = J_task.T @ np.linalg.solve(A, e)

            step_norm = np.linalg.norm(dq)
            if step_norm > max_step_rad:
                dq *= max_step_rad / step_norm

            # 한계 클램핑 적용 (항상 90% 한계 안에서만 작동하도록)
            q = self._clamp_joints_safe(q + dq)

        T_final = self.forward_kinematics(q)
        self.last_position_error = float(np.linalg.norm(target_pos - T_final[:3, 3]))
        R_final_err = target_rot_matrix @ T_final[:3, :3].T
        self.last_orientation_error = float(np.linalg.norm(R.from_matrix(R_final_err).as_rotvec()[:2]))
        return q

    def solve_from_degrees(self, target_pos, target_rot_matrix, current_joints_deg, **kw):
        q_rad = np.deg2rad(current_joints_deg.astype(np.float64))
        return np.rad2deg(self.solve(target_pos, target_rot_matrix, q_rad, **kw))
