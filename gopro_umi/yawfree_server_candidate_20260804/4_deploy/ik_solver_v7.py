#!/usr/bin/env python3
"""SO-100/SO-101 DLS IK solver: XYZ + roll/pitch with free yaw.

Joint values exposed by this class are radians.  LeRobot's
``RobotKinematics.forward_kinematics`` API expects degrees, so conversion is
performed only at that boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


_LEROBOT_CANDIDATES = (
    Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src",
    Path("/home/kimminje/lerobot/src"),
)
for _candidate in _LEROBOT_CANDIDATES:
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from lerobot.model.kinematics import RobotKinematics


class DLSInverseKinematicsV7:
    """Five-dimensional DLS task: XYZ plus global rotation-vector X/Y."""

    PHYSICAL_JOINT_LIMITS_RAD = {
        "shoulder_pan": (-1.91986, 1.91986),
        "shoulder_lift": (-1.74533, 1.74533),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.65806, 1.65806),
        "wrist_roll": (-2.74385, 2.84121),
    }

    def __init__(
        self,
        urdf_path: str,
        tcp_frame: str = "tcp_link",
        joint_names: list[str] | None = None,
        damping: float = 0.01,
        limit_margin_ratio: float = 0.90,
    ) -> None:
        if joint_names is None:
            joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ]
        unknown = sorted(set(joint_names) - set(self.PHYSICAL_JOINT_LIMITS_RAD))
        if unknown:
            raise ValueError(f"physical limits are missing for joints: {unknown}")
        if not 0.0 < limit_margin_ratio <= 1.0:
            raise ValueError("limit_margin_ratio must be in (0, 1]")

        self.joint_names = list(joint_names)
        self.n_joints = len(self.joint_names)
        self.damping = float(damping)
        self.limit_margin_ratio = float(limit_margin_ratio)
        self.safe_limits: dict[str, tuple[float, float]] = {}
        for name in self.joint_names:
            lo, hi = self.PHYSICAL_JOINT_LIMITS_RAD[name]
            center = (lo + hi) / 2.0
            safe_half_range = ((hi - lo) / 2.0) * self.limit_margin_ratio
            self.safe_limits[name] = (center - safe_half_range, center + safe_half_range)

        self._fk = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=tcp_frame,
            joint_names=self.joint_names,
        )
        self.last_position_error = float("inf")
        self.last_orientation_error = float("inf")
        self.last_iterations = 0
        self.last_converged = False
        self.last_clamp_events = 0
        self.last_physical_limit_attempt_events = 0

    def forward_kinematics(self, joints_rad: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints_rad, dtype=np.float64).reshape(self.n_joints)
        return np.asarray(self._fk.forward_kinematics(np.rad2deg(joints)), dtype=np.float64)

    def get_tcp_pose(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)

    def get_tcp_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)[:3, 3].copy()

    def get_tcp_rotation(self, joints_rad: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(joints_rad)[:3, :3].copy()

    def _clamp_joints_safe(self, joints_rad: np.ndarray, *, count: bool = True) -> np.ndarray:
        candidate = np.asarray(joints_rad, dtype=np.float64).copy()
        clamped = candidate.copy()
        for index, name in enumerate(self.joint_names):
            physical_lo, physical_hi = self.PHYSICAL_JOINT_LIMITS_RAD[name]
            safe_lo, safe_hi = self.safe_limits[name]
            if count and (candidate[index] < physical_lo or candidate[index] > physical_hi):
                self.last_physical_limit_attempt_events += 1
            clipped = float(np.clip(candidate[index], safe_lo, safe_hi))
            if count and not np.isclose(clipped, candidate[index], rtol=0.0, atol=1e-12):
                self.last_clamp_events += 1
            clamped[index] = clipped
        return clamped

    def _numerical_jacobian_6dof(self, joints_rad: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Return a 6x5 Jacobian for XYZ plus global rotation-vector XYZ."""
        jacobian = np.zeros((6, self.n_joints), dtype=np.float64)
        t0 = self.forward_kinematics(joints_rad)
        p0, r0 = t0[:3, 3], t0[:3, :3]
        for index in range(self.n_joints):
            q_plus = joints_rad.copy()
            q_plus[index] += eps
            t_plus = self.forward_kinematics(q_plus)
            jacobian[:3, index] = (t_plus[:3, 3] - p0) / eps
            r_delta = t_plus[:3, :3] @ r0.T
            jacobian[3:, index] = R.from_matrix(r_delta).as_rotvec() / eps
        return jacobian

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
        """Solve XYZ + global rotation X/Y; global Z/yaw is deliberately free."""
        self.last_clamp_events = 0
        self.last_physical_limit_attempt_events = 0
        q = self._clamp_joints_safe(
            np.asarray(current_joints, dtype=np.float64).reshape(self.n_joints), count=True
        )
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot_matrix = np.asarray(target_rot_matrix, dtype=np.float64).reshape(3, 3)
        self.last_converged = False

        for iteration in range(1, max_iter + 1):
            t_current = self.forward_kinematics(q)
            position_error = target_pos - t_current[:3, 3]
            rotation_error = R.from_matrix(target_rot_matrix @ t_current[:3, :3].T).as_rotvec()
            task_error = np.concatenate([position_error, rotation_error[:2]])
            position_norm = float(np.linalg.norm(position_error))
            roll_pitch_control_norm = float(np.linalg.norm(rotation_error[:2]))
            self.last_position_error = position_norm
            self.last_orientation_error = roll_pitch_control_norm
            self.last_iterations = iteration
            if position_norm < pos_tol and roll_pitch_control_norm < rot_tol:
                self.last_converged = True
                break

            jacobian_6d = self._numerical_jacobian_6dof(q)
            jacobian_task = np.vstack([jacobian_6d[:3, :], jacobian_6d[3:5, :]])
            damping = self.damping + 0.001 * position_norm
            system = jacobian_task @ jacobian_task.T + (damping**2) * np.eye(5)
            try:
                delta_q = jacobian_task.T @ np.linalg.solve(system, task_error)
            except np.linalg.LinAlgError:
                delta_q = jacobian_task.T @ np.linalg.lstsq(system, task_error, rcond=None)[0]
            step_norm = float(np.linalg.norm(delta_q))
            if step_norm > max_step_rad:
                delta_q *= max_step_rad / step_norm
            q = self._clamp_joints_safe(q + delta_q, count=True)

        final_pose = self.forward_kinematics(q)
        self.last_position_error = float(np.linalg.norm(target_pos - final_pose[:3, 3]))
        final_rotation_error = R.from_matrix(
            target_rot_matrix @ final_pose[:3, :3].T
        ).as_rotvec()
        self.last_orientation_error = float(np.linalg.norm(final_rotation_error[:2]))
        return q

    def solve_from_degrees(
        self,
        target_pos: np.ndarray,
        target_rot_matrix: np.ndarray,
        current_joints_deg: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        q_rad = np.deg2rad(np.asarray(current_joints_deg, dtype=np.float64))
        return np.rad2deg(self.solve(target_pos, target_rot_matrix, q_rad, **kwargs))

