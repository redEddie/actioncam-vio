"""Accepted pose-dependent support plus delta-feedback controller in URDF degrees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .gravity_compensation import GravityCompensator, GravityEvaluation


def _arm(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain five finite degree values")
    return result


@dataclass(frozen=True)
class TrajectoryOrigin:
    q_actual_0: np.ndarray
    q_nom_0: np.ndarray
    q_cmd_hold: np.ndarray
    b_support_0: np.ndarray


class BPDeltaController:
    """Stateful controller whose origin changes only on a true trajectory onset."""

    def __init__(self, k_ext: float, clamp_deg: float, gravity: GravityCompensator):
        self.k_ext = float(k_ext)
        self.clamp_deg = float(clamp_deg)
        if not isinstance(gravity, GravityCompensator):
            raise TypeError("gravity must be an initialized GravityCompensator")
        self.gravity = gravity
        if not np.isfinite(self.k_ext) or self.k_ext != 0.5:
            raise ValueError("canonical K_ext must be 0.5")
        if not np.isfinite(self.clamp_deg) or self.clamp_deg <= 0.0:
            raise ValueError("clamp_deg must be positive and finite")
        self._last_hold: np.ndarray | None = None
        self._origin: TrajectoryOrigin | None = None
        self._last_gravity: GravityEvaluation | None = None
        self._last_correction_deg: np.ndarray | None = None
        self._last_delta_support_deg: np.ndarray | None = None

    @property
    def origin(self) -> TrajectoryOrigin | None:
        return self._origin

    @property
    def last_gravity(self) -> GravityEvaluation | None:
        return self._last_gravity

    @property
    def last_correction_deg(self) -> np.ndarray | None:
        return None if self._last_correction_deg is None else self._last_correction_deg.copy()

    @property
    def last_delta_support_deg(self) -> np.ndarray | None:
        return None if self._last_delta_support_deg is None else self._last_delta_support_deg.copy()

    def _support(self, q_actual_deg: np.ndarray) -> np.ndarray:
        self._last_gravity = self.gravity.evaluate(np.deg2rad(q_actual_deg))
        return self._last_gravity.total_support_bias_deg.copy()

    def compute_hold(self, q_nom_hold: Sequence[float], q_actual: Sequence[float]) -> np.ndarray:
        nominal = _arm(q_nom_hold, "q_nom_hold")
        actual = _arm(q_actual, "q_actual")
        correction = np.clip(self.k_ext * (nominal - actual), -self.clamp_deg, self.clamp_deg)
        support = self._support(actual)
        self._last_correction_deg = correction.copy()
        self._last_delta_support_deg = np.zeros(5, dtype=np.float64)
        self._last_hold = nominal + support + correction
        return self._last_hold.copy()

    def begin_trajectory(
        self, q_actual_0: Sequence[float], q_nom_0: Sequence[float], q_cmd_hold: Sequence[float] | None = None
    ) -> TrajectoryOrigin:
        actual = _arm(q_actual_0, "q_actual_0").copy()
        nominal = _arm(q_nom_0, "q_nom_0").copy()
        hold_source = q_cmd_hold if q_cmd_hold is not None else self._last_hold
        if hold_source is None:
            raise RuntimeError("compute_hold must precede begin_trajectory")
        hold = _arm(hold_source, "q_cmd_hold").copy()
        support = self._support(actual)
        self._last_delta_support_deg = np.zeros(5, dtype=np.float64)
        self._origin = TrajectoryOrigin(actual, nominal, hold, support)
        return self._origin

    def compute_trajectory(self, q_nom: Sequence[float], q_actual: Sequence[float]) -> np.ndarray:
        if self._origin is None:
            raise RuntimeError("begin_trajectory must be called once at HOLD→TRAJECTORY onset")
        nominal = _arm(q_nom, "q_nom")
        actual = _arm(q_actual, "q_actual")
        delta_nominal = nominal - self._origin.q_nom_0
        delta_actual = actual - self._origin.q_actual_0
        correction = np.clip(
            self.k_ext * (delta_nominal - delta_actual), -self.clamp_deg, self.clamp_deg
        )
        support = self._support(actual)
        delta_support = support - self._origin.b_support_0
        self._last_delta_support_deg = delta_support.copy()
        self._last_correction_deg = correction.copy()
        return self._origin.q_cmd_hold + delta_nominal + delta_support + correction

    def diagnostics(self) -> dict[str, np.ndarray]:
        if self._last_gravity is None:
            raise RuntimeError("controller has no gravity evaluation")
        evaluation = self._last_gravity
        correction = (
            np.zeros(5, dtype=np.float64)
            if self._last_correction_deg is None
            else self._last_correction_deg
        )
        delta_support = (
            np.zeros(5, dtype=np.float64)
            if self._last_delta_support_deg is None
            else self._last_delta_support_deg
        )
        return {
            "tau_g_nm": evaluation.tau_g_nm.copy(),
            "tau_ref_nm": evaluation.tau_ref_nm.copy(),
            "gravity_ratio_unclipped": evaluation.ratio_unclipped.copy(),
            "gravity_ratio_clipped": evaluation.ratio_clipped.copy(),
            "gravity_bias_deg": evaluation.gravity_bias_deg.copy(),
            "static_bias_deg": evaluation.static_bias_deg.copy(),
            "total_support_bias_deg": evaluation.total_support_bias_deg.copy(),
            "delta_support_bias_deg": delta_support.copy(),
            "q_corr_deg": correction.copy(),
            "gravity_clamp_active": evaluation.clamp_active.copy(),
        }

    def reset(self) -> None:
        self._last_hold = None
        self._origin = None
        self._last_gravity = None
        self._last_correction_deg = None
        self._last_delta_support_deg = None


__all__ = ["BPDeltaController", "TrajectoryOrigin"]
