"""Import-safe SO-101 V1 pose-dependent gravity support.

The model parses one inertial URDF at initialization, caches its kinematic
tree and reference torque, and performs only NumPy FK/point-mass calculations
per control tick. It has no hardware, network, camera, logging, or file-write
capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
PITCH_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")


class GravityModelError(ValueError):
    """The gravity model or one of its numerical inputs violates the contract."""


def _vector(values: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise GravityModelError(f"{name} must contain {size} finite values")
    return result


def _readonly(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _xyz(element: ET.Element | None, attribute: str, default: str = "0 0 0") -> np.ndarray:
    if element is None:
        return np.zeros(3, dtype=np.float64)
    return _vector([float(value) for value in element.attrib.get(attribute, default).split()], 3, attribute)


def _rpy_rotation(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis RPY rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    roll, pitch, yaw = _vector(rpy, 3, "rpy")
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    vector = _vector(axis, 3, "joint axis")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise GravityModelError("revolute joint axis must be nonzero")
    x, y, z = vector / norm
    c, s = np.cos(float(angle)), np.sin(float(angle))
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rpy_rotation(rpy)
    result[:3, 3] = _vector(xyz, 3, "xyz")
    return result


@dataclass(frozen=True)
class LinkInertial:
    link: str
    mass_kg: float
    com_local_m: np.ndarray


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class GravityEvaluation:
    q_actual_urdf_rad: np.ndarray
    tau_g_nm: np.ndarray
    tau_ref_nm: np.ndarray
    ratio_unclipped: np.ndarray
    ratio_clipped: np.ndarray
    gravity_bias_deg: np.ndarray
    static_bias_deg: np.ndarray
    total_support_bias_deg: np.ndarray
    clamp_active: np.ndarray


class GravityCompensator:
    """Cached URDF gravity model plus reference-scaled position-domain bias."""

    @classmethod
    def from_runtime_config(cls, config: Any) -> "GravityCompensator":
        """Construct once from the validated canonical RuntimeConfig."""

        gravity = config.gravity
        camera = gravity["camera"]
        jaw = gravity["custom_jaw"]
        return cls(
            config.gravity_urdf,
            joint_order=config.joint_order,
            q_ref_urdf_rad=np.deg2rad(config.start_urdf_deg),
            static_bias_deg=config.static_support_bias_deg,
            gravity_reference_bias_deg=config.gravity_reference_bias_deg,
            ratio_clip=gravity["ratio_clip"],
            denominator_guard_nm=gravity["denominator_guard_nm"],
            gravity_m_s2=gravity["gravity_m_s2"],
            gripper_link=gravity["gripper_link"],
            tool_reference_link=gravity["tool_reference_link"],
            stock_moving_jaw_link=gravity["stock_moving_jaw_link"],
            camera_mass_kg=camera["mass_kg"],
            camera_backward_m=camera["backward_m"],
            camera_upward_m=camera["upward_m"],
            custom_jaw_assembly_mass_kg=jaw["assembly_mass_kg"],
            custom_jaw_length_scale=jaw["length_scale"],
            custom_jaw_com_fraction_from_base=jaw["com_fraction_from_base"],
        )

    def __init__(
        self,
        urdf_path: Path | str,
        *,
        joint_order: Sequence[str],
        q_ref_urdf_rad: Sequence[float],
        static_bias_deg: Sequence[float],
        gravity_reference_bias_deg: Sequence[float],
        ratio_clip: float,
        denominator_guard_nm: float,
        gravity_m_s2: float,
        gripper_link: str,
        tool_reference_link: str,
        stock_moving_jaw_link: str,
        camera_mass_kg: float,
        camera_backward_m: float,
        camera_upward_m: float,
        custom_jaw_assembly_mass_kg: float,
        custom_jaw_length_scale: float,
        custom_jaw_com_fraction_from_base: float,
    ):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise GravityModelError(f"gravity URDF not found: {self.urdf_path}")
        self.joint_order = tuple(joint_order)
        if self.joint_order != ARM_JOINTS:
            raise GravityModelError(f"gravity joint order must be {ARM_JOINTS}")
        self.q_ref_urdf_rad = _readonly(_vector(q_ref_urdf_rad, 5, "q_ref_urdf_rad"))
        self.static_bias_deg = _readonly(_vector(static_bias_deg, 5, "static_bias_deg"))
        self.gravity_reference_bias_deg = _readonly(
            _vector(gravity_reference_bias_deg, 5, "gravity_reference_bias_deg")
        )
        self.ratio_clip = float(ratio_clip)
        self.denominator_guard_nm = float(denominator_guard_nm)
        self.gravity_m_s2 = float(gravity_m_s2)
        if not np.isfinite(self.ratio_clip) or self.ratio_clip <= 0.0:
            raise GravityModelError("ratio_clip must be finite and positive")
        if not np.isfinite(self.denominator_guard_nm) or self.denominator_guard_nm <= 0.0:
            raise GravityModelError("denominator_guard_nm must be finite and positive")
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise GravityModelError("gravity_m_s2 must be finite and positive")
        if not np.allclose(self.static_bias_deg[[1, 2, 3]], 0.0, atol=0.0, rtol=0.0):
            raise GravityModelError("static bias may only support shoulder_pan and wrist_roll")
        if not np.allclose(self.gravity_reference_bias_deg[[0, 4]], 0.0, atol=0.0, rtol=0.0):
            raise GravityModelError("dynamic gravity bias may only support the three pitch joints")

        physical_values = np.asarray(
            [
                camera_mass_kg,
                camera_backward_m,
                camera_upward_m,
                custom_jaw_assembly_mass_kg,
                custom_jaw_length_scale,
                custom_jaw_com_fraction_from_base,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(physical_values).all() or np.any(physical_values <= 0.0):
            raise GravityModelError("payload masses and geometry values must be finite and positive")
        self.camera_mass_kg = float(camera_mass_kg)
        self.camera_backward_m = float(camera_backward_m)
        self.camera_upward_m = float(camera_upward_m)
        self.custom_jaw_assembly_mass_kg = float(custom_jaw_assembly_mass_kg)
        self.custom_jaw_length_scale = float(custom_jaw_length_scale)
        self.custom_jaw_com_fraction_from_base = float(custom_jaw_com_fraction_from_base)
        self.gripper_link = str(gripper_link)
        self.tool_reference_link = str(tool_reference_link)
        self.stock_moving_jaw_link = str(stock_moving_jaw_link)

        self._parse_urdf()
        reference_transforms, _, _ = self._kinematics(self.q_ref_urdf_rad)
        gripper_transform = reference_transforms[self.gripper_link]
        desired_camera_world_offset = np.array(
            [-self.camera_backward_m, 0.0, self.camera_upward_m], dtype=np.float64
        )
        self.camera_com_local_m = _readonly(
            gripper_transform[:3, :3].T @ desired_camera_world_offset
        )
        tool_world = reference_transforms[self.tool_reference_link][:3, 3]
        gripper_world = gripper_transform[:3, 3]
        self.stock_reach_vector_local_m = _readonly(
            gripper_transform[:3, :3].T @ (tool_world - gripper_world)
        )
        self.effective_jaw_reach_scale = (
            self.custom_jaw_length_scale * self.custom_jaw_com_fraction_from_base
        )
        self.custom_jaw_com_local_m = _readonly(
            self.stock_reach_vector_local_m * self.effective_jaw_reach_scale
        )

        self.tau_ref_nm = _readonly(self.gravity_torque(self.q_ref_urdf_rad))
        self.pitch_indices = tuple(self.joint_order.index(name) for name in PITCH_JOINTS)
        enabled_reference = np.abs(self.tau_ref_nm[list(self.pitch_indices)])
        if np.any(enabled_reference < self.denominator_guard_nm):
            bad = [
                name
                for name, value in zip(PITCH_JOINTS, enabled_reference, strict=True)
                if value < self.denominator_guard_nm
            ]
            raise GravityModelError(f"near-zero reference gravity torque: {', '.join(bad)}")
        reference = self.evaluate(self.q_ref_urdf_rad)
        expected = self.static_bias_deg + self.gravity_reference_bias_deg
        if not np.array_equal(reference.total_support_bias_deg, expected):
            raise GravityModelError("reference support does not reproduce the configured anchor")

    def _parse_urdf(self) -> None:
        try:
            robot = ET.parse(self.urdf_path).getroot()
        except (ET.ParseError, OSError) as exc:
            raise GravityModelError(f"cannot parse gravity URDF: {exc}") from exc
        if robot.tag != "robot":
            raise GravityModelError("gravity URDF root element must be <robot>")
        self.robot_name = robot.attrib.get("name", "")
        self.links = {element.attrib["name"]: element for element in robot.findall("link")}
        if not self.links:
            raise GravityModelError("gravity URDF contains no links")
        required_links = {self.gripper_link, self.tool_reference_link}
        missing_links = sorted(required_links.difference(self.links))
        if missing_links:
            raise GravityModelError(f"gravity URDF missing links: {', '.join(missing_links)}")

        joints: dict[str, JointSpec] = {}
        child_to_joint: dict[str, JointSpec] = {}
        children: dict[str, list[JointSpec]] = {name: [] for name in self.links}
        for element in robot.findall("joint"):
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise GravityModelError("URDF joint is missing parent or child")
            name = element.attrib["name"]
            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]
            if parent not in self.links or child not in self.links:
                raise GravityModelError(f"joint {name} references an unknown link")
            origin_element = element.find("origin")
            origin = _transform(
                _xyz(origin_element, "xyz"), _xyz(origin_element, "rpy")
            )
            axis_element = element.find("axis")
            axis = _xyz(axis_element, "xyz", "0 0 1")
            spec = JointSpec(name, element.attrib.get("type", "fixed"), parent, child, origin, axis)
            joints[name] = spec
            child_to_joint[child] = spec
            children[parent].append(spec)
        missing_joints = [name for name in self.joint_order if name not in joints]
        if missing_joints:
            raise GravityModelError(f"gravity URDF missing joints: {', '.join(missing_joints)}")
        roots = sorted(set(self.links).difference(child_to_joint))
        if len(roots) != 1:
            raise GravityModelError(f"gravity URDF must have one root link, found {roots}")
        self.root_link = roots[0]
        self.joints = joints
        self._children = children

        inertials: list[LinkInertial] = []
        missing_inertials: list[str] = []
        for name, link in self.links.items():
            inertial = link.find("inertial")
            if inertial is None:
                missing_inertials.append(name)
                continue
            mass_element = inertial.find("mass")
            if mass_element is None:
                raise GravityModelError(f"link {name} inertial has no mass")
            mass = float(mass_element.attrib["value"])
            origin = inertial.find("origin")
            com = _xyz(origin, "xyz")
            if not np.isfinite(mass) or mass < 0.0:
                raise GravityModelError(f"link {name} has invalid inertial mass")
            inertials.append(LinkInertial(name, mass, _readonly(com)))
        self.inertials = tuple(inertials)
        self.missing_inertial_links = tuple(missing_inertials)
        inertial_by_link = {item.link: item for item in self.inertials}
        self.stock_moving_jaw_present = self.stock_moving_jaw_link in self.links
        self.stock_moving_jaw_mass_kg = (
            inertial_by_link[self.stock_moving_jaw_link].mass_kg
            if self.stock_moving_jaw_link in inertial_by_link
            else 0.0
        )
        if self.stock_moving_jaw_present and self.stock_moving_jaw_link not in inertial_by_link:
            raise GravityModelError("stock moving jaw exists but has no inertial")

        self._descendant_links: dict[str, frozenset[str]] = {}
        for name in self.joint_order:
            descendants: set[str] = set()
            stack = [self.joints[name].child]
            while stack:
                link = stack.pop()
                if link in descendants:
                    continue
                descendants.add(link)
                stack.extend(child.child for child in self._children[link])
            self._descendant_links[name] = frozenset(descendants)

    def _kinematics(
        self, q_actual_urdf_rad: Sequence[float]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        q = _vector(q_actual_urdf_rad, 5, "q_actual_urdf_rad")
        q_by_name = dict(zip(self.joint_order, q, strict=True))
        transforms: dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}
        joint_origins: dict[str, np.ndarray] = {}
        joint_axes: dict[str, np.ndarray] = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_transform = transforms[parent]
            for joint in self._children[parent]:
                joint_frame = parent_transform @ joint.origin
                joint_origins[joint.name] = joint_frame[:3, 3].copy()
                joint_axes[joint.name] = joint_frame[:3, :3] @ joint.axis
                motion = np.eye(4, dtype=np.float64)
                position = q_by_name.get(joint.name, 0.0)
                if joint.kind in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_rotation(joint.axis, position)
                elif joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * position
                elif joint.kind != "fixed":
                    raise GravityModelError(f"unsupported joint type {joint.kind}: {joint.name}")
                transforms[joint.child] = joint_frame @ motion
                stack.append(joint.child)
        if set(transforms) != set(self.links):
            missing = sorted(set(self.links).difference(transforms))
            raise GravityModelError(f"unreachable links in gravity URDF: {missing}")
        return transforms, joint_origins, joint_axes

    def gravity_torque(
        self,
        q_actual_urdf_rad: Sequence[float],
        *,
        include_camera: bool = True,
        include_custom_jaw: bool = True,
    ) -> np.ndarray:
        """Return arm gravity torque in Nm for an explicit payload ablation."""

        q = _vector(q_actual_urdf_rad, 5, "q_actual_urdf_rad")
        transforms, joint_origins, joint_axes = self._kinematics(q)
        gravity_force_per_kg = np.array([0.0, 0.0, -self.gravity_m_s2], dtype=np.float64)
        point_masses: list[tuple[str, float, np.ndarray]] = []
        for inertial in self.inertials:
            if include_custom_jaw and inertial.link == self.stock_moving_jaw_link:
                continue
            point_masses.append((inertial.link, inertial.mass_kg, inertial.com_local_m))
        if include_camera:
            point_masses.append((self.gripper_link, self.camera_mass_kg, self.camera_com_local_m))
        if include_custom_jaw:
            point_masses.append(
                (self.gripper_link, self.custom_jaw_assembly_mass_kg, self.custom_jaw_com_local_m)
            )

        torque = np.zeros(5, dtype=np.float64)
        for index, joint_name in enumerate(self.joint_order):
            axis = joint_axes[joint_name]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 0.0 or not np.isfinite(axis_norm):
                raise GravityModelError(f"joint {joint_name} has an invalid world axis")
            axis = axis / axis_norm
            origin = joint_origins[joint_name]
            descendants = self._descendant_links[joint_name]
            for link_name, mass, com_local in point_masses:
                if link_name not in descendants or mass == 0.0:
                    continue
                transform = transforms[link_name]
                com_world = transform[:3, :3] @ com_local + transform[:3, 3]
                moment = np.cross(com_world - origin, mass * gravity_force_per_kg)
                torque[index] += float(axis @ moment)
        if torque.shape != (5,) or not np.isfinite(torque).all():
            raise GravityModelError("gravity torque is nonfinite or has the wrong shape")
        return torque

    def active_native_inertial_links(self, *, include_custom_jaw: bool = True) -> tuple[str, ...]:
        """Expose the deterministic stock-jaw replacement decision for audits/tests."""

        return tuple(
            inertial.link
            for inertial in self.inertials
            if not (include_custom_jaw and inertial.link == self.stock_moving_jaw_link)
        )

    def evaluate(self, q_actual_urdf_rad: Sequence[float]) -> GravityEvaluation:
        q = _vector(q_actual_urdf_rad, 5, "q_actual_urdf_rad")
        torque = self.gravity_torque(q)
        if not np.isfinite(torque).all():
            raise GravityModelError("gravity torque is nonfinite")
        ratio_unclipped = np.zeros(5, dtype=np.float64)
        for index in self.pitch_indices:
            denominator = self.tau_ref_nm[index]
            if abs(denominator) < self.denominator_guard_nm:
                raise GravityModelError(f"near-zero reference torque for {self.joint_order[index]}")
            ratio_unclipped[index] = torque[index] / denominator
        ratio_clipped = np.clip(ratio_unclipped, -self.ratio_clip, self.ratio_clip)
        gravity_bias = self.gravity_reference_bias_deg * ratio_clipped
        support = self.static_bias_deg + gravity_bias
        if not np.isfinite(ratio_unclipped).all() or not np.isfinite(gravity_bias).all():
            raise GravityModelError("gravity ratio or bias is nonfinite")
        return GravityEvaluation(
            _readonly(q),
            _readonly(torque),
            self.tau_ref_nm,
            _readonly(ratio_unclipped),
            _readonly(ratio_clipped),
            _readonly(gravity_bias),
            self.static_bias_deg,
            _readonly(support),
            np.asarray(np.abs(ratio_unclipped) > self.ratio_clip, dtype=bool),
        )

    def support_bias_deg(self, q_actual_urdf_rad: Sequence[float]) -> np.ndarray:
        return self.evaluate(q_actual_urdf_rad).total_support_bias_deg.copy()

    def gravity_bias_deg(self, q_actual_urdf_rad: Sequence[float]) -> np.ndarray:
        return self.evaluate(q_actual_urdf_rad).gravity_bias_deg.copy()

    def camera_world_offset(self, q_actual_urdf_rad: Sequence[float]) -> np.ndarray:
        transforms, _, _ = self._kinematics(q_actual_urdf_rad)
        rotation = transforms[self.gripper_link][:3, :3]
        return rotation @ self.camera_com_local_m

    def debug_snapshot(self, q_actual_urdf_rad: Sequence[float]) -> Mapping[str, object]:
        evaluation = self.evaluate(q_actual_urdf_rad)
        return {
            "q_actual_urdf_rad": evaluation.q_actual_urdf_rad.copy(),
            "tau_g_nm": evaluation.tau_g_nm.copy(),
            "tau_ref_nm": evaluation.tau_ref_nm.copy(),
            "gravity_ratio_unclipped": evaluation.ratio_unclipped.copy(),
            "gravity_ratio_clipped": evaluation.ratio_clipped.copy(),
            "gravity_bias_deg": evaluation.gravity_bias_deg.copy(),
            "static_bias_deg": evaluation.static_bias_deg.copy(),
            "total_support_bias_deg": evaluation.total_support_bias_deg.copy(),
            "gravity_clamp_active": evaluation.clamp_active.copy(),
        }


__all__ = [
    "ARM_JOINTS",
    "GravityCompensator",
    "GravityEvaluation",
    "GravityModelError",
    "PITCH_JOINTS",
]
