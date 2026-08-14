from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from config.runtime_config import load_runtime_config
from control.bp_delta_controller import BPDeltaController
from control.gravity_compensation import GravityCompensator, GravityModelError, PITCH_JOINTS


CONFIG_PATH = DEPLOY / "config" / "deployment.yaml"


@pytest.fixture()
def config():
    return load_runtime_config(CONFIG_PATH)


@pytest.fixture()
def gravity(config):
    return GravityCompensator.from_runtime_config(config)


def _build(config, **overrides):
    urdf_path = overrides.pop("urdf_path", config.gravity_urdf)
    settings = config.gravity
    camera = settings["camera"]
    jaw = settings["custom_jaw"]
    arguments = {
        "joint_order": config.joint_order,
        "q_ref_urdf_rad": np.deg2rad(config.start_urdf_deg),
        "static_bias_deg": config.static_support_bias_deg,
        "gravity_reference_bias_deg": config.gravity_reference_bias_deg,
        "ratio_clip": settings["ratio_clip"],
        "denominator_guard_nm": settings["denominator_guard_nm"],
        "gravity_m_s2": settings["gravity_m_s2"],
        "gripper_link": settings["gripper_link"],
        "tool_reference_link": settings["tool_reference_link"],
        "stock_moving_jaw_link": settings["stock_moving_jaw_link"],
        "camera_mass_kg": camera["mass_kg"],
        "camera_backward_m": camera["backward_m"],
        "camera_upward_m": camera["upward_m"],
        "custom_jaw_assembly_mass_kg": jaw["assembly_mass_kg"],
        "custom_jaw_length_scale": jaw["length_scale"],
        "custom_jaw_com_fraction_from_base": jaw["com_fraction_from_base"],
    }
    arguments.update(overrides)
    return GravityCompensator(urdf_path, **arguments)


def test_urdf_parse_inertials_and_stock_jaw_contract(gravity):
    assert gravity.root_link == "base_link"
    assert gravity.gripper_link == "gripper_link"
    assert gravity.stock_moving_jaw_link == "moving_jaw_so101_v1_link"
    assert gravity.stock_moving_jaw_present
    assert gravity.stock_moving_jaw_mass_kg == pytest.approx(0.012)
    assert gravity.missing_inertial_links == ()
    assert set(gravity.joint_order) == set(gravity.joints).intersection(gravity.joint_order)
    assert gravity.stock_moving_jaw_link in gravity.active_native_inertial_links(
        include_custom_jaw=False
    )
    assert gravity.stock_moving_jaw_link not in gravity.active_native_inertial_links(
        include_custom_jaw=True
    )


def test_reference_anchor_exact_and_denominator_guard(config, gravity):
    evaluation = gravity.evaluate(np.deg2rad(config.start_urdf_deg))
    old_bias = np.array([0.264, 0.615, 2.286, 0.703, 0.352])
    assert np.array_equal(evaluation.gravity_bias_deg, config.gravity_reference_bias_deg)
    assert np.array_equal(evaluation.total_support_bias_deg, old_bias)
    assert np.array_equal(evaluation.ratio_unclipped[[1, 2, 3]], np.ones(3))
    assert np.min(np.abs(gravity.tau_ref_nm[[1, 2, 3]])) >= gravity.denominator_guard_nm
    with pytest.raises(GravityModelError, match="near-zero reference"):
        _build(config, denominator_guard_nm=100.0)


def test_payload_and_geometry_contract(config, gravity):
    settings = config.gravity
    assert gravity.camera_mass_kg == settings["camera"]["mass_kg"]
    assert gravity.custom_jaw_assembly_mass_kg == settings["custom_jaw"]["assembly_mass_kg"]
    expected_world = np.array(
        [-settings["camera"]["backward_m"], 0.0, settings["camera"]["upward_m"]]
    )
    assert np.allclose(gravity.camera_world_offset(gravity.q_ref_urdf_rad), expected_world, atol=1e-12)
    assert gravity.effective_jaw_reach_scale == pytest.approx(2.0 / 3.0)
    assert np.allclose(
        gravity.custom_jaw_com_local_m,
        gravity.stock_reach_vector_local_m * (2.0 / 3.0),
        atol=1e-15,
    )


def test_payload_ablation_contributes_to_pitch_torque(gravity):
    poses = [
        gravity.q_ref_urdf_rad,
        gravity.q_ref_urdf_rad + np.deg2rad([0.0, 10.0, -8.0, 6.0, 0.0]),
        gravity.q_ref_urdf_rad + np.deg2rad([5.0, -12.0, 10.0, -5.0, 8.0]),
    ]
    for pose in poses:
        native = gravity.gravity_torque(pose, include_camera=False, include_custom_jaw=False)
        camera = gravity.gravity_torque(pose, include_camera=True, include_custom_jaw=False)
        jaw = gravity.gravity_torque(pose, include_camera=False, include_custom_jaw=True)
        full = gravity.gravity_torque(pose, include_camera=True, include_custom_jaw=True)
        assert np.linalg.norm((camera - native)[[1, 2, 3]]) > 0.0
        assert np.linalg.norm((jaw - native)[[1, 2, 3]]) > 0.0
        assert np.allclose(full - jaw, camera - native, atol=1e-14)


def test_ratio_clamp_and_finite_pose_sweep(gravity):
    rng = np.random.default_rng(11)
    samples = [gravity.q_ref_urdf_rad]
    for _ in range(200):
        samples.append(
            gravity.q_ref_urdf_rad
            + np.deg2rad(
                [
                    rng.uniform(-30.0, 30.0),
                    rng.uniform(-70.0, 100.0),
                    rng.uniform(-90.0, 15.0),
                    rng.uniform(-80.0, 40.0),
                    rng.uniform(-90.0, 90.0),
                ]
            )
        )
    saw_clamp = False
    for pose in samples:
        evaluation = gravity.evaluate(pose)
        for values in (
            evaluation.tau_g_nm,
            evaluation.ratio_unclipped,
            evaluation.ratio_clipped,
            evaluation.gravity_bias_deg,
            evaluation.total_support_bias_deg,
        ):
            assert values.shape == (5,) and np.isfinite(values).all()
        assert np.max(np.abs(evaluation.ratio_clipped)) <= gravity.ratio_clip
        saw_clamp = saw_clamp or bool(np.any(evaluation.clamp_active))
    assert saw_clamp


@pytest.mark.parametrize(
    "bad",
    [np.zeros(4), np.zeros(6), np.full(5, np.nan), np.full(5, np.inf)],
)
def test_gravity_input_shape_and_finite_gates(gravity, bad):
    with pytest.raises(GravityModelError):
        gravity.evaluate(bad)


def test_missing_urdf_joint_link_and_nonfinite_output_gates(config, gravity, tmp_path, monkeypatch):
    with pytest.raises(GravityModelError, match="not found"):
        _build(config, urdf_path=tmp_path / "missing.urdf")

    text = config.gravity_urdf.read_text()
    missing_joint = tmp_path / "missing_joint.urdf"
    missing_joint.write_text(text.replace('name="elbow_flex"', 'name="elbow_missing"', 1))
    with pytest.raises(GravityModelError, match="missing joints"):
        GravityCompensator(missing_joint, **_constructor_arguments(config))

    missing_link = tmp_path / "missing_link.urdf"
    missing_link.write_text(text.replace('name="gripper_frame_link"', 'name="tool_missing"', 1))
    with pytest.raises(GravityModelError, match="missing links"):
        GravityCompensator(missing_link, **_constructor_arguments(config))

    monkeypatch.setattr(gravity, "gravity_torque", lambda *_args, **_kwargs: np.full(5, np.nan))
    with pytest.raises(GravityModelError, match="nonfinite"):
        gravity.evaluate(gravity.q_ref_urdf_rad)


def _constructor_arguments(config):
    settings = config.gravity
    camera = settings["camera"]
    jaw = settings["custom_jaw"]
    return {
        "joint_order": config.joint_order,
        "q_ref_urdf_rad": np.deg2rad(config.start_urdf_deg),
        "static_bias_deg": config.static_support_bias_deg,
        "gravity_reference_bias_deg": config.gravity_reference_bias_deg,
        "ratio_clip": settings["ratio_clip"],
        "denominator_guard_nm": settings["denominator_guard_nm"],
        "gravity_m_s2": settings["gravity_m_s2"],
        "gripper_link": settings["gripper_link"],
        "tool_reference_link": settings["tool_reference_link"],
        "stock_moving_jaw_link": settings["stock_moving_jaw_link"],
        "camera_mass_kg": camera["mass_kg"],
        "camera_backward_m": camera["backward_m"],
        "camera_upward_m": camera["upward_m"],
        "custom_jaw_assembly_mass_kg": jaw["assembly_mass_kg"],
        "custom_jaw_length_scale": jaw["length_scale"],
        "custom_jaw_com_fraction_from_base": jaw["com_fraction_from_base"],
    }


def test_bp_delta_dynamic_support_tick0_continuity_and_true_reset(config, gravity):
    controller = BPDeltaController(config.k_ext, config.q_corr_clamp_deg, gravity)
    q_ref = config.start_urdf_deg.copy()
    old_bias = config.static_support_bias_deg + config.gravity_reference_bias_deg

    hold = controller.compute_hold(q_ref, q_ref)
    assert np.array_equal(hold, q_ref + old_bias)
    changed_actual = q_ref + np.array([0.0, 20.0, -15.0, 10.0, 0.0])
    changed_hold = controller.compute_hold(q_ref, changed_actual)
    assert not np.array_equal(changed_hold - q_ref, old_bias)

    hold = controller.compute_hold(q_ref, q_ref)
    origin = controller.begin_trajectory(q_ref, q_ref, hold)
    tick0 = controller.compute_trajectory(q_ref, q_ref)
    assert np.array_equal(tick0, hold)

    delta_nom = np.array([0.0, 1.0, -0.5, 0.25, 0.0])
    unchanged_actual_command = controller.compute_trajectory(q_ref + delta_nom, q_ref)
    expected_correction = np.clip(config.k_ext * delta_nom, -2.0, 2.0)
    assert np.allclose(unchanged_actual_command, hold + delta_nom + expected_correction)

    moved_actual = q_ref + np.array([0.0, 8.0, -4.0, 3.0, 0.0])
    dynamic_command = controller.compute_trajectory(q_ref + delta_nom, moved_actual)
    current_support = gravity.support_bias_deg(np.deg2rad(moved_actual))
    expected = (
        hold
        + delta_nom
        + (current_support - origin.b_support_0)
        + np.clip(config.k_ext * (delta_nom - (moved_actual - q_ref)), -2.0, 2.0)
    )
    assert np.allclose(dynamic_command, expected)
    diagnostics = controller.diagnostics()
    assert set(diagnostics) == {
        "tau_g_nm",
        "tau_ref_nm",
        "gravity_ratio_unclipped",
        "gravity_ratio_clipped",
        "gravity_bias_deg",
        "static_bias_deg",
        "total_support_bias_deg",
        "delta_support_bias_deg",
        "q_corr_deg",
        "gravity_clamp_active",
    }
    assert all(np.asarray(value).shape == (5,) for value in diagnostics.values())

    # New chunks, overlap updates, and reanchors have no controller reset API.
    controller.compute_trajectory(q_ref + 2.0 * delta_nom, moved_actual)
    assert controller.origin is origin
    controller.compute_trajectory(q_ref + 3.0 * delta_nom, moved_actual)
    assert controller.origin is origin

    controller.reset()
    new_hold = controller.compute_hold(q_ref, moved_actual)
    new_origin = controller.begin_trajectory(moved_actual, q_ref, new_hold)
    assert new_origin is not origin
    assert np.array_equal(controller.compute_trajectory(q_ref, moved_actual), new_hold)


def test_no_elbow_sag_fit_or_legacy_gain_in_canonical_gravity_source():
    source = (DEPLOY / "control" / "gravity_compensation.py").read_text().lower()
    assert "7°" not in source
    assert "7 degree" not in source
    assert "sag calibration" not in source
    assert "inverse stiffness" not in source
    assert "1.1" not in source
    assert all(name in PITCH_JOINTS for name in ("shoulder_lift", "elbow_flex", "wrist_flex"))
