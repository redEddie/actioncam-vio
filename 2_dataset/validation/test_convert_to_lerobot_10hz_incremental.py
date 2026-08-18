#!/usr/bin/env python3
"""Motorless tests for the reconstructed canonical Zarr→LeRobot converter."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = (
    PROJECT_ROOT / "2_dataset/conversion/convert_to_lerobot_10hz_incremental.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_converter", CONVERTER_PATH)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


class FakeImages:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values, dtype=np.uint8)
        self.shape = self.values.shape
        self.dtype = self.values.dtype

    def read_index(self, index: int) -> np.ndarray:
        return self.values[index]


class FakeDataset:
    def __init__(self):
        self.current: list[dict] = []
        self.episodes: list[list[dict]] = []

    def add_frame(self, frame: dict) -> None:
        self.current.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self.current)
        self.current = []


def make_source(episodes: list[np.ndarray]) -> converter.SourceDataset:
    states = np.concatenate(episodes).astype(np.float32)
    images = np.zeros((len(states), 224, 224, 3), dtype=np.uint8)
    images[..., 0] = np.arange(len(states), dtype=np.uint8)[:, None, None]
    ends = np.cumsum([len(episode) for episode in episodes], dtype=np.int64)
    return converter.SourceDataset(
        path=Path("/fake/replay_buffer_yawfree.zarr"),
        attrs=dict(converter.EXPECTED_SOURCE_ATTRS),
        images=FakeImages(images),
        positions=states[:, :3],
        roll_pitch=states[:, 3:5],
        gripper=states[:, 5:6],
        episode_ends=ends,
    )


class DeltaMathTest(unittest.TestCase):
    def test_step_to_step_not_anchor_relative_or_absolute(self) -> None:
        states = np.array(
            [
                [0.10, -0.20, 0.30, -0.40, 0.05, 0.20],
                [0.12, -0.23, 0.31, -0.35, 0.02, 0.60],
                [0.09, -0.19, 0.35, -0.37, 0.08, 0.10],
            ],
            dtype=np.float32,
        )
        actions = converter.compute_incremental_actions(states)
        self.assertEqual(actions.shape, (2, 6))
        np.testing.assert_array_equal(actions[0], states[1] - states[0])
        np.testing.assert_array_equal(actions[1], states[2] - states[1])
        self.assertFalse(np.array_equal(actions[1], states[2] - states[0]))
        self.assertFalse(np.array_equal(actions[0], states[1]))

    def test_zero_motion_and_gripper_open_close(self) -> None:
        states = np.zeros((4, 6), dtype=np.float32)
        states[:, 5] = [0.2, 0.8, 0.3, 0.3]
        actions = converter.compute_incremental_actions(states)
        np.testing.assert_array_equal(actions[:3, :5], 0.0)
        np.testing.assert_allclose(actions[:3, 5], [0.6, -0.5, 0.0], atol=1e-7)

    def test_invalid_shape_finite_gripper_and_wrap_rejected(self) -> None:
        with self.assertRaises(converter.ConverterError):
            converter.compute_incremental_actions(np.zeros((3, 7), dtype=np.float32))
        nonfinite = np.zeros((3, 6), dtype=np.float32)
        nonfinite[1, 0] = np.nan
        with self.assertRaises(converter.ConverterError):
            converter.compute_incremental_actions(nonfinite)
        bad_gripper = np.zeros((3, 6), dtype=np.float32)
        bad_gripper[1, 5] = 1.1
        with self.assertRaises(converter.ConverterError):
            converter.compute_incremental_actions(bad_gripper)
        wrapped = np.zeros((3, 6), dtype=np.float32)
        wrapped[:, 3] = [0.0, 3.2, 3.3]
        with self.assertRaises(converter.ConverterError):
            converter.compute_incremental_actions(wrapped)


class SamplingTest(unittest.TestCase):
    def test_nearest_timestamp_half_up_is_deterministic(self) -> None:
        first = converter.select_10hz_samples(1_000)
        second = converter.select_10hz_samples(1_000)
        np.testing.assert_array_equal(first.indices, second.indices)
        self.assertTrue(np.all(np.diff(first.indices) > 0))
        self.assertLessEqual(
            float(np.abs(first.timestamp_errors).max()),
            0.5 / converter.SOURCE_FPS_DEFAULT + 1e-12,
        )
        np.testing.assert_allclose(np.diff(first.target_timestamps), 0.1, atol=1e-12)

    def test_fifteen_action_window_is_sequential(self) -> None:
        states = np.zeros((16, 6), dtype=np.float32)
        step = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06], dtype=np.float32)
        for index in range(1, len(states)):
            states[index] = states[index - 1] + step
        # Keep the normalized gripper in range while retaining a nonzero delta.
        states[:, 5] = np.linspace(0.0, 0.75, len(states), dtype=np.float32)
        actions = converter.compute_incremental_actions(states)
        window = actions[:15]
        self.assertEqual(window.shape, (15, 6))
        np.testing.assert_array_equal(window, np.diff(states, axis=0))
        anchor_relative = states[1:16] - states[0]
        self.assertFalse(np.array_equal(window[1:], anchor_relative[1:]))


class EpisodeWriterTest(unittest.TestCase):
    def test_episode_boundary_isolation_without_synthetic_terminal_row(self) -> None:
        episode_a = np.array(
            [
                [0, 0, 0, 0, 0, 0.1],
                [1, 0, 0, 0, 0, 0.2],
                [3, 0, 0, 0, 0, 0.3],
            ],
            dtype=np.float32,
        )
        episode_b = np.array(
            [
                [100, 0, 0, 0, 0, 0.9],
                [104, 0, 0, 0, 0, 0.8],
                [109, 0, 0, 0, 0, 0.7],
            ],
            dtype=np.float32,
        )
        source = make_source([episode_a, episode_b])
        sink = FakeDataset()
        episodes, frames = converter.write_selected_episodes(
            source, sink, source_fps=10.0, episode_ids=[1, 2], task="test"
        )
        self.assertEqual((episodes, frames), (2, 4))
        self.assertEqual([len(ep) for ep in sink.episodes], [2, 2])
        np.testing.assert_array_equal(
            sink.episodes[0][1][converter.ACTION_KEY], episode_a[2] - episode_a[1]
        )
        np.testing.assert_array_equal(
            sink.episodes[0][-1][converter.ACTION_KEY], episode_a[2] - episode_a[1]
        )
        np.testing.assert_array_equal(
            sink.episodes[1][0][converter.ACTION_KEY], episode_b[1] - episode_b[0]
        )
        self.assertNotEqual(float(sink.episodes[0][-1][converter.ACTION_KEY][0]), 97.0)

    def test_source_schema_and_feature_schema(self) -> None:
        base = np.zeros((3, 6), dtype=np.float32)
        source = make_source([base])
        converter.validate_source(source)
        features = converter.build_lerobot_features()
        self.assertEqual(features[converter.STATE_KEY]["shape"], (6,))
        self.assertEqual(features[converter.ACTION_KEY]["shape"], (6,))
        self.assertEqual(features[converter.IMAGE_KEY]["shape"], (224, 224, 3))
        self.assertEqual(
            features[converter.ACTION_KEY]["names"]["axes"], list(converter.ACTION_ORDER)
        )


class BundleContractTest(unittest.TestCase):
    def test_final_bundle_contract(self) -> None:
        checks = converter.validate_bundle_contract(PROJECT_ROOT)
        self.assertTrue(all(checks.values()))
        self.assertEqual(converter.PROVENANCE_CLASSIFICATION, "RECONSTRUCTED_CANONICAL_IMPLEMENTATION")


if __name__ == "__main__":
    unittest.main()
