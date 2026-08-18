#!/usr/bin/env python3
"""Convert the yaw-free UMI Zarr into canonical 10 Hz LeRobot rows.

Provenance
==========
This is a ``RECONSTRUCTED_CANONICAL_IMPLEMENTATION`` of the frozen dataset
contract.  It is not the recovered historical converter that produced the
step-20000 training dataset.

Each output row stores one real six-dimensional state transition:

    state  = [X, Y, Z, Roll, Pitch, Gripper]
    action = S[t + 1] - S[t]

The terminal state of each episode is retained only as the target of the final
transition; no synthetic zero-action row is written.  The dataset therefore
remains row-wise ``[6]``; SmolVLA constructs its ``[15, 6]`` future action
window from dataset timestamps and ``chunk_size=15``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


PROVENANCE_CLASSIFICATION = "RECONSTRUCTED_CANONICAL_IMPLEMENTATION"
CONVERTER_VERSION = "2.0.0"
TARGET_FPS = 10
ACTION_DIM = 6
CHUNK_SIZE = 15
CANONICAL_DATASET_NAME = "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
SOURCE_FPS_DEFAULT = 60_000 / 1_001
TASK_DESCRIPTION = "pick and place the target object"
STATE_AXES = ("X", "Y", "Z", "Roll", "Pitch", "Gripper")
ACTION_ORDER = ("dX", "dY", "dZ", "dRoll", "dPitch", "dGripper")
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

EXPECTED_SOURCE_ATTRS = {
    "pose_frame": "robot_base_tcp_roll_pitch_pose",
    "task_dof": "xyz_roll_pitch",
    "coordinate_transforms_applied_by_builder": "none",
    "yaw_removed": True,
}


class ConverterError(RuntimeError):
    """Base error for deterministic converter failures."""


class DependencyMissingError(ConverterError):
    """A required already-installed dependency is unavailable."""


class ImageArray(Protocol):
    shape: tuple[int, ...]
    dtype: np.dtype

    def read_index(self, index: int) -> np.ndarray:
        """Read one HWC image without materializing the full image array."""


@dataclass(frozen=True)
class SourceDataset:
    """Validated yaw-free Zarr content with lazy RGB access."""

    path: Path
    attrs: dict[str, Any]
    images: ImageArray
    positions: np.ndarray
    roll_pitch: np.ndarray
    gripper: np.ndarray
    episode_ends: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return np.concatenate((self.positions, self.roll_pitch, self.gripper), axis=1).astype(
            np.float32, copy=False
        )

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate((np.array([0], dtype=np.int64), self.episode_ends[:-1]))


@dataclass(frozen=True)
class SampleSelection:
    indices: np.ndarray
    target_timestamps: np.ndarray
    selected_source_timestamps: np.ndarray

    @property
    def timestamp_errors(self) -> np.ndarray:
        return self.selected_source_timestamps - self.target_timestamps


class _ZarrArrayAdapter:
    """Small adapter for a normal Zarr array."""

    def __init__(self, array: Any):
        self._array = array
        self.shape = tuple(int(value) for value in array.shape)
        self.dtype = np.dtype(array.dtype)

    def read_all(self) -> np.ndarray:
        return np.asarray(self._array[:])

    def read_index(self, index: int) -> np.ndarray:
        return np.asarray(self._array[index])


class _ZarrV2DirectoryArray:
    """Read-only fallback for the simple, regular Zarr-v2 arrays in this project.

    Zarr 3.3 can be extremely slow while opening this archived v2 DirectoryStore
    in the current environment.  This fallback reads the v2 metadata and chunks
    directly through ``numcodecs``.  It intentionally accepts only arrays whose
    non-leading dimensions occupy a single chunk, which is exactly the audited
    source schema.  It never writes to the source store.
    """

    def __init__(self, path: Path):
        try:
            from numcodecs import get_codec
        except ImportError as exc:
            raise DependencyMissingError(
                "numcodecs is required to read the source Zarr-v2 DirectoryStore"
            ) from exc

        metadata_path = path / ".zarray"
        if not metadata_path.is_file():
            raise ConverterError(f"missing Zarr-v2 array metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("zarr_format") != 2:
            raise ConverterError(f"fallback reader only accepts Zarr v2: {path}")
        if metadata.get("filters") not in (None, []):
            raise ConverterError(f"unsupported Zarr filters in {path}")
        if metadata.get("order") != "C":
            raise ConverterError(f"only C-order Zarr arrays are supported: {path}")

        self.path = path
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.chunks = tuple(int(value) for value in metadata["chunks"])
        self.dtype = np.dtype(metadata["dtype"])
        if len(self.shape) != len(self.chunks) or self.chunks[1:] != self.shape[1:]:
            raise ConverterError(
                f"fallback reader requires full non-leading chunks: {path}, "
                f"shape={self.shape}, chunks={self.chunks}"
            )
        compressor = metadata.get("compressor")
        self._codec = get_codec(compressor) if compressor else None
        self._cache_index: int | None = None
        self._cache: np.ndarray | None = None

    def _read_chunk(self, chunk_index: int) -> np.ndarray:
        if chunk_index == self._cache_index and self._cache is not None:
            return self._cache
        coordinates = ".".join((str(chunk_index), *("0" for _ in self.shape[1:])))
        chunk_path = self.path / coordinates
        if not chunk_path.is_file():
            raise ConverterError(f"missing Zarr chunk: {chunk_path}")
        raw: bytes | bytearray | memoryview = chunk_path.read_bytes()
        if self._codec is not None:
            raw = self._codec.decode(raw)
        array = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks, order="C")
        self._cache_index = chunk_index
        self._cache = array
        return array

    def read_all(self) -> np.ndarray:
        output = np.empty(self.shape, dtype=self.dtype)
        chunk_count = math.ceil(self.shape[0] / self.chunks[0])
        for chunk_index in range(chunk_count):
            start = chunk_index * self.chunks[0]
            stop = min(start + self.chunks[0], self.shape[0])
            output[start:stop] = self._read_chunk(chunk_index)[: stop - start]
        return output

    def read_index(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        chunk_index, offset = divmod(index, self.chunks[0])
        return np.asarray(self._read_chunk(chunk_index)[offset])


def discover_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], *here.parents):
        if (candidate / "7_storage" / "Delta_Weights" / "config.json").is_file() and (
            candidate / "2_dataset"
        ).is_dir():
            return candidate
    raise ConverterError(
        "cannot locate project root containing 7_storage/Delta_Weights"
    )


def build_lerobot_features() -> dict[str, dict[str, Any]]:
    return {
        IMAGE_KEY: {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"axes": list(STATE_AXES)},
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"axes": list(ACTION_ORDER)},
        },
    }


def validate_bundle_contract(project_root: Path) -> dict[str, Any]:
    bundle = project_root / "7_storage" / "Delta_Weights"
    config = json.loads((bundle / "config.json").read_text())
    train_config = json.loads((bundle / "train_config.json").read_text())
    preprocessor = json.loads(
        (bundle / "policy_preprocessor.json").read_text()
    )
    postprocessor = json.loads(
        (bundle / "policy_postprocessor.json").read_text()
    )
    policy = train_config["policy"]
    rename_map = train_config.get("rename_map", {})
    preprocessor_steps = {
        step["registry_name"]: step["config"] for step in preprocessor.get("steps", [])
    }
    preprocessor_features = preprocessor_steps.get("normalizer_processor", {}).get(
        "features", {}
    )
    postprocessor_steps = {
        step["registry_name"]: step["config"] for step in postprocessor.get("steps", [])
    }
    postprocessor_features = postprocessor_steps.get("unnormalizer_processor", {}).get(
        "features", {}
    )

    checks = {
        "state_feature": config["input_features"].get(STATE_KEY, {}).get("shape") == [6],
        "action_feature": config["output_features"].get(ACTION_KEY, {}).get("shape") == [6],
        "image_feature": config["input_features"].get("observation.images.camera1", {}).get("shape")
        == [3, 224, 224],
        "chunk_size": config.get("chunk_size") == CHUNK_SIZE,
        "n_action_steps": config.get("n_action_steps") == CHUNK_SIZE,
        "model_image_size": config.get("resize_imgs_with_padding") == [256, 256],
        "empty_cameras": config.get("empty_cameras") == 2,
        "rename_map": rename_map.get(IMAGE_KEY) == "observation.images.camera1",
        "train_policy_match": policy.get("chunk_size") == CHUNK_SIZE
        and policy.get("n_action_steps") == CHUNK_SIZE,
        "preprocessor_state": preprocessor_features.get(STATE_KEY, {}).get("shape") == [6],
        "preprocessor_action": preprocessor_features.get(ACTION_KEY, {}).get("shape") == [6],
        "preprocessor_image": preprocessor_features.get(
            "observation.images.camera1", {}
        ).get("shape")
        == [3, 224, 224],
        "preprocessor_rename": preprocessor_steps.get(
            "rename_observations_processor", {}
        ).get("rename_map", {}).get(IMAGE_KEY)
        == "observation.images.camera1",
        "postprocessor_action": postprocessor_features.get(ACTION_KEY, {}).get("shape")
        == [6],
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ConverterError(f"Delta_Weights contract mismatch: {', '.join(failed)}")
    return checks


def _load_source_with_zarr2(path: Path) -> SourceDataset | None:
    """Use the official API when an already-installed Zarr 2.x is available."""

    try:
        import zarr
    except ImportError:
        return None
    major = int(str(zarr.__version__).split(".", 1)[0])
    if major >= 3:
        return None
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    return SourceDataset(
        path=path,
        attrs=attrs,
        images=_ZarrArrayAdapter(root["data/camera0_rgb"]),
        positions=np.asarray(root["data/robot0_eef_pos"][:], dtype=np.float32),
        roll_pitch=np.asarray(root["data/robot0_eef_roll_pitch"][:], dtype=np.float32),
        gripper=np.asarray(root["data/robot0_gripper_width"][:], dtype=np.float32),
        episode_ends=np.asarray(root["meta/episode_ends"][:], dtype=np.int64),
    )


def _load_source_v2_fallback(path: Path) -> SourceDataset:
    attrs_path = path / ".zattrs"
    if not attrs_path.is_file():
        raise ConverterError(f"missing Zarr root attributes: {attrs_path}")
    attrs = json.loads(attrs_path.read_text())
    images = _ZarrV2DirectoryArray(path / "data" / "camera0_rgb")
    positions = _ZarrV2DirectoryArray(path / "data" / "robot0_eef_pos").read_all()
    roll_pitch = _ZarrV2DirectoryArray(path / "data" / "robot0_eef_roll_pitch").read_all()
    gripper = _ZarrV2DirectoryArray(path / "data" / "robot0_gripper_width").read_all()
    episode_ends = _ZarrV2DirectoryArray(path / "meta" / "episode_ends").read_all()
    return SourceDataset(
        path=path,
        attrs=attrs,
        images=images,
        positions=np.asarray(positions, dtype=np.float32),
        roll_pitch=np.asarray(roll_pitch, dtype=np.float32),
        gripper=np.asarray(gripper, dtype=np.float32),
        episode_ends=np.asarray(episode_ends, dtype=np.int64),
    )


def load_yawfree_source(path: Path) -> SourceDataset:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    source = _load_source_with_zarr2(path)
    if source is None:
        source = _load_source_v2_fallback(path)
    validate_source(source)
    return source


def validate_source(source: SourceDataset) -> None:
    for key, expected in EXPECTED_SOURCE_ATTRS.items():
        if source.attrs.get(key) != expected:
            raise ConverterError(
                f"source Zarr attribute {key!r} must be {expected!r}, got {source.attrs.get(key)!r}"
            )
    frame_count = int(source.images.shape[0])
    expected_shapes = {
        "image": (frame_count, 224, 224, 3),
        "position": (frame_count, 3),
        "roll_pitch": (frame_count, 2),
        "gripper": (frame_count, 1),
    }
    actual_shapes = {
        "image": tuple(source.images.shape),
        "position": tuple(source.positions.shape),
        "roll_pitch": tuple(source.roll_pitch.shape),
        "gripper": tuple(source.gripper.shape),
    }
    if actual_shapes != expected_shapes:
        raise ConverterError(f"unexpected yaw-free source shapes: {actual_shapes}")
    if np.dtype(source.images.dtype) != np.dtype(np.uint8):
        raise ConverterError(f"source images must be uint8 RGB, got {source.images.dtype}")
    if source.episode_ends.ndim != 1 or source.episode_ends.size == 0:
        raise ConverterError("episode_ends must be a non-empty vector")
    if int(source.episode_ends[-1]) != frame_count:
        raise ConverterError("final episode end must equal source frame count")
    if np.any(np.diff(np.concatenate(([0], source.episode_ends))) <= 0):
        raise ConverterError("episode_ends must describe strictly positive episodes")
    states = source.states
    if states.shape != (frame_count, ACTION_DIM) or not np.isfinite(states).all():
        raise ConverterError("source 6D state has wrong shape or contains NaN/Inf")
    if float(source.gripper.min()) < -1e-6 or float(source.gripper.max()) > 1.0 + 1e-6:
        raise ConverterError("source gripper must already be normalized to [0,1]")


def select_10hz_samples(
    source_count: int,
    source_fps: float = SOURCE_FPS_DEFAULT,
    target_fps: int = TARGET_FPS,
) -> SampleSelection:
    """Select nearest source timestamps for an episode-local 10 Hz clock.

    Ties use half-up rounding rather than Python's banker rounding.  Target times
    never extend beyond the final source timestamp.
    """

    if source_count < 2:
        raise ConverterError("an episode needs at least two source frames")
    if not math.isfinite(source_fps) or source_fps < target_fps:
        raise ConverterError("source_fps must be finite and at least target_fps")
    last_source_time = (source_count - 1) / source_fps
    target_count = math.floor(last_source_time * target_fps + 1e-12) + 1
    target_timestamps = np.arange(target_count, dtype=np.float64) / target_fps
    indices = np.floor(target_timestamps * source_fps + 0.5).astype(np.int64)
    indices = np.clip(indices, 0, source_count - 1)
    if indices.size < 2 or np.any(np.diff(indices) <= 0):
        raise ConverterError("10 Hz sampling produced fewer than two unique increasing samples")
    selected_source_timestamps = indices.astype(np.float64) / source_fps
    return SampleSelection(indices, target_timestamps, selected_source_timestamps)


def validate_state_matrix(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != ACTION_DIM or states.shape[0] < 2:
        raise ConverterError(f"states must have shape [N,6] with N>=2, got {states.shape}")
    if not np.isfinite(states).all():
        raise ConverterError("states contain NaN/Inf")
    if float(states[:, 5].min()) < -1e-6 or float(states[:, 5].max()) > 1.0 + 1e-6:
        raise ConverterError("gripper state must stay in [0,1]")
    return states


def compute_incremental_actions(states: np.ndarray) -> np.ndarray:
    """Return the ``N-1`` real step-to-step transitions for ``N`` states."""

    states = validate_state_matrix(states)
    deltas = np.diff(states, axis=0).astype(np.float32, copy=False)
    # The audited source contains no Roll/Pitch wrap crossings.  Reject future
    # crossings rather than silently changing the frozen training semantics.
    if np.any(np.abs(deltas[:, 3:5]) > np.pi):
        raise ConverterError(
            "Roll/Pitch wrap crossing detected; canonical direct-difference policy cannot continue"
        )
    return deltas


def _selected_episode_ids(
    episode_count: int, requested: Sequence[int] | None, max_episodes: int | None
) -> list[int]:
    if requested:
        ids = list(requested)
        if len(ids) != len(set(ids)):
            raise ConverterError("--episodes contains duplicates")
        invalid = [value for value in ids if value < 1 or value > episode_count]
        if invalid:
            raise ConverterError(f"episode IDs out of range 1..{episode_count}: {invalid}")
        ids.sort()
    else:
        ids = list(range(1, episode_count + 1))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ConverterError("--max-episodes must be positive")
        ids = ids[:max_episodes]
    return ids


def audit_source(
    source: SourceDataset,
    source_fps: float,
    episode_ids: Sequence[int] | None = None,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    validate_source(source)
    ids = _selected_episode_ids(len(source.episode_ends), episode_ids, max_episodes)
    states = source.states
    starts = source.episode_starts
    timestamp_errors: list[np.ndarray] = []
    source_spacings: list[np.ndarray] = []
    reconstruction_errors: list[np.ndarray] = []
    episode_metrics: list[dict[str, Any]] = []
    sampled_frames = 0
    source_wrap_crossings = np.zeros(2, dtype=np.int64)
    wrap_crossings = np.zeros(2, dtype=np.int64)
    duplicate_indices = 0
    timestamp_monotonic_violations = 0

    for episode_id in ids:
        start = int(starts[episode_id - 1])
        end = int(source.episode_ends[episode_id - 1])
        selection = select_10hz_samples(end - start, source_fps)
        selected_states = states[start + selection.indices]
        actions = compute_incremental_actions(selected_states)
        source_episode = states[start:end]
        timestamp_errors.append(np.abs(selection.timestamp_errors))
        source_spacings.append(np.diff(selection.selected_source_timestamps))
        duplicate_indices += int(np.count_nonzero(np.diff(selection.indices) <= 0))
        timestamp_monotonic_violations += int(
            np.count_nonzero(np.diff(selection.selected_source_timestamps) <= 0)
        )
        source_wrap_crossings += np.sum(
            np.abs(np.diff(source_episode[:, 3:5], axis=0)) > np.pi, axis=0
        )
        wrap_crossings += np.sum(np.abs(np.diff(selected_states[:, 3:5], axis=0)) > np.pi, axis=0)
        reconstructed = selected_states[:-1].astype(np.float64) + actions.astype(np.float64)
        reconstruction_errors.append(
            np.abs(reconstructed - selected_states[1:].astype(np.float64))
        )
        sampled_frames += len(actions)
        episode_error = np.abs(selection.timestamp_errors)
        episode_spacing = np.diff(selection.selected_source_timestamps)
        episode_metrics.append(
            {
                "episode_id": episode_id,
                "source_sample_count": end - start,
                "source_duration_s": (end - start - 1) / source_fps,
                "target_sample_count": len(selection.indices),
                "output_transition_count": len(actions),
                "first_target_timestamp_s": float(selection.target_timestamps[0]),
                "last_target_timestamp_s": float(selection.target_timestamps[-1]),
                "mean_selected_source_spacing_s": float(episode_spacing.mean()),
                "min_selected_source_spacing_s": float(episode_spacing.min()),
                "max_selected_source_spacing_s": float(episode_spacing.max()),
                "mean_abs_timestamp_error_s": float(episode_error.mean()),
                "p95_abs_timestamp_error_s": float(np.percentile(episode_error, 95)),
                "max_abs_timestamp_error_s": float(episode_error.max()),
            }
        )

    errors = np.concatenate(timestamp_errors)
    spacings = np.concatenate(source_spacings)
    reconstruction = np.concatenate(reconstruction_errors)
    representative_indices = (0, source.images.shape[0] // 2, source.images.shape[0] - 1)
    image_samples = [source.images.read_index(index) for index in representative_indices]
    for image in image_samples:
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ConverterError("source RGB sample violates uint8 HWC 224x224x3 contract")

    return {
        "provenance_classification": PROVENANCE_CLASSIFICATION,
        "converter_version": CONVERTER_VERSION,
        "source_path": str(source.path),
        "source_time_basis": "episode-local implicit frame_index / source_fps",
        "source_fps": source_fps,
        "target_fps": TARGET_FPS,
        "sampling_rule": "nearest source timestamp to n/10, deterministic half-up ties",
        "episodes_total": int(len(source.episode_ends)),
        "episodes_audited": ids,
        "source_frames_total": int(source.images.shape[0]),
        "sampled_frames": sampled_frames,
        "mean_abs_timestamp_error_s": float(errors.mean()),
        "p95_abs_timestamp_error_s": float(np.percentile(errors, 95)),
        "max_abs_timestamp_error_s": float(errors.max()),
        "mean_selected_source_spacing_s": float(spacings.mean()),
        "min_selected_source_spacing_s": float(spacings.min()),
        "max_selected_source_spacing_s": float(spacings.max()),
        "target_spacing_s": 1.0 / TARGET_FPS,
        "duplicate_selected_indices": duplicate_indices,
        "timestamp_monotonic_violations": timestamp_monotonic_violations,
        "source_roll_wrap_crossings": int(source_wrap_crossings[0]),
        "source_pitch_wrap_crossings": int(source_wrap_crossings[1]),
        "roll_wrap_crossings": int(wrap_crossings[0]),
        "pitch_wrap_crossings": int(wrap_crossings[1]),
        "state_min": source.states.min(axis=0).astype(float).tolist(),
        "state_max": source.states.max(axis=0).astype(float).tolist(),
        "reconstruction_max_abs_error": reconstruction.max(axis=0).astype(float).tolist(),
        "reconstruction_mean_abs_error": reconstruction.mean(axis=0).astype(float).tolist(),
        "image_shape": list(image_samples[0].shape),
        "image_dtype": str(image_samples[0].dtype),
        "image_range": [
            int(min(sample.min() for sample in image_samples)),
            int(max(sample.max() for sample in image_samples)),
        ],
        "terminal_action_policy": "terminal state is target-only; no synthetic zero-action row",
        "cross_episode_actions": 0,
        "state_order": list(STATE_AXES),
        "action_order": list(ACTION_ORDER),
        "dataset_row_action_shape": [ACTION_DIM],
        "training_window_shape": [CHUNK_SIZE, ACTION_DIM],
        "episode_metrics": episode_metrics,
    }


def write_selected_episodes(
    source: SourceDataset,
    dataset: Any,
    source_fps: float,
    episode_ids: Sequence[int],
    task: str,
) -> tuple[int, int]:
    """Write rows to a LeRobot-like sink; injection keeps validation motorless."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyMissingError("Pillow is required to write image rows") from exc

    states = source.states
    starts = source.episode_starts
    written_frames = 0
    for episode_id in episode_ids:
        start = int(starts[episode_id - 1])
        end = int(source.episode_ends[episode_id - 1])
        selection = select_10hz_samples(end - start, source_fps)
        selected_states = states[start + selection.indices]
        actions = compute_incremental_actions(selected_states)
        for local_row, source_local_index in enumerate(selection.indices[:-1]):
            source_index = start + int(source_local_index)
            image = source.images.read_index(source_index)
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ConverterError(f"invalid RGB frame at source index {source_index}")
            dataset.add_frame(
                {
                    IMAGE_KEY: Image.fromarray(image),
                    STATE_KEY: selected_states[local_row].copy(),
                    ACTION_KEY: actions[local_row].copy(),
                    "task": task,
                }
            )
            written_frames += 1
        dataset.save_episode()
    return len(episode_ids), written_frames


def _prepare_output(output: Path, source: Path, project_root: Path, overwrite: bool) -> None:
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        return
    if not overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite explicitly")
    target = output.resolve()
    source_root = source.resolve()
    project = project_root.resolve()
    immutable_model = (project_root / "7_storage" / "Delta_Weights").resolve()
    historical_archive = (project_root / "etc" / "archive").resolve()
    if (
        target == project
        or target == source_root
        or target.is_relative_to(source_root)
        or target.is_relative_to(immutable_model)
        or target.is_relative_to(historical_archive)
        or target == Path(target.anchor)
        or target.is_symlink()
    ):
        raise ConverterError(f"refusing unsafe output replacement: {output}")
    if not target.is_dir() or (target / ".git").exists():
        raise ConverterError(f"refusing to replace non-directory or Git output: {output}")
    shutil.rmtree(target)
    output.parent.mkdir(parents=True, exist_ok=True)


def _create_lerobot_dataset(
    project_root: Path, output: Path, repo_id: str
) -> Any:
    vendored_source = project_root / "etc" / "third_party" / "lerobot" / "src"
    if not vendored_source.is_dir():
        raise DependencyMissingError(f"vendored LeRobot source not found: {vendored_source}")
    if str(vendored_source) not in sys.path:
        sys.path.insert(0, str(vendored_source))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyMissingError(
            "pinned LeRobot 0.6.1 runtime dependencies are not available in this Python environment"
        ) from exc
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=TARGET_FPS,
        features=build_lerobot_features(),
        robot_type="umi_tcp",
        root=output,
        use_videos=False,
    )


def convert(args: argparse.Namespace) -> dict[str, Any]:
    project_root = discover_project_root()
    validate_bundle_contract(project_root)
    source = load_yawfree_source(args.input_zarr)
    episode_ids = _selected_episode_ids(
        len(source.episode_ends), args.episodes, args.max_episodes
    )
    audit = audit_source(source, args.source_fps, episode_ids)
    if args.validate_only:
        return audit

    output = args.output_dir.resolve()
    _prepare_output(output, source.path, project_root, args.overwrite)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"temporary conversion output exists: {temporary}")
    dataset = _create_lerobot_dataset(project_root, temporary, args.repo_id)
    try:
        episodes, frames = write_selected_episodes(
            source, dataset, args.source_fps, episode_ids, args.task
        )
        dataset.finalize()
    except Exception:
        # Preserve partial output for diagnosis; never delete it implicitly.
        raise

    provenance = {
        **audit,
        "repo_id": args.repo_id,
        "output_path": str(output),
        "episodes_written": episodes,
        "frames_written": frames,
        "features": build_lerobot_features(),
        "historical_source_recovered": False,
    }
    metadata_path = temporary / "meta" / "conversion_provenance.json"
    metadata_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = discover_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-zarr",
        type=Path,
        default=project_root
        / "7_storage/zarr/replay_buffer_yawfree.zarr",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root
        / "7_storage"
        / "lerobot"
        / CANONICAL_DATASET_NAME,
    )
    parser.add_argument(
        "--repo-id",
        default=f"gopro_umi/{CANONICAL_DATASET_NAME}",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=SOURCE_FPS_DEFAULT,
        help="implicit source clock; audited GoPro files are 60000/1001 Hz",
    )
    parser.add_argument("--episodes", type=int, nargs="+", help="1-based episode IDs")
    parser.add_argument("--max-episodes", type=int, help="deterministic prefix smoke conversion")
    parser.add_argument("--task", default=TASK_DESCRIPTION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="audit source, sampling, actions, and bundle without creating output",
    )
    parser.add_argument("--report-json", type=Path, help="optional explicit report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = convert(args)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.report_json is not None:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(rendered + "\n")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
