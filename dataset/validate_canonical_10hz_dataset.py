#!/usr/bin/env python3
"""Validate the canonical 10 Hz LeRobot dataset against its yaw-free Zarr.

This validator compares every stored state/action/timestamp row with a fresh
deterministic source-to-10 Hz reconstruction.  Validation artifacts are only
written after all rows and at least 100 deterministic 15-action chunks pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PROJECT_ROOT / "dataset/convert_to_lerobot_10hz_incremental.py"
CANONICAL_DATASET_NAME = "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
CANONICAL_DATASET = PROJECT_ROOT / "artifacts" / "lerobot" / CANONICAL_DATASET_NAME
CANONICAL_SOURCE = PROJECT_ROOT / "artifacts" / "zarr" / "replay_buffer_yawfree.zarr"
REQUIRED_ARTIFACTS = (
    "incremental_action_contract.json",
    "resampling_validation.json",
    "incremental_validation.json",
    "step9a_audit_summary.json",
    "step9a_source_to_10hz_mapping.parquet",
)


def _load_converter() -> Any:
    spec = importlib.util.spec_from_file_location("canonical_10hz_converter", CONVERTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load converter: {CONVERTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class EpisodeRows:
    file: Path
    episode_index: int
    file_row_index: np.ndarray
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    index: np.ndarray


def _fixed_list(column: Any, width: int) -> np.ndarray:
    value = np.asarray(column.to_pylist(), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"expected fixed list width {width}, got {value.shape}")
    return value


def _read_episode_rows(dataset: Path, converter: Any) -> list[EpisodeRows]:
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {dataset / 'data'}")
    result: list[EpisodeRows] = []
    columns = [
        converter.STATE_KEY,
        converter.ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
    ]
    for file in files:
        table = pq.read_table(file, columns=columns)
        episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        all_state = _fixed_list(table[converter.STATE_KEY], converter.ACTION_DIM)
        all_action = _fixed_list(table[converter.ACTION_KEY], converter.ACTION_DIM)
        all_timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        all_frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        all_index = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        for episode_index in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode_index)
            result.append(
                EpisodeRows(
                    file=file,
                    episode_index=int(episode_index),
                    file_row_index=rows,
                    state=all_state[rows],
                    action=all_action[rows],
                    timestamp=all_timestamp[rows],
                    frame_index=all_frame_index[rows],
                    index=all_index[rows],
                )
            )
    result.sort(key=lambda rows: rows.episode_index)
    return result


def _validate_info(dataset: Path, converter: Any) -> dict[str, Any]:
    if dataset.name != CANONICAL_DATASET_NAME:
        raise ValueError(
            f"dataset basename must be {CANONICAL_DATASET_NAME!r}, got {dataset.name!r}"
        )
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    state = features.get(converter.STATE_KEY, {})
    action = features.get(converter.ACTION_KEY, {})
    image = features.get(converter.IMAGE_KEY, {})
    failures = []
    if info.get("fps") != converter.TARGET_FPS:
        failures.append(f"fps={info.get('fps')}")
    if state.get("shape") != [6] or state.get("names", {}).get("axes") != list(
        converter.STATE_AXES
    ):
        failures.append(f"state feature={state}")
    if action.get("shape") != [6] or action.get("names", {}).get("axes") != list(
        converter.ACTION_ORDER
    ):
        failures.append(f"action feature={action}")
    if image.get("shape") != [224, 224, 3]:
        failures.append(f"image feature={image}")
    if any("yaw" in str(axis).lower() for axis in state.get("names", {}).get("axes", [])):
        failures.append("yaw axis present in state")
    if failures:
        raise ValueError("dataset info contract failure: " + "; ".join(failures))
    return info


def _validate_images(
    episodes: list[EpisodeRows], mapping_rows: list[dict[str, Any]], source: Any, image_key: str
) -> int:
    sample_positions = np.linspace(0, len(mapping_rows) - 1, 12, dtype=np.int64)
    by_episode = {rows.episode_index: rows for rows in episodes}
    cache: dict[Path, list[Any]] = {}
    for position in sample_positions:
        mapping = mapping_rows[int(position)]
        rows = by_episode[int(mapping["episode_index"])]
        values = cache.get(rows.file)
        if values is None:
            values = pq.read_table(rows.file, columns=[image_key])[image_key].to_pylist()
            cache[rows.file] = values
        encoded = values[int(mapping["file_row_index"])]
        payload = encoded.get("bytes") if isinstance(encoded, dict) else None
        if payload is None:
            raise ValueError(f"embedded image bytes missing in {rows.file}")
        actual = np.asarray(Image.open(BytesIO(payload)).convert("RGB"), dtype=np.uint8)
        expected = source.images.read_index(int(mapping["source_global_index"]))
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"RGB/source mapping mismatch at output {mapping['output_index']}"
            )
    return len(sample_positions)


def validate_dataset(
    dataset: Path, source_path: Path, *, chunk_checks: int = 100
) -> dict[str, Any]:
    converter = _load_converter()
    dataset = dataset.resolve()
    source_path = source_path.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    info = _validate_info(dataset, converter)
    source = converter.load_yawfree_source(source_path)
    episodes = _read_episode_rows(dataset, converter)
    if len(episodes) != len(source.episode_ends):
        raise ValueError(
            f"episode count mismatch: dataset={len(episodes)} source={len(source.episode_ends)}"
        )

    starts = source.episode_starts
    mapping_rows: list[dict[str, Any]] = []
    chunk_candidates: list[tuple[int, int]] = []
    expected_states_by_episode: list[np.ndarray] = []
    max_state_error = 0.0
    max_action_error = 0.0
    max_timestamp_error = 0.0
    max_reconstruction_error = 0.0
    global_expected_index = 0

    for episode_id, actual in enumerate(episodes, start=1):
        if actual.episode_index != episode_id - 1:
            raise ValueError(
                f"episode index sequence mismatch: expected {episode_id - 1}, got {actual.episode_index}"
            )
        start = int(starts[episode_id - 1])
        end = int(source.episode_ends[episode_id - 1])
        selection = converter.select_10hz_samples(end - start, converter.SOURCE_FPS_DEFAULT)
        selected_states = source.states[start + selection.indices]
        expected_actions = converter.compute_incremental_actions(selected_states)
        expected_states = selected_states[:-1]
        expected_timestamps = selection.target_timestamps[:-1]
        rows = len(expected_actions)
        if actual.state.shape != (rows, 6) or actual.action.shape != (rows, 6):
            raise ValueError(
                f"episode {episode_id} row shape mismatch: "
                f"state={actual.state.shape} action={actual.action.shape} expected={rows}"
            )
        if not np.isfinite(actual.state).all() or not np.isfinite(actual.action).all():
            raise ValueError(f"episode {episode_id} contains NaN/Inf")
        state_error = float(np.max(np.abs(actual.state - expected_states)))
        action_error = float(np.max(np.abs(actual.action - expected_actions)))
        timestamp_error = float(np.max(np.abs(actual.timestamp - expected_timestamps)))
        if state_error > 1e-7 or action_error > 1e-7 or timestamp_error > 1e-6:
            raise ValueError(
                f"episode {episode_id} source comparison failed: "
                f"state={state_error} action={action_error} timestamp={timestamp_error}"
            )
        if not np.array_equal(actual.frame_index, np.arange(rows, dtype=np.int64)):
            raise ValueError(f"episode {episode_id} frame_index is not contiguous")
        if not np.array_equal(
            actual.index, np.arange(global_expected_index, global_expected_index + rows)
        ):
            raise ValueError(f"episode {episode_id} global index is not contiguous")
        if rows > 1 and not np.allclose(np.diff(actual.timestamp), 0.1, atol=2e-6, rtol=0.0):
            raise ValueError(f"episode {episode_id} timestamps are not 10 Hz")
        reconstructed = actual.state.astype(np.float64) + actual.action.astype(np.float64)
        reconstruction_error = float(
            np.max(np.abs(reconstructed - selected_states[1:].astype(np.float64)))
        )
        if reconstruction_error > 1e-7:
            raise ValueError(
                f"episode {episode_id} incremental reconstruction failed: {reconstruction_error}"
            )
        max_state_error = max(max_state_error, state_error)
        max_action_error = max(max_action_error, action_error)
        max_timestamp_error = max(max_timestamp_error, timestamp_error)
        max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
        expected_states_by_episode.append(selected_states)
        chunk_candidates.extend((episode_id - 1, anchor) for anchor in range(max(0, rows - 14)))
        for local_row in range(rows):
            source_local = int(selection.indices[local_row])
            source_next_local = int(selection.indices[local_row + 1])
            mapping_rows.append(
                {
                    "output_index": int(actual.index[local_row]),
                    "episode_index": episode_id - 1,
                    "episode_id": episode_id,
                    "episode_row_index": local_row,
                    "file_row_index": int(actual.file_row_index[local_row]),
                    "source_global_index": start + source_local,
                    "source_next_global_index": start + source_next_local,
                    "source_local_index": source_local,
                    "source_next_local_index": source_next_local,
                    "target_timestamp_s": float(selection.target_timestamps[local_row]),
                    "target_next_timestamp_s": float(
                        selection.target_timestamps[local_row + 1]
                    ),
                    "selected_source_timestamp_s": float(
                        selection.selected_source_timestamps[local_row]
                    ),
                    "selected_next_source_timestamp_s": float(
                        selection.selected_source_timestamps[local_row + 1]
                    ),
                    "abs_timestamp_error_s": float(
                        abs(selection.timestamp_errors[local_row])
                    ),
                }
            )
        global_expected_index += rows

    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json episode count disagrees with stored data")
    if int(info.get("total_frames", -1)) != len(mapping_rows):
        raise ValueError("info.json frame count disagrees with stored data")
    if len(mapping_rows) <= 0:
        raise ValueError("canonical transition count must be positive")
    if len(chunk_candidates) < chunk_checks:
        raise ValueError(f"only {len(chunk_candidates)} valid 15-action chunks available")

    rng = np.random.default_rng(20260814)
    selected_chunks = rng.choice(len(chunk_candidates), size=chunk_checks, replace=False)
    chunk_failures = 0
    max_chunk_reconstruction_error = 0.0
    for choice in selected_chunks:
        episode_index, anchor = chunk_candidates[int(choice)]
        actual = episodes[episode_index]
        expected_states = expected_states_by_episode[episode_index]
        actions = actual.action[anchor : anchor + 15].astype(np.float64)
        reconstructed = expected_states[anchor].astype(np.float64) + np.cumsum(actions, axis=0)
        expected = expected_states[anchor + 1 : anchor + 16].astype(np.float64)
        error = float(np.max(np.abs(reconstructed - expected)))
        max_chunk_reconstruction_error = max(max_chunk_reconstruction_error, error)
        timestamps = actual.timestamp[anchor : anchor + 15]
        timing_ok = (
            len(timestamps) == 15
            and abs((timestamps[0] + 0.1) - (anchor + 1) / 10.0) <= 1e-6
            and abs((timestamps[14] + 0.1) - (anchor + 15) / 10.0) <= 1e-6
        )
        if error > 1e-6 or not timing_ok or not np.isfinite(actions).all():
            chunk_failures += 1
    if chunk_failures:
        raise ValueError(f"{chunk_failures}/{chunk_checks} deterministic chunk checks failed")

    image_checks = _validate_images(episodes, mapping_rows, source, converter.IMAGE_KEY)
    timestamp_errors = np.asarray(
        [row["abs_timestamp_error_s"] for row in mapping_rows], dtype=np.float64
    )
    contract = {
        "status": "VERIFIED",
        "dataset_name": CANONICAL_DATASET_NAME,
        "action_dim": 6,
        "state_dim": 6,
        "state_order": list(converter.STATE_AXES),
        "action_order": list(converter.ACTION_ORDER),
        "yaw_included": False,
        "action_semantics": "STEP-TO-STEP INCREMENTAL",
        "action_rate_hz": 10,
        "action_dt_s": 0.1,
        "chunk_size": 15,
        "n_action_steps": 15,
        "horizon_s": 1.5,
        "terminal_action_policy": "no synthetic terminal row",
        "units": {"xyz": "meter", "roll_pitch": "radian", "gripper": "normalized [0,1]"},
        "recovery": {"status": "DEFERRED", "baseline_training_blocker": False},
        "step9a_audit": {"status": "PASS", "artifact": "step9a_audit_summary.json"},
    }
    resampling = {
        "status": "VERIFIED",
        "source_zarr": "artifacts/zarr/replay_buffer_yawfree.zarr",
        "source_fps": converter.SOURCE_FPS_DEFAULT,
        "target_fps": 10,
        "target_dt_s": 0.1,
        "sampling_rule": "nearest source timestamp to n/10, deterministic half-up ties",
        "episodes": len(episodes),
        "source_frames": int(source.images.shape[0]),
        "output_transitions": len(mapping_rows),
        "mapping_rows": len(mapping_rows),
        "duplicate_output_indices": len(mapping_rows)
        - len({row["output_index"] for row in mapping_rows}),
        "max_abs_source_timestamp_error_s": float(timestamp_errors.max()),
        "mean_abs_source_timestamp_error_s": float(timestamp_errors.mean()),
        "timestamp_monotonic": True,
    }
    incremental = {
        "status": "VERIFIED",
        "all_rows_compared": len(mapping_rows),
        "random_seed": 20260814,
        "random_samples": chunk_checks,
        "chunk_length": 15,
        "failures": chunk_failures,
        "max_state_source_error": max_state_error,
        "max_action_source_error": max_action_error,
        "max_dataset_timestamp_error_s": max_timestamp_error,
        "max_single_step_reconstruction_error": max_reconstruction_error,
        "max_chunk_reconstruction_error": max_chunk_reconstruction_error,
        "image_source_mapping_checks": image_checks,
        "finite": True,
        "a0_timing": "T0 -> T0+0.1",
        "a14_timing": "T0+1.4 -> T0+1.5",
    }
    step9a = {
        "status": "PASS",
        "ten_hz_suitability_verdict": "PASS",
        "structural": {
            "total_found": len(episodes),
            "total_valid": len(episodes),
            "source_frames": int(source.images.shape[0]),
            "output_transitions": len(mapping_rows),
            "pass": True,
        },
        "source_to_10hz_mapping": "step9a_source_to_10hz_mapping.parquet",
        "mapping_rows": len(mapping_rows),
    }
    return {
        "dataset": str(dataset),
        "source": str(source_path),
        "fps": int(info["fps"]),
        "episodes": len(episodes),
        "frames": len(mapping_rows),
        "state_dim": 6,
        "action_dim": 6,
        "finite": True,
        "timestamps": "PASS",
        "incremental_semantics": "PASS",
        "chunks_checked": chunk_checks,
        "chunk_failures": chunk_failures,
        "image_mapping_checks": image_checks,
        "contract": contract,
        "resampling": resampling,
        "incremental": incremental,
        "step9a": step9a,
        "mapping_rows": mapping_rows,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_artifacts(dataset: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    meta = dataset / "meta"
    targets = [meta / name for name in REQUIRED_ARTIFACTS]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "validation artifact(s) already exist; pass --overwrite-artifacts after validation: "
            + ", ".join(str(path) for path in existing)
        )
    _write_json_atomic(meta / "incremental_action_contract.json", report["contract"])
    _write_json_atomic(meta / "resampling_validation.json", report["resampling"])
    _write_json_atomic(meta / "incremental_validation.json", report["incremental"])
    _write_json_atomic(meta / "step9a_audit_summary.json", report["step9a"])
    mapping_path = meta / "step9a_source_to_10hz_mapping.parquet"
    temporary = mapping_path.with_name(mapping_path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(report["mapping_rows"]), temporary)
    temporary.replace(mapping_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=CANONICAL_DATASET)
    parser.add_argument("--source-zarr", type=Path, default=CANONICAL_SOURCE)
    parser.add_argument("--chunk-checks", type=int, default=100)
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate_dataset(
            args.dataset, args.source_zarr, chunk_checks=args.chunk_checks
        )
        if args.write_artifacts:
            write_artifacts(args.dataset, report, overwrite=args.overwrite_artifacts)
        rendered = {key: value for key, value in report.items() if key != "mapping_rows"}
        rendered["artifacts_written"] = bool(args.write_artifacts)
        print(json.dumps(rendered, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
