#!/usr/bin/env python3
"""Fail-closed, hardware-free preflight for canonical incremental training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DATASET_NAME = "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
DEFAULT_DATASET = PROJECT_ROOT / "7_storage" / "lerobot" / CANONICAL_DATASET_NAME
DEFAULT_CHECKPOINT = PROJECT_ROOT / "7_storage" / "Delta_Weights"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "3_training" / "smolvla_10hz_chunk15_incremental_baseline_v1"
)
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_ORDER = ["X", "Y", "Z", "Roll", "Pitch", "Gripper"]
ACTION_ORDER = ["dX", "dY", "dZ", "dRoll", "dPitch", "dGripper"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _feature_axes(feature: dict[str, Any]) -> list[str] | None:
    names = feature.get("names", {})
    return names.get("axes") if isinstance(names, dict) else None


def _validate_numeric_parquet(dataset: Path) -> tuple[int, int, float]:
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no dataset parquet files under {dataset / 'data'}")
    total_rows = 0
    episodes: set[int] = set()
    worst_dt_error = 0.0
    for file in files:
        table = pq.read_table(
            file,
            columns=[STATE_KEY, ACTION_KEY, "timestamp", "episode_index", "frame_index"],
        )
        states = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        episode_values = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if states.shape != (len(table), 6) or actions.shape != (len(table), 6):
            raise ValueError(f"wrong state/action shape in {file}: {states.shape}, {actions.shape}")
        if not (np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(timestamps).all()):
            raise ValueError(f"NaN/Inf found in {file}")
        for episode_index in np.unique(episode_values):
            rows = np.flatnonzero(episode_values == episode_index)
            if not np.array_equal(frame_indices[rows], np.arange(len(rows), dtype=np.int64)):
                raise ValueError(
                    f"non-contiguous frame_index for episode {episode_index} in {file}"
                )
            episode_timestamps = timestamps[rows]
            if len(episode_timestamps) > 1:
                error = float(np.max(np.abs(np.diff(episode_timestamps) - 0.1)))
                worst_dt_error = max(worst_dt_error, error)
                if error > 2e-6:
                    raise ValueError(
                        f"non-10 Hz timestamps for episode {episode_index} in {file}: "
                        f"worst error {error}"
                    )
            episodes.add(int(episode_index))
        total_rows += len(table)
    if episodes != set(range(len(episodes))):
        raise ValueError("episode indices are not contiguous from zero")
    return total_rows, len(episodes), worst_dt_error


def _assert_training_output_is_safe(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir() or output.is_symlink():
        raise FileExistsError(f"training output path exists and is not a valid directory: {output}")

    allowed_scaffolding = {"checkpoints", "final"}
    for child in output.iterdir():
        if child.name not in allowed_scaffolding:
            raise FileExistsError(
                f"refusing to overwrite training output containing unexpected entry: {child}"
            )
        if not child.is_dir() or child.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite training output with invalid scaffolding: {child}"
            )
        if any(child.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite training output with non-empty scaffolding: {child}"
            )


def run_preflight(dataset: Path, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output = output.resolve()
    if dataset.name != CANONICAL_DATASET_NAME:
        raise ValueError(
            f"training refuses noncanonical dataset basename {dataset.name!r}; "
            f"expected {CANONICAL_DATASET_NAME!r}"
        )
    required = {
        "info": dataset / "meta" / "info.json",
        "contract": dataset / "meta" / "incremental_action_contract.json",
        "resampling": dataset / "meta" / "resampling_validation.json",
        "incremental": dataset / "meta" / "incremental_validation.json",
        "step9a_summary": dataset / "meta" / "step9a_audit_summary.json",
        "step9a_mapping": dataset / "meta" / "step9a_source_to_10hz_mapping.parquet",
        "checkpoint_config": checkpoint_path / "config.json",
        "checkpoint_weights": checkpoint_path / "model.safetensors",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("preflight missing required files: " + ", ".join(missing))
    _assert_training_output_is_safe(output)

    info = _read_json(required["info"])
    contract = _read_json(required["contract"])
    resampling = _read_json(required["resampling"])
    incremental = _read_json(required["incremental"])
    step9a = _read_json(required["step9a_summary"])
    checkpoint = _read_json(required["checkpoint_config"])
    features = info.get("features", {})
    state = features.get(STATE_KEY, {})
    action = features.get(ACTION_KEY, {})
    failures: list[str] = []

    if info.get("fps") != 10:
        failures.append(f"dataset fps={info.get('fps')} not 10")
    total_episodes = info.get("total_episodes")
    total_frames = info.get("total_frames")
    if not isinstance(total_episodes, int) or total_episodes <= 0:
        failures.append(f"dataset total_episodes must be positive integer, got {total_episodes}")
    if not isinstance(total_frames, int) or total_frames <= 0:
        failures.append(f"dataset total_frames must be positive integer, got {total_frames}")
    if state.get("shape") != [6] or _feature_axes(state) != STATE_ORDER:
        failures.append(f"dataset state schema invalid: {state}")
    if action.get("shape") != [6] or _feature_axes(action) != ACTION_ORDER:
        failures.append(f"dataset action schema invalid: {action}")
    if any("yaw" in axis.lower() for axis in STATE_ORDER + ACTION_ORDER):
        failures.append("canonical axis constants unexpectedly contain yaw")

    expected_contract = {
        "dataset_name": CANONICAL_DATASET_NAME,
        "state_dim": 6,
        "action_dim": 6,
        "state_order": STATE_ORDER,
        "action_order": ACTION_ORDER,
        "yaw_included": False,
        "action_semantics": "STEP-TO-STEP INCREMENTAL",
        "action_rate_hz": 10,
        "action_dt_s": 0.1,
        "chunk_size": 15,
        "n_action_steps": 15,
        "horizon_s": 1.5,
        "terminal_action_policy": "no synthetic terminal row",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            failures.append(f"contract {key}={contract.get(key)!r}, expected {expected!r}")
    if contract.get("status") != "VERIFIED":
        failures.append(f"dataset contract status={contract.get('status')}")
    recovery = contract.get("recovery", {})
    if recovery.get("status") not in {"VERIFIED", "DEFERRED"}:
        failures.append(f"recovery status={recovery.get('status')}")
    if recovery.get("status") == "DEFERRED" and recovery.get("baseline_training_blocker") is not False:
        failures.append("deferred recovery is not explicitly non-blocking for baseline")
    if contract.get("step9a_audit", {}).get("status") != "PASS":
        failures.append(f"contract STEP 9A status={contract.get('step9a_audit')}")

    structural = step9a.get("structural", {})
    if (
        step9a.get("status") != "PASS"
        or step9a.get("ten_hz_suitability_verdict") != "PASS"
        or not structural.get("pass")
        or structural.get("total_valid") != total_episodes
        or structural.get("output_transitions") != total_frames
    ):
        failures.append(f"STEP 9A structural/suitability audit is not PASS for {total_episodes} episodes / {total_frames} frames")
    if (
        resampling.get("status") != "VERIFIED"
        or resampling.get("target_fps") != 10
        or resampling.get("mapping_rows") != total_frames
        or resampling.get("duplicate_output_indices") != 0
        or resampling.get("timestamp_monotonic") is not True
    ):
        failures.append(f"10 Hz resampling validation invalid: {resampling}")
    if (
        incremental.get("status") != "VERIFIED"
        or incremental.get("random_samples", 0) < 100
        or incremental.get("failures") != 0
        or incremental.get("finite") is not True
        or incremental.get("chunk_length") != 15
    ):
        failures.append(f"incremental validation invalid: {incremental}")

    mapping = pq.read_table(required["step9a_mapping"])
    if len(mapping) != total_frames:
        failures.append(f"source-to-10Hz mapping rows={len(mapping)}, expected {total_frames}")
    else:
        indices = np.asarray(mapping["output_index"].to_numpy(), dtype=np.int64)
        current_time = np.asarray(mapping["target_timestamp_s"].to_numpy(), dtype=np.float64)
        next_time = np.asarray(mapping["target_next_timestamp_s"].to_numpy(), dtype=np.float64)
        if not np.array_equal(indices, np.arange(total_frames, dtype=np.int64)):
            failures.append("source-to-10Hz output indices are not unique contiguous values")
        if not (np.isfinite(current_time).all() and np.isfinite(next_time).all()):
            failures.append("source-to-10Hz mapping contains NaN/Inf timestamps")
        elif float(np.max(np.abs((next_time - current_time) - 0.1))) > 1e-6:
            failures.append("source-to-10Hz mapping does not encode 0.1 s transitions")

    rows, episodes, worst_dt_error = _validate_numeric_parquet(dataset)
    if rows != total_frames or episodes != total_episodes:
        failures.append(f"stored parquet inventory rows={rows} episodes={episodes}, expected rows={total_frames} episodes={total_episodes}")
    if worst_dt_error > 1e-6:
        failures.append(f"parquet dataset worst dt error={worst_dt_error} exceeds tolerance")

    output_features = checkpoint.get("output_features", {}).get(ACTION_KEY, {})
    input_features = checkpoint.get("input_features", {}).get(STATE_KEY, {})
    if output_features.get("shape") != [6] or input_features.get("shape") != [6]:
        failures.append("source checkpoint state/action feature is not 6D")
    if checkpoint.get("chunk_size") != 15 or checkpoint.get("n_action_steps") != 15:
        failures.append("source checkpoint chunk/n_action_steps is not 15/15")
    if checkpoint.get("normalization_mapping", {}).get("ACTION") != "MEAN_STD":
        failures.append("source checkpoint action normalization is not MEAN_STD")
    if failures:
        raise RuntimeError("TRAINING BLOCKED:\n- " + "\n- ".join(failures))
    return {
        "status": "PASS",
        "project_root": str(PROJECT_ROOT),
        "dataset": str(dataset),
        "checkpoint": str(checkpoint_path),
        "output": str(output),
        "fps": 10,
        "episodes": episodes,
        "frames": rows,
        "state_dim": 6,
        "action_dim": 6,
        "chunk_size": 15,
        "horizon_s": 1.5,
        "yaw_absent": True,
        "finite": True,
        "worst_timestamp_spacing_error_s": worst_dt_error,
        "training_steps_executed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_preflight(args.dataset, args.checkpoint, args.output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
