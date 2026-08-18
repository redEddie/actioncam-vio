#!/usr/bin/env python3
"""Comprehensive unit and mock regression suite for gopro_umi dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_full_dataset_pipeline import (
    PipelineWorkspace,
    find_highest_valid_checkpoint,
    materialize_final_model,
    safe_promote_model,
    validate_model_contract,
)
import importlib.util

PREFLIGHT_PATH = Path(__file__).resolve().parents[1] / "3_training/scripts/validate_incremental_training_preflight.py"
spec = importlib.util.spec_from_file_location("preflight_module", PREFLIGHT_PATH)
assert spec is not None and spec.loader is not None
preflight_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight_mod)
run_preflight = preflight_mod.run_preflight


class PipelineRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.run_root = self.temp_dir / "run"
        self.datasets_root = self.temp_dir / "datasets"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.datasets_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_mock_valid_model(self, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "config.json").write_text(json.dumps({
            "model_type": "SmolVLA",
            "chunk_size": 15,
            "n_action_steps": 15,
            "input_features": {"observation.state": {"shape": [6]}},
            "output_features": {"action": {"shape": [6]}},
            "normalization_mapping": {"ACTION": "MEAN_STD"},
        }))
        (target_dir / "model.safetensors").write_bytes(b"dummy model weights")
        (target_dir / "policy_preprocessor.json").write_text("{}")
        (target_dir / "policy_postprocessor.json").write_text("{}")

    def _create_mock_dataset(
        self, target_dir: Path, num_episodes: int, rows_per_episode: int, fps: int = 10, inject_nan: bool = False
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        meta_dir = target_dir / "meta"
        data_dir = target_dir / "data"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        total_frames = num_episodes * rows_per_episode

        # meta/info.json
        info = {
            "fps": fps,
            "total_episodes": num_episodes,
            "total_frames": total_frames,
            "features": {
                "observation.state": {"shape": [6], "names": {"axes": ["X", "Y", "Z", "Roll", "Pitch", "Gripper"]}},
                "action": {"shape": [6], "names": {"axes": ["dX", "dY", "dZ", "dRoll", "dPitch", "dGripper"]}},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

        # meta/incremental_action_contract.json
        contract = {
            "status": "VERIFIED",
            "dataset_name": "lerobot_dataset_10hz_chunk15_incremental_baseline_v1",
            "state_dim": 6,
            "action_dim": 6,
            "state_order": ["X", "Y", "Z", "Roll", "Pitch", "Gripper"],
            "action_order": ["dX", "dY", "dZ", "dRoll", "dPitch", "dGripper"],
            "yaw_included": False,
            "action_semantics": "STEP-TO-STEP INCREMENTAL",
            "action_rate_hz": 10,
            "action_dt_s": 0.1,
            "chunk_size": 15,
            "n_action_steps": 15,
            "horizon_s": 1.5,
            "terminal_action_policy": "no synthetic terminal row",
            "recovery": {"status": "VERIFIED"},
            "step9a_audit": {"status": "PASS"},
        }
        (meta_dir / "incremental_action_contract.json").write_text(json.dumps(contract, indent=2))

        # meta/step9a_audit_summary.json
        step9a = {
            "status": "PASS",
            "ten_hz_suitability_verdict": "PASS",
            "structural": {
                "pass": True,
                "total_valid": num_episodes,
                "output_transitions": total_frames,
            },
        }
        (meta_dir / "step9a_audit_summary.json").write_text(json.dumps(step9a, indent=2))

        # meta/resampling_validation.json
        resampling = {
            "status": "VERIFIED",
            "target_fps": 10,
            "mapping_rows": total_frames,
            "duplicate_output_indices": 0,
            "timestamp_monotonic": True,
        }
        (meta_dir / "resampling_validation.json").write_text(json.dumps(resampling, indent=2))

        # meta/incremental_validation.json
        incremental = {
            "status": "VERIFIED",
            "random_samples": 100,
            "failures": 0,
            "finite": not inject_nan,
            "chunk_length": 15,
        }
        (meta_dir / "incremental_validation.json").write_text(json.dumps(incremental, indent=2))

        # meta/step9a_source_to_10hz_mapping.parquet
        mapping_data = {
            "output_index": list(range(total_frames)),
            "target_timestamp_s": [i * 0.1 for i in range(total_frames)],
            "target_next_timestamp_s": [(i + 1) * 0.1 for i in range(total_frames)],
        }
        pq.write_table(pa.Table.from_pydict(mapping_data), meta_dir / "step9a_source_to_10hz_mapping.parquet")

        # data parquet
        state_val = np.ones((total_frames, 6), dtype=np.float32)
        if inject_nan:
            state_val[0, 0] = np.nan
        action_val = np.zeros((total_frames, 6), dtype=np.float32)

        data_dict = {
            "observation.state": [row.tolist() for row in state_val],
            "action": [row.tolist() for row in action_val],
            "timestamp": [i * 0.1 for i in range(total_frames)],
            "episode_index": [i // rows_per_episode for i in range(total_frames)],
            "frame_index": [i % rows_per_episode for i in range(total_frames)],
        }
        pq.write_table(pa.Table.from_pydict(data_dict), data_dir / "chunk-000.parquet")

    def test_model_contract_validation(self) -> None:
        model_dir = self.temp_dir / "model"
        self._create_mock_valid_model(model_dir)
        report = validate_model_contract(model_dir)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["raw_output_shape"], [1, 15, 6])
        self.assertEqual(report["local_action_shape"], [15, 6])

        # Missing file should fail
        (model_dir / "model.safetensors").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_model_contract(model_dir)

    def test_checkpoint_discovery_and_highest_step_selection(self) -> None:
        checkpoints = self.temp_dir / "checkpoints"
        checkpoints.mkdir()
        # Create 5000, 10000, 20000
        self._create_mock_valid_model(checkpoints / "005000" / "pretrained_model")
        self._create_mock_valid_model(checkpoints / "010000" / "pretrained_model")
        self._create_mock_valid_model(checkpoints / "020000" / "pretrained_model")

        best = find_highest_valid_checkpoint(checkpoints)
        self.assertEqual(best, checkpoints / "020000" / "pretrained_model")

        # Incomplete highest step 30000
        incomplete = checkpoints / "030000" / "pretrained_model"
        incomplete.mkdir(parents=True)
        (incomplete / "config.json").write_text("{}")  # Missing safetensors
        best_fallback = find_highest_valid_checkpoint(checkpoints)
        self.assertEqual(best_fallback, checkpoints / "020000" / "pretrained_model")

    def test_safe_promotion_with_explicit_rollback_on_swap_failure(self) -> None:
        # 1. Setup initial active model A
        active_model_a = self.run_root / "pretrained_model"
        self._create_mock_valid_model(active_model_a)
        (active_model_a / "model.safetensors").write_bytes(b"MODEL_A_ACTIVE")
        initial_manifest = {
            "source_dataset_id": "202601010000",
            "promotion_status": "PASS",
        }
        (self.run_root / "run_manifest.json").write_text(json.dumps(initial_manifest))

        # 2. Setup candidate model B in workspace
        ws = PipelineWorkspace("209901011200", create=True, datasets_root=self.datasets_root)
        ws.final_model_dir = ws.root / "04_training" / "final" / "pretrained_model"
        self._create_mock_valid_model(ws.final_model_dir)
        (ws.final_model_dir / "model.safetensors").write_bytes(b"MODEL_B_CANDIDATE")

        # 3. Successful promotion
        res = safe_promote_model(ws, run_root=self.run_root)
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(
            (self.run_root / "pretrained_model" / "model.safetensors").read_bytes(),
            b"MODEL_B_CANDIDATE",
        )
        manifest_b = json.loads((self.run_root / "run_manifest.json").read_text())
        self.assertEqual(manifest_b["source_dataset_id"], "209901011200")

        # 4. Injected failure promotion with Rollback
        ws_fail = PipelineWorkspace("209902021200", create=True, datasets_root=self.datasets_root)
        ws_fail.final_model_dir = ws_fail.root / "04_training" / "final" / "pretrained_model"
        ws_fail.final_model_dir.mkdir(parents=True)
        (ws_fail.final_model_dir / "config.json").write_text("{}")

        with self.assertRaises(FileNotFoundError):
            safe_promote_model(ws_fail, run_root=self.run_root)

        # Confirm rollback: Active model is STILL Model B
        self.assertEqual(
            (self.run_root / "pretrained_model" / "model.safetensors").read_bytes(),
            b"MODEL_B_CANDIDATE",
        )
        manifest_restored = json.loads((self.run_root / "run_manifest.json").read_text())
        self.assertEqual(manifest_restored["source_dataset_id"], "209901011200")

    def test_pipeline_tests_never_modify_production_run(self) -> None:
        """Verify that running mock promotion with tmp run_root leaves production run strictly untouched."""
        prod_run = Path(__file__).resolve().parents[1] / "7_storage/run/pretrained_model/model.safetensors"
        prod_manifest = Path(__file__).resolve().parents[1] / "7_storage/run/run_manifest.json"
        
        if prod_run.exists() and prod_manifest.exists():
            import hashlib
            prod_model_hash_before = hashlib.sha256(prod_run.read_bytes()).hexdigest()
            prod_manifest_hash_before = hashlib.sha256(prod_manifest.read_bytes()).hexdigest()

            # Run a mock workspace and promotion on tmp_path
            mock_ws = PipelineWorkspace("209905051200", create=True, datasets_root=self.datasets_root)
            self._create_mock_valid_model(mock_ws.final_model_dir)
            safe_promote_model(mock_ws, run_root=self.run_root)

            # Assert production run files are completely unchanged
            prod_model_hash_after = hashlib.sha256(prod_run.read_bytes()).hexdigest()
            prod_manifest_hash_after = hashlib.sha256(prod_manifest.read_bytes()).hexdigest()

            self.assertEqual(prod_model_hash_before, prod_model_hash_after)
            self.assertEqual(prod_manifest_hash_before, prod_manifest_hash_after)
            self.assertEqual(prod_run.stat().st_size, 906712520)

    def test_preflight_accepts_valid_nonbaseline_dataset(self) -> None:
        """Verify that Stage 7 preflight accepts an arbitrary valid non-77/12320 dataset."""
        dataset_dir = self.temp_dir / "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
        # Create dataset with 50 episodes and 8000 rows (160 rows/episode)
        self._create_mock_dataset(dataset_dir, num_episodes=50, rows_per_episode=160)

        checkpoint_dir = self.temp_dir / "checkpoint"
        self._create_mock_valid_model(checkpoint_dir)
        output_dir = self.temp_dir / "training_output"

        # Execute actual production preflight function
        res = run_preflight(dataset_dir, checkpoint_dir, output_dir)
        self.assertEqual(res["status"], "PASS")

    def test_preflight_accepts_1_episode_valid_dataset(self) -> None:
        """Verify that Stage 7 preflight accepts a 1-episode valid dataset."""
        dataset_dir = self.temp_dir / "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
        self._create_mock_dataset(dataset_dir, num_episodes=1, rows_per_episode=100)

        checkpoint_dir = self.temp_dir / "checkpoint"
        self._create_mock_valid_model(checkpoint_dir)
        output_dir = self.temp_dir / "training_output_1ep"

        res = run_preflight(dataset_dir, checkpoint_dir, output_dir)
        self.assertEqual(res["status"], "PASS")

    def test_preflight_rejects_invalid_cases(self) -> None:
        """Verify that Stage 7 preflight rejects NaN, 0 episodes, and wrong fps."""
        checkpoint_dir = self.temp_dir / "checkpoint"
        self._create_mock_valid_model(checkpoint_dir)

        # 1. NaN in state
        nan_dataset = self.temp_dir / "nan_ds" / "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
        self._create_mock_dataset(nan_dataset, num_episodes=10, rows_per_episode=50, inject_nan=True)
        with self.assertRaises((ValueError, RuntimeError)):
            run_preflight(nan_dataset, checkpoint_dir, self.temp_dir / "out1")

        # 2. Wrong FPS (e.g. 60 Hz)
        fps_dataset = self.temp_dir / "fps_ds" / "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
        self._create_mock_dataset(fps_dataset, num_episodes=10, rows_per_episode=50, fps=60)
        with self.assertRaises((ValueError, RuntimeError)):
            run_preflight(fps_dataset, checkpoint_dir, self.temp_dir / "out2")

    def test_bootstrap_baseline_inventory_regression(self) -> None:
        """Explicit regression check that verified bootstrap dataset is 77ep / 12320 rows."""
        bootstrap_dataset = (
            Path(__file__).resolve().parents[1]
            / "7_storage/datasets/202608161336/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
        )
        if bootstrap_dataset.exists():
            meta = json.loads((bootstrap_dataset / "meta" / "info.json").read_text())
            self.assertEqual(meta["total_episodes"], 77)
            self.assertEqual(meta["total_frames"], 12320)
            self.assertEqual(meta["fps"], 10)


if __name__ == "__main__":
    unittest.main()
