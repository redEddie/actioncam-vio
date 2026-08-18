from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_defaults_share_canonical_storage_contract() -> None:
    launcher = _load("test_train_delta_module", "3_training/scripts/train_delta.py")
    preflight = _load(
        "test_training_preflight_module",
        "3_training/scripts/validate_incremental_training_preflight.py",
    )
    expected = (
        PROJECT_ROOT
        / "7_storage/lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
    )
    assert launcher.PROJECT_ROOT == PROJECT_ROOT
    assert preflight.PROJECT_ROOT == PROJECT_ROOT
    assert launcher.DEFAULT_DATASET_ROOT == expected
    assert preflight.DEFAULT_DATASET == expected
    assert launcher.DEFAULT_SOURCE_CHECKPOINT == PROJECT_ROOT / "7_storage/Delta_Weights"
    assert preflight.DEFAULT_CHECKPOINT == PROJECT_ROOT / "7_storage/Delta_Weights"
    assert launcher.VENDORED_LEROBOT_SRC == PROJECT_ROOT / "etc/third_party/lerobot/src"


def test_preflight_rejects_noncanonical_dataset_before_training(tmp_path: Path) -> None:
    preflight = _load(
        "test_training_preflight_rejection",
        "3_training/scripts/validate_incremental_training_preflight.py",
    )
    wrong_dataset = tmp_path / "lerobot_dataset_yawfree"
    wrong_dataset.mkdir()
    with pytest.raises(ValueError, match="noncanonical dataset basename"):
        preflight.run_preflight(
            wrong_dataset,
            PROJECT_ROOT / "7_storage/Delta_Weights",
            tmp_path / "output",
        )


def test_training_output_safety_guard(tmp_path: Path) -> None:
    preflight = _load(
        "test_training_output_guard",
        "3_training/scripts/validate_incremental_training_preflight.py",
    )
    guard = preflight._assert_training_output_is_safe

    # 1. Output absent -> PASS
    guard(tmp_path / "absent_output")

    # 2. Output exists empty -> PASS
    empty_out = tmp_path / "empty_output"
    empty_out.mkdir()
    guard(empty_out)

    # 3. Output with empty checkpoints & empty final -> PASS
    scaffold_out = tmp_path / "scaffold_output"
    (scaffold_out / "checkpoints").mkdir(parents=True)
    (scaffold_out / "final").mkdir(parents=True)
    guard(scaffold_out)

    # 4. Only empty checkpoints -> PASS
    only_ckpts = tmp_path / "only_ckpts"
    (only_ckpts / "checkpoints").mkdir(parents=True)
    guard(only_ckpts)

    # 5. Only empty final -> PASS
    only_final = tmp_path / "only_final"
    (only_final / "final").mkdir(parents=True)
    guard(only_final)

    # 6. Numeric checkpoint directory inside checkpoints -> FAIL
    with_step = tmp_path / "with_step"
    (with_step / "checkpoints" / "010000").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="non-empty scaffolding"):
        guard(with_step)

    # 7. model.safetensors inside checkpoints -> FAIL
    with_weights = tmp_path / "with_weights"
    (with_weights / "checkpoints").mkdir(parents=True)
    (with_weights / "checkpoints" / "model.safetensors").write_bytes(b"dummy")
    with pytest.raises(FileExistsError, match="non-empty scaffolding"):
        guard(with_weights)

    # 8. final/pretrained_model exists -> FAIL
    with_final = tmp_path / "with_final"
    (with_final / "final" / "pretrained_model").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="non-empty scaffolding"):
        guard(with_final)

    # 9. Unexpected regular file inside output -> FAIL
    with_file = tmp_path / "with_file"
    with_file.mkdir()
    (with_file / "some_file.txt").write_bytes(b"data")
    with pytest.raises(FileExistsError, match="unexpected entry"):
        guard(with_file)

    # 10. Unexpected directory inside output -> FAIL
    with_dir = tmp_path / "with_dir"
    (with_dir / "unknown_dir").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="unexpected entry"):
        guard(with_dir)

    # 11. Scaffolding is a symlink -> FAIL
    with_symlink = tmp_path / "with_symlink"
    with_symlink.mkdir()
    target = tmp_path / "dummy_target"
    target.mkdir()
    (with_symlink / "checkpoints").symlink_to(target)
    with pytest.raises(FileExistsError, match="invalid scaffolding"):
        guard(with_symlink)

    # 12. Output itself is a file -> FAIL
    file_as_out = tmp_path / "file_as_out"
    file_as_out.write_bytes(b"not a dir")
    with pytest.raises(FileExistsError, match="not a valid directory"):
        guard(file_as_out)
