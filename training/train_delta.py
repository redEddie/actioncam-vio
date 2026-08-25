#!/usr/bin/env python3
"""Portable launcher for canonical 10 Hz / chunk-15 incremental training.

``--preflight-only`` resolves and validates every project-owned input, imports
LeRobot, then exits without checking CUDA or executing a training step.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "artifacts"
CANONICAL_DATASET_NAME = "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
DEFAULT_DATASET_ROOT = STORAGE_ROOT / "lerobot" / CANONICAL_DATASET_NAME
DEFAULT_SOURCE_CHECKPOINT = STORAGE_ROOT / "Delta_Weights"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "training" / "smolvla_10hz_chunk15_incremental_baseline_v1"
)
VENDORED_LEROBOT_SRC = PROJECT_ROOT / "etc" / "third_party" / "lerobot" / "src"
PREFLIGHT_SCRIPT = Path(__file__).resolve().parent / "validate_incremental_training_preflight.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmolVLA 10 Hz Chunk-15 Delta Action Training"
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=1, help="physical CUDA GPU index")
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--smoke", action="store_true", help="two training steps on episode 0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def resolve_lerobot() -> tuple[Path, bool]:
    """Import LeRobot normally, falling back to the project-owned vendored source."""

    used_vendored = False
    if importlib.util.find_spec("lerobot") is None:
        if not VENDORED_LEROBOT_SRC.is_dir():
            raise ModuleNotFoundError(
                "lerobot is not installed and vendored source is missing: "
                f"{VENDORED_LEROBOT_SRC}"
            )
        sys.path.insert(0, str(VENDORED_LEROBOT_SRC))
        used_vendored = True
    module = importlib.import_module("lerobot")
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError("cannot determine imported lerobot source")
    return Path(module_file).resolve(), used_vendored


def _load_preflight_module() -> Any:
    spec = importlib.util.spec_from_file_location("training_preflight", PREFLIGHT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preflight module: {PREFLIGHT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_preflight_validation(
    dataset_root: Path, checkpoint_path: Path, output_dir: Path
) -> dict[str, Any]:
    module = _load_preflight_module()
    return module.run_preflight(dataset_root, checkpoint_path, output_dir)


def validate_lerobot_dataset_handoff(dataset_root: Path) -> dict[str, Any]:
    """Open the local dataset with the actual 15-step training index contract."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {"action": [0.1 * step for step in range(1, 16)]}
    dataset = LeRobotDataset(
        repo_id=f"gopro_umi/{CANONICAL_DATASET_NAME}",
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )
    sample = dataset[0]
    state = sample["observation.state"]
    action = sample["action"]
    if (
        len(dataset) <= 0
        or dataset.num_episodes <= 0
        or dataset.fps != 10
        or tuple(state.shape) != (6,)
        or tuple(action.shape) != (15, 6)
        or not bool(state.isfinite().all())
        or not bool(action.isfinite().all())
    ):
        raise RuntimeError(
            "LeRobot dataset handoff does not provide finite state [6] / action [15,6]"
        )
    return {
        "frames": len(dataset),
        "episodes": dataset.num_episodes,
        "fps": dataset.fps,
        "state_shape": list(state.shape),
        "action_window_shape": list(action.shape),
    }


def check_cuda(gpu_index: int) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing impractical CPU fallback")
    if torch.cuda.device_count() <= gpu_index:
        raise RuntimeError(
            f"configured CUDA device {gpu_index} is unavailable "
            f"(device count: {torch.cuda.device_count()})"
        )
    print(f"CUDA GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")


def _training_environment(gpu_index: int, used_vendored: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if used_vendored:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(VENDORED_LEROBOT_SRC)
            if not existing
            else str(VENDORED_LEROBOT_SRC) + os.pathsep + existing
        )
    return env


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_size != 15:
        raise ValueError("canonical training requires --chunk-size=15")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")

    dataset_root = args.dataset_root.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    if args.smoke:
        steps = 2
        output_dir = (
            PROJECT_ROOT
            / "training"
            / f"{CANONICAL_DATASET_NAME}_smoke_step9d_b32"
        )
        episodes_arg = ["--dataset.episodes=[0]"]
        extra_train_args = ["--log_freq=1", "--save_freq=2"]
    else:
        steps = args.steps
        output_dir = args.output_dir.resolve()
        episodes_arg = []
        extra_train_args = []

    lerobot_source, used_vendored = resolve_lerobot()
    preflight = run_preflight_validation(dataset_root, source_checkpoint, output_dir)
    lerobot_dataset = validate_lerobot_dataset_handoff(dataset_root)
    resolved = {
        "project_root": str(PROJECT_ROOT),
        "storage_root": str(STORAGE_ROOT),
        "dataset": str(dataset_root),
        "source_checkpoint": str(source_checkpoint),
        "output": str(output_dir),
        "python": sys.executable,
        "lerobot_import_source": str(lerobot_source),
        "lerobot_dataset": lerobot_dataset,
        "preflight": preflight["status"],
        "training_steps_executed": 0,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0

    check_cuda(args.gpu)
    train_cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={source_checkpoint}",
        f"--policy.repo_id=gopro_umi/{CANONICAL_DATASET_NAME}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id=gopro_umi/{CANONICAL_DATASET_NAME}",
        f"--dataset.root={dataset_root}",
        f"--batch_size={args.batch_size}",
        f"--steps={steps}",
        f"--output_dir={output_dir}",
        '--rename_map={"observation.images.top":"observation.images.camera1"}',
        "--policy.empty_cameras=2",
        "--policy.freeze_vision_encoder=false",
        "--policy.train_expert_only=false",
        "--dataset.image_transforms.enable=true",
        "--policy.chunk_size=15",
        "--policy.n_action_steps=15",
        *episodes_arg,
        *extra_train_args,
    ]
    print("training command:")
    print(" ".join(train_cmd))
    result = subprocess.run(
        train_cmd,
        env=_training_environment(args.gpu, used_vendored),
        cwd=PROJECT_ROOT,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
