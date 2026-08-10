#!/usr/bin/env python3
"""Train a separate yaw-free 6D SmolVLA from the base checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def discover_project_root() -> Path:
    """Support both an activated script and an isolated candidate copy."""
    script_root = Path(__file__).resolve().parent.parent
    for root in (script_root, script_root.parent):
        if all((root / name).is_dir() for name in ("2_dataset", "3_training", "4_deploy")):
            return root
    raise RuntimeError("cannot locate project root containing 2_dataset/3_training/4_deploy")


PROJECT_ROOT = discover_project_root()
LEROBOT_PATH = PROJECT_ROOT / "4_deploy" / "Teleop" / "lerobot" / "src"
if not LEROBOT_PATH.is_dir():
    LEROBOT_PATH = Path("/home/kimminje/lerobot/src")
DATASET_ROOT = PROJECT_ROOT / "2_dataset" / "lerobot_dataset_yawfree"
OUTPUT_DIR = PROJECT_ROOT / "3_training" / "smolvla_yawfree_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_dataset() -> None:
    info_path = DATASET_ROOT / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing yaw-free LeRobot dataset: {info_path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    expected = {
        "observation.images.top": [224, 224, 3],
        "observation.state": [6],
        "action": [6],
    }
    bad = {key: features.get(key, {}).get("shape") for key, value in expected.items() if features.get(key, {}).get("shape") != value}
    axes = features.get("action", {}).get("names", {}).get("axes")
    if info.get("total_episodes") != 56 or info.get("total_frames") != 55_751 or bad or axes != ["x", "y", "z", "roll", "pitch", "gripper"]:
        raise ValueError(f"invalid 6D dataset: episodes={info.get('total_episodes')}, frames={info.get('total_frames')}, bad_shapes={bad}, axes={axes}")
    print("Validated yaw-free LeRobot dataset: 56 episodes, 55,751 frames, state/action=6D")


def main() -> int:
    args = parse_args()
    validate_dataset()
    if args.validate_only:
        return 0
    if OUTPUT_DIR.exists():
        if OUTPUT_DIR.is_dir() and not any(OUTPUT_DIR.iterdir()):
            OUTPUT_DIR.rmdir()
        else:
            raise FileExistsError(f"refusing to overwrite existing yaw-free run: {OUTPUT_DIR}")
    input_features = ('{"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},'
                      '"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},'
                      '"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]},'
                      '"observation.state":{"type":"STATE","shape":[6]}}')
    output_features = '{"action":{"type":"ACTION","shape":[6]}}'
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        "--policy.path=lerobot/smolvla_base",
        "--policy.repo_id=gopro_umi/smolvla_yawfree_finetuned",
        "--policy.push_to_hub=false",
        f"--policy.input_features={input_features}",
        f"--policy.output_features={output_features}",
        "--dataset.repo_id=gopro_umi/smolvla_yawfree_dataset",
        f"--dataset.root={DATASET_ROOT}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--output_dir={OUTPUT_DIR}",
        "--rename_map={\"observation.images.top\": \"observation.images.camera1\"}",
        "--policy.empty_cameras=0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LEROBOT_PATH) + ":" + env.get("PYTHONPATH", "")
    print("Running:\n" + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
