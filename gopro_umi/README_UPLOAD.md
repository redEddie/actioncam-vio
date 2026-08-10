# gopro_umi project snapshot

This directory contains the code, configuration, documentation, tests, robot
assets, and lightweight reproducibility artifacts from the surrounding
`gopro_umi` workspace.

## Included

- Data-pipeline helper scripts outside the `actioncam-vio` repository.
- Dataset conversion scripts and LeRobot metadata for the regular and
  yaw-free datasets.
- Regular and yaw-free training scripts, configs, and lightweight checkpoint
  metadata.
- Deployment, evaluation, calibration, teleoperation, debugging, and test
  scripts.
- SO-100/SO-101 URDF and mesh assets.
- The locally modified LeRobot worktree, including the SO follower change.
- UMI source trees and the top-level `TEST` utilities.
- Yaw-free server handoff candidates without full model weights.

The five selected raw demonstrations and their complete processed VIO results
are stored at the repository root under `Episode/` and `Episode_result/`. See
`HERO13_SAMPLE_EPISODES.md` for the exact mapping.

## Intentionally excluded

- All other raw episode videos and generated video outputs.
- Full LeRobot datasets and replay-buffer Zarr arrays.
- Model and optimizer weight files larger than GitHub's 100 MiB limit.
- Hugging Face/SmolVLA caches, virtual environments, Python bytecode, build
  logs, nested Git object databases, and large temporary output archives.
- GoPro firmware blobs and installers.

These exclusions keep the branch cloneable while retaining the code and small
metadata needed to understand and reproduce the project.

## Vendored source revisions

- `umi/`: `redEddie/umi-realsense` at
  `b12fc81fb9ad60f4ec02c5d515dcea434f9ebf27`
- `2_dataset/umi_official/`: `real-stanford/universal_manipulation_interface`
  at `d095ba9590df789df5189eea5ee7e431689038a6`
- `4_deploy/Teleop/lerobot/`: `huggingface/lerobot` at
  `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`, plus the local
  `so_follower.py` modification
