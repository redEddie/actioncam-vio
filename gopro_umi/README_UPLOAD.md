# Canonical gopro_umi source snapshot

This directory is the cleaned, source-only snapshot of the canonical
`gopro_umi` workspace as of 2026-08-14.

## Included

- Dataset conversion and validation source under `2_dataset/`.
- Recovered final training launcher, preflight, contract, and tests under
  `3_training/scripts/`.
- Canonical deployment, configuration, IK/URDF assets, controller, inference,
  trajectory, telemetry, evaluation, and tests under `4_deploy/`.
- Canonical manual tools, project documentation, and `pipeline.md`.
- Lightweight model configuration and processor metadata in `Delta_Weights/`.
- LeRobot as a pinned Git submodule at commit
  `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`.

## Intentionally excluded

- `Episode/`, `Episode_result/`, raw videos, and generated SLAM outputs.
- Replay-buffer Zarr and serialized LeRobot datasets.
- `Delta_Weights/model.safetensors`, old checkpoints, optimizer state, and
  training runs.
- Hugging Face/SmolVLA caches, virtual environments, bytecode, and test caches.
- Generated reports/media/logs, backup archives, firmware snapshots, and old
  checkpoint families.
- Legacy runners, patch/debug history, controller/start-pose experiments, and
  other archive-only material that is not needed to run the canonical pipeline.
- Historical benchmark scripts that still reference superseded action shapes.
- The actioncam-vio repository itself, because it already owns the Git root;
  recursive duplication would create an invalid project-within-project tree.

The excluded model and dataset artifacts must be restored separately at the
paths documented in `docs/CHECKPOINTS.md`. Clone with submodules:

```bash
git clone --branch hero13 --recurse-submodules \
  https://github.com/redEddie/actioncam-vio.git
```

No training, inference, hardware access, or model modification is performed by
this source upload.
