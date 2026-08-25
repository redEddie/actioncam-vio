# STEP 3B Baseline — Actioncam-VIO and Gripper

Recorded: 2026-08-13 (Asia/Seoul)

This record was captured before STEP 3B relocation and endpoint changes. No hardware, camera, network, inference, or training operation was performed.

## Actioncam-VIO pre-move baseline

- Source: `1_data_pipeline/actioncam-vio/`
- Target: `1_capture/actioncam-vio/`
- Action: whole-repository same-filesystem rename
- Filesystem device: `66306`
- Source directory inode: `28837662`
- Total size: `24,545,629,390` bytes
- All filesystem entries: `22,484`
- Regular files: `20,178`
- Git-tracked paths: `2,142`
- Git HEAD: `b25a431dcc631f99bc5a6055ad77aa8a9c97e317`
- Git tree: `5a6b4dbd97ae40f77a93121922691c4f7ee12694`
- Branch: `hero13`
- Status: clean (`hero13...origin/hero13`)
- Working-tree diff: empty
- `git fsck --no-dangling`: clean

The tracked `Episode/`, `Episode_result/`, and nested `gopro_umi/` subtrees remain inside this repository and move with it.

## Pre-change source hashes

| Path | SHA256 |
|---|---|
| `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` | `5379104a8d426bacd39e0b67854cd48c3c771e4e13357985b660e032c2c995cb` |
| `2_dataset/validation/test_convert_to_lerobot_10hz_incremental.py` | `63f474e91ed2204cb0cb53953018837db90849143fe3ee1d2bcd6bd35cfa5b55` |
| `4_deploy/config/physical_start.json` | `98bb272f2919a298440381fbca4576719e5b5a7fb5624d965d4b16bfb8d8dde9` |
| `4_deploy/config/motor_mapping.json` | `becffde025d1f21c95bec8fa6927fdca2ae951054d175f6f9922d128ad114385` |
| `4_deploy/config/deployment.yaml` | `805c4b40c04660e3cedf65e8298787898d9b7988d3f281f112546f24ad1fd186` |
| `4_deploy/control/joint_mapping.py` | `6bad2637d511a421dea6d3c417dfdc71b6e2bd8000d1465bf30d3483f30472d1` |
| `4_deploy/control/bp_delta_controller.py` | `c164c4fea5d91915ae329f2424103958241c55660700b86753d39816c4b1b3e7` |
| `4_deploy/control/motor_io.py` | `d7d95768ff86b7f7a9020ae909b548e89241b7c392b0a3a0cd187e3ec70b4369` |
| `4_deploy/ik/ik_solver_v7.py` | `473e9a243aa3cde64952f314c81ba34a6695537f8280be3635ada5f3a95d9a84` |
| `archive/legacy_dataset/zarr/replay_buffer_yawfree.zarr/.zattrs` | `157c656e7193501c23205419bf44e3d7689e4f90aae09eb05599ce35ad2a81ed` |
| `archive/legacy_dataset/zarr/replay_buffer_yawfree.zarr/.zgroup` | `2383746e67b4bcc2762b3f100f06c3fa2d5f149ab5a8e5da5d33521464a01959` |

The source-Zarr file-list/size/mtime manifest contains 2,978 files totaling
8,965,709,145 bytes and has SHA256
`d6e0d3f29354980835b19919b2e9976951cafc1ad48c7ed7267007428c1c01d5`.

Pinned LeRobot source before STEP 3B: HEAD `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`, clean.

## Immutable model bundle hashes

| Path | SHA256 |
|---|---|
| `Delta_Weights/config.json` | `e000bb9c17c71927403a236a07393b3af463209a6845ad9b01f749939c3df784` |
| `Delta_Weights/model.safetensors` | `10620e8429f4e4df3922730076ed1bfbd26b9b6b45f505beb5ad5ccc264c5a66` |
| `Delta_Weights/policy_postprocessor.json` | `2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b` |
| `Delta_Weights/policy_postprocessor_step_0_unnormalizer_processor.safetensors` | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `Delta_Weights/policy_preprocessor.json` | `a46e21831a76107905845bc5477c37a53be487c10cd2e67edcc70b300e8953ca` |
| `Delta_Weights/policy_preprocessor_step_5_normalizer_processor.safetensors` | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `Delta_Weights/train_config.json` | `06a469c7669691b7bfae61e35c9e23335e8131731013347fd4ea17cc862fbda4` |

## Gripper pre-patch baseline

- Previous active config: `safe_raw_open = 1500`, `safe_raw_closed = 600`.
- Normalized semantics proven by both current deployment mapping and dataset calibration: `G=0` means closed and `G=1` means open.
- STEP 3B authoritative replacement: `open_raw = 600`, `closed_raw = 3000`.
- The normalized training state/action contract remains unchanged.
