# STEP 3 migration baseline

Captured at `2026-08-13T17:45:46+09:00` before any source move, extraction,
normalization, or archive operation. The user-created full compressed backup at
`/home/kimminje/Desktop/gopro_umi_backup_20260813_1717.tar.gz` is outside the
project and is not modified by this migration.

## Critical first-party sources

| Current path | Size | SHA256 | Type | Planned target/action |
|---|---:|---|---|---|
| `4_deploy/deploy_canonical.py` | 21,379 | `bac8ac17d5ab3978b2d437b494c7bb7a2fc1a465eb4b3c33b45493773a48f012` | Python source | extract to canonical runtime modules, then archive |
| `4_deploy/yawfree_physical_start_v2.json` | 1,118 | `24498de75e7359d6073f6ed05ac7bf251867d126f173af85171b4efab7c7e261` | JSON provenance | normalize to config, archive original |
| `4_deploy/yawfree_user_defined_joint_zero.json` | 1,506 | `dcaa3f308ee166320c686fbd9d446da2dd654e9c173ab85e7b7bd4f9ca47e674` | JSON provenance | normalize to config, archive original |
| `4_deploy/ik_solver_v7.py` | 7,511 | `250143c27d7fb0ce71f2e4ee7d567f3deec5092ec77ebf94616e377e9dac71d2` | Python source | move as-is to `4_deploy/ik/` |
| `4_deploy/arrival_state_reanchor.py` | 2,818 | `4beb693063fc94fcc6dad955d956dabdbf9fdb9abdb68750374c5365612df802` | Python source | split into incremental/reanchor modules |
| `4_deploy/chunk_overlap_ensemble.py` | 6,990 | `b178f258c7fe7ce06b4d2d447419b638c65b6609344d8bda7a80ba00fce5811a` | Python source | move to `trajectory/overlap.py` |
| `4_deploy/trajectory_sampler_30hz.py` | 1,748 | `635c5f11249f89df37b83916f210ac30f07ff95fdc6fcada2c11b41570d39d4d` | Python source | move to `trajectory/interpolate.py` |
| `4_deploy/deploy_smolvla_yawfree.py` | 24,299 | `52f8fc74d32c28d0a13cf8e3c298d671c015ea8741ec7322bdb88283cebcebcb` | Python source | extract mapping/gripper helpers, then archive |
| `4_deploy/test_smolvla_one_step_closedloop_hold.py` | 21,466 | `7ca8d7771d97a95dc8b727c6d89d562d64080eb11901ba1c7771639e208b403b` | Python experiment | extract BP+delta equations, then archive |
| `4_deploy/URDF/so_arm_with_gopro_final.urdf` | 35,804 | `68087e6926b0d41858ae2831c2bafe4d4ecb68f4773e820965283b9194979ec0` | URDF | move to `4_deploy/ik/urdf/` |
| `4_deploy/URDF/assets/` + `meshes/` | 34,317,513 | aggregate `5dccba235880c0ad82c9533d275963e9d86104821cea701209b46380414b1067` | 50 asset files | move to `4_deploy/ik/urdf/` |

The asset aggregate is SHA256 over the sorted per-file SHA256 manifest.

## External SOFollower calibration

- Source: `/home/kimminje/.cache/huggingface/lerobot/calibration/robots/so_follower/None.json`
- Size: 905 bytes
- SHA256: `59749bc7139446aa57eab8281ef324696e658fea1fc24fbe488a4718d129aecc`
- Action: copy once to `4_deploy/config/so_follower_calibration.json`; no hardware calibration is performed.

## Delta_Weights immutable bundle

Bundle size at baseline: 906,741,474 bytes.

| File | Size | SHA256 |
|---|---:|---|
| `config.json` | 2,479 | `e000bb9c17c71927403a236a07393b3af463209a6845ad9b01f749939c3df784` |
| `model.safetensors` | 906,712,520 | `10620e8429f4e4df3922730076ed1bfbd26b9b6b45f505beb5ad5ccc264c5a66` |
| `policy_postprocessor.json` | 660 | `2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b` |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | 6,544 | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `policy_preprocessor.json` | 1,985 | `a46e21831a76107905845bc5477c37a53be487c10cd2e67edcc70b300e8953ca` |
| `policy_preprocessor_step_5_normalizer_processor.safetensors` | 6,544 | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `train_config.json` | 6,646 | `06a469c7669691b7bfae61e35c9e23335e8131731013347fd4ea17cc862fbda4` |

Migration rule: keep this directory in place and byte-identical.

## Nested repositories

### actioncam-vio

- Git root: `1_data_pipeline/actioncam-vio`
- HEAD: `b25a431dcc631f99bc5a6055ad77aa8a9c97e317`
- Branch: `hero13`
- Status: clean
- Diff stat: empty
- Action: tracked source/Git metadata to `1_capture/actioncam-vio`; ignored data and delete-candidates remain separately preserved.

### LeRobot

- Git root: `4_deploy/Teleop/lerobot`
- HEAD: `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`
- Branch: `main`
- Status: one modified file, `src/lerobot/robots/so_follower/so_follower.py`
- Diff stat: 6 insertions
- Action: move repository to `third_party/lerobot`, archive the complete diff, then remove only the recorded hard-coded-port edit.

The complete LeRobot diff is preserved at
`archive/patch_history/vendor/lerobot_so_follower_local_port.patch`.

## Hardware/network boundary

Baseline and migration validation do not open `/dev/ttyACM0`, connect a motor
bus, access a camera, establish SSH, or execute a model. All such operations
remain behind explicit runtime methods and are tested only with fakes.

## Source conflict and batch rollback

Static Git inspection during the actioncam move showed that `Episode/`,
`Episode_result/`, and the nested `gopro_umi/` copy are tracked by the
`actioncam-vio` repository, contrary to the STEP 2 assumption that they could
be excluded while retaining a clean relocated worktree. The actioncam batch was
rolled back in full. Restored evidence:

- HEAD: `b25a431dcc631f99bc5a6055ad77aa8a9c97e317`
- status: clean (0 lines)
- `Episode/`: 113 files, 7,121,962,829 bytes
- `Episode_result/`: 1,852 files, 2,429,030,855 bytes
- sample MP4 SHA256: `3fb593f687324165c52e7f1f7e79d823f6084a402c0b1d4cfed0904900e58335`
- sample result SHA256: `d15b92ab890c3f0c28cd6e138477de5dd2e5b4d70e696a13da2e0110e5d427b5`

The unresolved split is recorded rather than hidden or implemented by a new
sparse-worktree/symlink architecture.

Other upstream snapshots moved with clean histories:

- `umi`: HEAD `b12fc81fb9ad60f4ec02c5d515dcea434f9ebf27`, branch `main`
- `umi_official`: HEAD `d095ba9590df789df5189eea5ee7e431689038a6`, branch `main`
