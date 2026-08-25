# Dataset and checkpoint provenance

## Final local runtime bundle

`7_storage/Delta_Weights/` remains immutable. Baseline and final hashes are:

| File | SHA256 |
|---|---|
| `config.json` | `e000bb9c17c71927403a236a07393b3af463209a6845ad9b01f749939c3df784` |
| `model.safetensors` | `10620e8429f4e4df3922730076ed1bfbd26b9b6b45f505beb5ad5ccc264c5a66` |
| `policy_postprocessor.json` | `2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b` |
| `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `policy_preprocessor.json` | `a46e21831a76107905845bc5477c37a53be487c10cd2e67edcc70b300e8953ca` |
| `policy_preprocessor_step_5_normalizer_processor.safetensors` | `bf5209c1901c8a25307cf1b008462663175ec7011614fd9d72f3a3b830729bfc` |
| `train_config.json` | `06a469c7669691b7bfae61e35c9e23335e8131731013347fd4ea17cc862fbda4` |

## Final remote identity

- Dataset/run identity: `smolvla_10hz_chunk15_incremental_baseline_v1`.
- Checkpoint step: 20000.
- Server path: `/home/kimminje/gopro_umi/3_training/smolvla_10hz_chunk15_incremental_baseline_v1/checkpoints/020000/pretrained_model`.

## Known provenance gaps

The exact historical converter that produced the 10 Hz, 15-step, 6D,
step-to-step-incremental LeRobot dataset has not been recovered. The final
training launcher source **was recovered from the training server on
2026-08-14**. No separate final YAML/JSON training config was present in the
recovered source set; the exact final invocation settings are embedded in the
Python and shell launchers. Legacy 60 Hz/absolute converters and launchers are
archived and are not relabeled as canonical.

## Recovered final training launcher

Server source directory: `/home/kimminje/gopro_umi/3_training/`.
Canonical local provenance directory: `3_training/scripts/`.

| Local file | Server-original SHA256 | Role |
|---|---|---|
| `train_delta.py` | `6464699278a2b7577c64e4fe0e312e34f7a9f4e0eaac073d6144493b05aefbba` | Primary final Python launcher |
| `train_delta.sh` | `e614a960028758c95bfb7c3cf78a931358a641971ef4f6083cf2ae46ca2b72c2` | Original shell launcher/provenance |
| `validate_incremental_training_preflight.py` | `b8b2d4495d69a0abc57ba33b4d7a1be719841fba98ed39df0be559a946aa2f67` | Dataset/checkpoint fail-closed gate |
| `incremental_action_contract.py` | `82fa971142aee04086fae1527c05a9abcc7a5242d1bf9bab0f56686f9c547772` | 6D sequential incremental contract |
| `inference_delta_snippet.py` | `ba5d9b584f25178287803279ba280c0d7be31aada4c1adea2a6082d41d52d5e9` | Contract-compatible inference example |
| `tests/test_incremental_action_contract.py` | `2cfe7b104c1b0c30a8bcee8baa46e0e22572448099e35c9e335208f720217d30` | Recovered contract test |

The table records the recovered server-original hashes. The active launcher
copies were subsequently path-repaired to derive `PROJECT_ROOT` from the
script location, use `7_storage`, and support `--preflight-only`; the archived
provenance remains unchanged. The final launcher contract includes 20,000 steps, batch
size 32, GPU index 1, chunk size and `n_action_steps` 15, two empty cameras,
the `top`→`camera1` rename, image transforms, and full vision/expert training.
This retrieval recovers Item #2 launcher provenance; it does not imply a
training run was executed locally.

Recovered documents with conflicting historical chunk-50/chunk-60 semantics
are preserved as provenance at `etc/archive/legacy_training/documentation/` and
must not be used as the current training contract.

## Reconstructed canonical converter

- Historical final converter: **NOT RECOVERED**.
- Current reconstructed canonical implementation (2026-08-13):
  `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py`.
- Provenance classification: `RECONSTRUCTED_CANONICAL_IMPLEMENTATION`.
- Source contract: yaw-free
  `7_storage/zarr/replay_buffer_yawfree.zarr`, 10 Hz deterministic
  nearest-timestamp sampling from the audited `60000/1001 Hz` source clock,
  row-wise state/action shape `[6]`, step-to-step incremental action, no
  synthetic terminal row, and no cross-episode delta.
- Local source validation covered all 77 episodes and 74,109 source frames;
  it produced 12,320 real 10 Hz transition rows with zero duplicate selections, zero
  Roll/Pitch wrap crossings, and zero cross-episode actions.

This source is contract-compatible with the frozen bundle, but historical
equivalence is not claimed.  That requires a future comparison with the
server dataset
`PROJECT_ROOT/7_storage/lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1`.
The final training launcher is recovered as documented above. A separate
standalone final config file was not found; launcher-embedded configuration is
the recovered source of truth.

### STEP 3A.1 historical LeRobot validation (2026-08-13, superseded terminal policy)

Using only
`/home/kimminje/miniconda3/envs/lerobot_env2/bin/python` and the pinned
project source at `etc/third_party/lerobot` (HEAD
`1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`):

- Real LeRobot 0.6.1 disk serialization: **VERIFIED**.
- Real LeRobot reopen/readback: **VERIFIED**.
- Serialized 224×224 RGB, 6D state, and 6D action roundtrip: **VERIFIED**.
- Training-style `delta_timestamps` indexing to a `[15,6]` sequential future
  action window: **VERIFIED**.
- Three-episode boundary isolation and terminal zero actions: **VERIFIED**.
- Full 77-episode source validate-only audit: **VERIFIED**, zero contract
  violations.

These results complete validation of the current reconstructed canonical
converter. They do not recover the historical final converter and do not
prove exact equivalence with the historical server dataset.

### PIPELINE-LINK-01 canonical dataset validation (2026-08-14)

- Actual canonical dataset: **77 episodes / 12,320 real transitions / 10 Hz**.
- State/action: **6D/6D**, yaw absent, finite: **VERIFIED**.
- Full source row comparison and source→10 Hz mapping: **VERIFIED**.
- Deterministic 15-action chunk checks: **100 passed / 0 failed**.
- Synthetic terminal actions: **0**.
- Training `--preflight-only`: **PASS**, training steps executed: **0**.

## Dependency pins

- LeRobot: `1bb9933215dcb7ffeeae6d3746cda3f73f5a59e2`.
- actioncam-vio: `b25a431dcc631f99bc5a6055ad77aa8a9c97e317`.
- Project-local SOFollower calibration SHA256:
  `59749bc7139446aa57eab8281ef324696e658fea1fc24fbe488a4718d129aecc`.
