# Benchmark 03 — Model Action Contract Validation Report

## 1. Purpose
This benchmark strictly validates whether the LeRobot dataset metadata, training configuration, SmolVLA model checkpoint, pre/postprocessor pipelines, and physical delta semantics strictly conform to the canonical model contract (10 Hz, 15-step chunk, 6D yaw-free step-to-step incremental action).

---

## 2. Existing Canonical Sources Inspected
* **Model Checkpoint & Tokenizer Config**: [`7_storage/Delta_Weights/config.json`](file:///home/kimminje/Desktop/project/gopro_umi/7_storage/Delta_Weights/config.json)
* **Pre/Postprocessor Pipelines**: [`7_storage/Delta_Weights/policy_preprocessor.json`](file:///home/kimminje/Desktop/project/gopro_umi/7_storage/Delta_Weights/policy_preprocessor.json) & [`7_storage/Delta_Weights/policy_postprocessor.json`](file:///home/kimminje/Desktop/project/gopro_umi/7_storage/Delta_Weights/policy_postprocessor.json)
* **Normalization Parameters**: [`7_storage/Delta_Weights/policy_preprocessor_step_5_normalizer_processor.safetensors`](file:///home/kimminje/Desktop/project/gopro_umi/7_storage/Delta_Weights/policy_preprocessor_step_5_normalizer_processor.safetensors)
* **Dataset Conversion Contract**: [`2_dataset/conversion/convert_to_lerobot_10hz_incremental.py`](file:///home/kimminje/Desktop/project/gopro_umi/2_dataset/conversion/convert_to_lerobot_10hz_incremental.py)
* **Actual Canonical Dataset**: `7_storage/lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1` (77 episodes, 12,320 transitions, 10 Hz)
* **Training Contract & Sampler**: [`3_training/scripts/incremental_action_contract.py`](file:///home/kimminje/Desktop/project/gopro_umi/3_training/scripts/incremental_action_contract.py)
* **Deployment Specifications**: [`4_deploy/config/deployment.yaml`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/config/deployment.yaml) & [`4_deploy/inference/action_postprocess.py`](file:///home/kimminje/Desktop/project/gopro_umi/4_deploy/inference/action_postprocess.py)

---

## 3. Dataset Contract
* **Single-row Action Shape**: `[6]` (in LeRobot Parquet rows)
* **Observation State Shape**: `[6]` (`[X, Y, Z, Roll, Pitch, Gripper]`)
* **Action Axis Ordering**: `("dX", "dY", "dZ", "dRoll", "dPitch", "dGripper")`
* **Yaw-Free State**: **YES** (Yaw is excluded from state representation)
* **Yaw-Free Action**: **YES** (Yaw is excluded from action representation)
* **Sampling Rate & Interval**: 10.0 Hz (0.100 s)

---

## 4. Model / Checkpoint Contract
* **Model Type**: SmolVLA (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone)
* **Action Dimension**: 6
* **Chunk Size / Action Steps**: 15 / 15
* **Action Horizon**: 1.5 s
* **Model Wire Output Shape**: `[1, 15, 6]`
* **Local Postprocessed Chunk Shape**: `[15, 6]`

---

## 5. Action Semantics & Trajectory Reconstruction
* **Step-to-Step Incremental**: **PASS** ($A[t] = S[t+1] - S[t]$)
* **Reconstruction**: $S[k+1] = S[k] + A[k]$ verified with numerical error $< 1.10 \times 10^{-16}$.
* **Physical Units**:
  * $dX, dY, dZ$: meters
  * $d\text{Roll}, d\text{Pitch}$: radians
  * $d\text{Gripper}$: normalized delta ($S_{\text{gripper}} \in [0, 1]$)

---

## 6. Preprocessor / Postprocessor Contract
* **Normalization Mode**: `MEAN_STD` on 6D Action and 6D State
* **Unnormalizer Pipeline**: Loaded directly via `PolicyProcessorPipeline` (`UnnormalizerProcessorStep` + `DeviceProcessorStep`)
* **Unnormalizer Numerical Consistency**: Max absolute error compared to analytical $y = x \cdot \text{std} + \text{mean}$ is `0.00e+00`.
* **NaN / Inf Count**: 0 / 0

---

## 7. Contract Verification Matrix

| Contract Item | Expected | Observed | Status |
| :--- | :---: | :---: | :---: |
| **Dataset Action Shape** | `[6]` | `[6]` | **PASS** |
| **Actual Canonical Dataset** | 77 episodes / 12,320 transitions | 77 / 12,320 | **PASS** |
| **Observation State Shape** | `[6]` | `[6]` | **PASS** |
| **Action Order** | `dX,dY,dZ,dRoll,dPitch,dGripper` | `dX,dY,dZ,dRoll,dPitch,dGripper` | **PASS** |
| **Yaw-free State** | Excluded | Excluded | **PASS** |
| **Yaw-free Action** | Excluded | Excluded | **PASS** |
| **Model Action Dim** | 6 | 6 | **PASS** |
| **Chunk Size** | 15 | 15 | **PASS** |
| **Raw Wire Shape** | `[1, 15, 6]` | `[1, 15, 6]` | **PASS** |
| **Postprocessed Shape** | `[15, 6]` | `[15, 6]` | **PASS** |
| **Sampling Rate** | 10 Hz | 10 Hz | **PASS** |
| **Sampling Interval** | 0.100 s | 0.100 s | **PASS** |
| **Horizon** | 1.5 s | 1.5 s | **PASS** |
| **Physical Units** | `m, rad, norm` | `m, rad, norm` | **PASS** |
| **Incremental Semantics** | $S[k+1] = S[k] + A[k]$ | $S[k+1] = S[k] + A[k]$ | **PASS** |
| **Normalization** | `MEAN_STD` (6D) | `MEAN_STD` (6D) | **PASS** |
| **Postprocessing** | Verified | Verified | **PASS** |

---

## 8. Hardware, Remote & Immutability Audit
* `MOTOR ACCESSED`: **NO**
* `MOTOR READ/WRITE`: **NO**
* `CAMERA ACCESSED`: **NO**
* `SSH ACCESSED`: **NO**
* `REMOTE INFERENCE`: **NO**
* `EXISTING DATASET/WEIGHT FILES MODIFIED/DELETED`: **0**

---

## 9. Verdict
```text
MODEL_ACTION_CONTRACT = PASS
```
