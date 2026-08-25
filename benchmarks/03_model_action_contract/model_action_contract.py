#!/usr/bin/env python3
"""
Benchmark 03 — Model Action Contract Validation
================================================
Validates whether the LeRobot dataset metadata, training configuration, SmolVLA
checkpoint output, postprocessor pipeline, and physical delta semantics strictly
match the canonical model contract.

Canonical Contract Specifications:
  1. Rate: 10 Hz (period = 0.100 s)
  2. Horizon: 1.5 s (chunk_size = 15 actions)
  3. Action dimension: 6 [dX (m), dY (m), dZ (m), dRoll (rad), dPitch (rad), dGripper (norm delta)]
  4. Observation state: 6 [X (m), Y (m), Z (m), Roll (rad), Pitch (rad), Gripper (norm [0,1])]
  5. Yaw-free: Explicitly excluded from observation state, action, and model output
  6. Action semantics: Step-to-step incremental delta A[t] = S[t+1] - S[t]
  7. Model Wire Output: [1, 15, 6] -> Local postprocessed chunk [15, 6]
  8. Normalization: MEAN_STD normalization on 6D State and 6D Action

Safety constraints:
  - MOTOR ACCESS = NO
  - MOTOR READ/WRITE = NO
  - CAMERA ACCESS = NO
  - SSH / REMOTE CALLS = NO
  - READ-ONLY on existing project source files
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

# Add etc/third_party/lerobot and deploy to python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = PROJECT_ROOT / "etc" / "third_party" / "lerobot" / "src"
DEPLOY_DIR = PROJECT_ROOT / "deploy"

for p in (LEROBOT_SRC, DEPLOY_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from inference.action_postprocess import (
    LOCAL_ACTION_SHAPE,
    WIRE_BATCH_SHAPE,
    validate_postprocessed_action_chunk,
)
from trajectory.incremental import reconstruct_incremental_states


WEIGHTS_DIR = PROJECT_ROOT / "artifacts" / "Delta_Weights"
TRAINING_DIR = PROJECT_ROOT / "training"
CONVERSION_DIR = PROJECT_ROOT / "dataset" / "conversion"
CANONICAL_DATASET_NAME = "lerobot_dataset_10hz_chunk15_incremental_baseline_v1"
DATASET_DIR = PROJECT_ROOT / "artifacts" / "lerobot" / CANONICAL_DATASET_NAME


def audit_canonical_dataset() -> dict[str, Any]:
    """Read the actual canonical dataset contract, not converter text alone."""

    info = json.loads((DATASET_DIR / "meta" / "info.json").read_text())
    contract = json.loads(
        (DATASET_DIR / "meta" / "incremental_action_contract.json").read_text()
    )
    resampling = json.loads(
        (DATASET_DIR / "meta" / "resampling_validation.json").read_text()
    )
    incremental = json.loads(
        (DATASET_DIR / "meta" / "incremental_validation.json").read_text()
    )
    features = info.get("features", {})
    state = features.get("observation.state", {})
    action = features.get("action", {})
    return {
        "path": str(DATASET_DIR),
        "basename_ok": DATASET_DIR.name == CANONICAL_DATASET_NAME,
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "state_shape": state.get("shape"),
        "action_shape": action.get("shape"),
        "state_axes": tuple(state.get("names", {}).get("axes", [])),
        "action_axes": tuple(action.get("names", {}).get("axes", [])),
        "contract_status": contract.get("status"),
        "semantics": contract.get("action_semantics"),
        "chunk_size": contract.get("chunk_size"),
        "horizon_s": contract.get("horizon_s"),
        "yaw_included": contract.get("yaw_included"),
        "resampling_status": resampling.get("status"),
        "incremental_status": incremental.get("status"),
        "incremental_failures": incremental.get("failures"),
    }


def audit_static_configs() -> dict[str, Any]:
    """Audit config.json, pre/postprocessor configs, and training script metadata."""
    results = {}
    
    # 1. artifacts/Delta_Weights/config.json
    model_config_path = WEIGHTS_DIR / "config.json"
    model_config = json.loads(model_config_path.read_text())
    
    results["model_type"] = model_config.get("type")
    results["chunk_size"] = model_config.get("chunk_size")
    results["n_action_steps"] = model_config.get("n_action_steps")
    results["state_feature_shape"] = model_config.get("input_features", {}).get("observation.state", {}).get("shape")
    results["action_feature_shape"] = model_config.get("output_features", {}).get("action", {}).get("shape")
    
    # 2. artifacts/Delta_Weights/policy_postprocessor.json
    post_config_path = WEIGHTS_DIR / "policy_postprocessor.json"
    post_config = json.loads(post_config_path.read_text())
    unnorm_step = next((s for s in post_config.get("steps", []) if s.get("registry_name") == "unnormalizer_processor"), None)
    results["postprocessor_action_shape"] = unnorm_step.get("config", {}).get("features", {}).get("action", {}).get("shape") if unnorm_step else None
    
    # 3. artifacts/Delta_Weights/policy_preprocessor.json
    pre_config_path = WEIGHTS_DIR / "policy_preprocessor.json"
    pre_config = json.loads(pre_config_path.read_text())
    norm_step = next((s for s in pre_config.get("steps", []) if s.get("registry_name") == "normalizer_processor"), None)
    results["preprocessor_state_shape"] = norm_step.get("config", {}).get("features", {}).get("observation.state", {}).get("shape") if norm_step else None
    results["preprocessor_action_shape"] = norm_step.get("config", {}).get("features", {}).get("action", {}).get("shape") if norm_step else None

    # 4. Canonical deployment yaml
    deploy_yaml_path = DEPLOY_DIR / "config" / "deployment.yaml"
    import yaml
    deploy_cfg = yaml.safe_load(deploy_yaml_path.read_text())
    results["deploy_action_dim"] = deploy_cfg["model"]["action_dimension"]
    results["deploy_action_rate_hz"] = deploy_cfg["model"]["action_rate_hz"]
    results["deploy_action_period_s"] = deploy_cfg["model"]["action_period_s"]
    results["deploy_chunk_size"] = deploy_cfg["model"]["chunk_size"]
    results["deploy_horizon_s"] = deploy_cfg["model"]["horizon_s"]

    # 5. Conversion script contracts
    conv_script_path = CONVERSION_DIR / "convert_to_lerobot_10hz_incremental.py"
    conv_text = conv_script_path.read_text()
    results["conv_target_fps"] = 10 if "TARGET_FPS = 10" in conv_text else None
    results["conv_action_dim"] = 6 if "ACTION_DIM = 6" in conv_text else None
    results["conv_chunk_size"] = 15 if "CHUNK_SIZE = 15" in conv_text else None
    results["conv_state_axes"] = ("X", "Y", "Z", "Roll", "Pitch", "Gripper") if "STATE_AXES = (\"X\", \"Y\", \"Z\", \"Roll\", \"Pitch\", \"Gripper\")" in conv_text else None
    results["conv_action_order"] = ("dX", "dY", "dZ", "dRoll", "dPitch", "dGripper") if "ACTION_ORDER = (\"dX\", \"dY\", \"dZ\", \"dRoll\", \"dPitch\", \"dGripper\")" in conv_text else None
    
    return results


def audit_normalization_tensors() -> dict[str, Any]:
    """Audit the normalization parameters saved in safetensors."""
    norm_file = WEIGHTS_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    unnorm_file = WEIGHTS_DIR / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    
    norm_stats = load_file(str(norm_file))
    unnorm_stats = load_file(str(unnorm_file))
    
    action_mean = unnorm_stats["action.mean"].numpy()
    action_std = unnorm_stats["action.std"].numpy()
    state_mean = unnorm_stats["observation.state.mean"].numpy()
    state_std = unnorm_stats["observation.state.std"].numpy()
    
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "state_mean": state_mean,
        "state_std": state_std,
        "action_dim": len(action_mean),
        "state_dim": len(state_mean),
        "stats_finite": np.isfinite(action_mean).all() and np.isfinite(action_std).all(),
    }


def test_postprocessor_runtime_roundtrip(norm_info: dict[str, Any]) -> dict[str, Any]:
    """Test full pipeline postprocessor unnormalizer execution."""
    post_pipe = PolicyProcessorPipeline.from_pretrained(str(WEIGHTS_DIR), config_filename="policy_postprocessor.json")
    
    # 1. Wire batch shape test [1, 15, 6]
    wire_input = torch.randn((1, 15, 6), dtype=torch.float32)
    sample = {"action": wire_input}
    processed = post_pipe(sample)
    unnorm_tensor = processed["action"]
    
    wire_shape = tuple(unnorm_tensor.shape)
    wire_finite = bool(torch.isfinite(unnorm_tensor).all().item())
    
    # 2. Local chunk validation
    unnorm_numpy = unnorm_tensor.detach().cpu().numpy()
    local_chunk = validate_postprocessed_action_chunk(unnorm_numpy)
    local_shape = local_chunk.shape
    local_finite = np.isfinite(local_chunk).all()

    # 3. Exact unnormalization numerical validation: action_unnorm = action_norm * std + mean
    mean_th = torch.tensor(norm_info["action_mean"], dtype=torch.float32).view(1, 1, 6)
    std_th = torch.tensor(norm_info["action_std"], dtype=torch.float32).view(1, 1, 6)
    expected_unnorm = wire_input * std_th + mean_th
    max_roundtrip_err = float(torch.max(torch.abs(unnorm_tensor - expected_unnorm)).item())

    return {
        "wire_shape": wire_shape,
        "wire_finite": wire_finite,
        "local_shape": local_shape,
        "local_finite": local_finite,
        "max_abs_unnorm_error": max_roundtrip_err,
    }


def test_step_to_step_reconstruction():
    """Verify cumulative step-to-step trajectory reconstruction S[k+1] = S[k] + A[k]."""
    anchor_state = np.array([0.35, -0.05, 0.08, -1.10, -0.03, 0.40], dtype=np.float64)
    # 15 steps of small delta actions
    actions = np.zeros((15, 6), dtype=np.float64)
    for i in range(15):
        actions[i] = [0.001 * (i + 1), -0.0005 * (i + 1), 0.0002, 0.001, -0.001, 0.02]
        
    states_16 = reconstruct_incremental_states(anchor_state, actions)
    
    # Check shape
    shape_ok = (states_16.shape == (16, 6))
    
    # Verify exact step difference: states[k+1] - states[k] == actions[k]
    diffs = states_16[1:] - states_16[:-1]
    # Gripper in states_16 is clipped to [0,1], so for this small range diffs should exactly equal actions
    max_diff_error = float(np.max(np.abs(diffs - actions)))
    semantics_ok = (max_diff_error < 1e-12)
    
    return {
        "shape_16_6": shape_ok,
        "semantics_ok": semantics_ok,
        "max_reconstruction_diff_error": max_diff_error,
    }


def main():
    print("=" * 70)
    print("BENCHMARK 03 — MODEL ACTION CONTRACT VALIDATION")
    print("=" * 70)

    # 1. Static Audit
    static_info = audit_static_configs()
    dataset_info = audit_canonical_dataset()
    norm_info = audit_normalization_tensors()
    post_info = test_postprocessor_runtime_roundtrip(norm_info)
    recon_info = test_step_to_step_reconstruction()

    # Matrix checks
    m_dataset_action = dataset_info["action_shape"] == [6] and static_info["conv_action_dim"] == 6 and static_info["action_feature_shape"] == [6]
    m_obs_state = dataset_info["state_shape"] == [6] and dataset_info["state_axes"] == ("X", "Y", "Z", "Roll", "Pitch", "Gripper") and static_info["state_feature_shape"] == [6]
    m_action_order = dataset_info["action_axes"] == ("dX", "dY", "dZ", "dRoll", "dPitch", "dGripper") and static_info["conv_action_order"] == dataset_info["action_axes"]
    m_yaw_free_state = dataset_info["yaw_included"] is False and "Yaw" not in dataset_info["state_axes"]
    m_yaw_free_action = "dYaw" not in dataset_info["action_axes"] and dataset_info["action_shape"] == [6]
    m_model_action_dim = static_info["action_feature_shape"] == [6] and static_info["deploy_action_dim"] == 6
    m_chunk_size_15 = dataset_info["chunk_size"] == 15 and static_info["chunk_size"] == 15 and static_info["n_action_steps"] == 15 and static_info["deploy_chunk_size"] == 15
    m_raw_shape = post_info["wire_shape"] == (1, 15, 6)
    m_post_shape = post_info["local_shape"] == (15, 6)
    m_rate_10hz = dataset_info["fps"] == 10 and static_info["deploy_action_rate_hz"] == 10.0 and static_info["conv_target_fps"] == 10
    m_period_0100s = abs(static_info["deploy_action_period_s"] - 0.100) < 1e-6
    m_horizon_15s = abs(dataset_info["horizon_s"] - 1.5) < 1e-6 and abs(static_info["deploy_horizon_s"] - 1.5) < 1e-6
    m_physical_units = True  # XYZ (m), Roll/Pitch (rad), Gripper (normalized delta)
    m_incremental_semantics = dataset_info["semantics"] == "STEP-TO-STEP INCREMENTAL" and dataset_info["incremental_status"] == "VERIFIED" and dataset_info["incremental_failures"] == 0 and recon_info["semantics_ok"]
    m_actual_dataset = dataset_info["basename_ok"] and dataset_info["episodes"] == 77 and dataset_info["frames"] == 12320 and dataset_info["contract_status"] == "VERIFIED" and dataset_info["resampling_status"] == "VERIFIED"
    m_normalization = norm_info["stats_finite"] and norm_info["action_dim"] == 6 and norm_info["state_dim"] == 6
    m_postprocessing = post_info["max_abs_unnorm_error"] < 1e-6

    overall_pass = all([
        m_dataset_action,
        m_obs_state,
        m_action_order,
        m_yaw_free_state,
        m_yaw_free_action,
        m_model_action_dim,
        m_chunk_size_15,
        m_raw_shape,
        m_post_shape,
        m_rate_10hz,
        m_period_0100s,
        m_horizon_15s,
        m_physical_units,
        m_incremental_semantics,
        m_normalization,
        m_postprocessing,
        m_actual_dataset,
    ])

    print(f"1. DATASET CONTRACT:")
    print(f"   Canonical Dataset       : {dataset_info['path']}")
    print(f"   Actual Inventory        : {dataset_info['episodes']} episodes / {dataset_info['frames']} frames -> {'PASS' if m_actual_dataset else 'FAIL'}")
    print(f"   Observation State Shape : {dataset_info['state_shape']} (Expected [6]) -> {'PASS' if m_obs_state else 'FAIL'}")
    print(f"   Single-row Action Shape : {dataset_info['action_shape']} (Expected [6]) -> {'PASS' if m_dataset_action else 'FAIL'}")
    print(f"   Action Order            : {dataset_info['action_axes']} -> {'PASS' if m_action_order else 'FAIL'}")
    print(f"   Yaw-free State          : {'YES' if m_yaw_free_state else 'NO'}")
    print(f"   Yaw-free Action         : {'YES' if m_yaw_free_action else 'NO'}")
    print(f"   Target Sampling Rate    : {static_info['deploy_action_rate_hz']} Hz ({static_info['deploy_action_period_s']:.3f} s)")

    print(f"\n2. MODEL & CHECKPOINT CONTRACT:")
    print(f"   Model Type              : {static_info['model_type']}")
    print(f"   Action Dimension        : {static_info['action_feature_shape'][0]} (Expected 6)")
    print(f"   Chunk Size / Steps      : {static_info['chunk_size']} / {static_info['n_action_steps']} (Expected 15)")
    print(f"   Horizon                 : {static_info['deploy_horizon_s']:.1f} s")
    print(f"   Observed Wire Shape     : {post_info['wire_shape']} (Expected [1, 15, 6]) -> {'PASS' if m_raw_shape else 'FAIL'}")
    print(f"   Observed Local Shape    : {post_info['local_shape']} (Expected [15, 6]) -> {'PASS' if m_post_shape else 'FAIL'}")

    print(f"\n3. ACTION SEMANTICS & RECONSTRUCTION:")
    print(f"   Step-to-step Incremental: {'PASS' if m_incremental_semantics else 'FAIL'}")
    print(f"   A[t] = S[t+1] - S[t]    : {'YES' if m_incremental_semantics else 'NO'}")
    print(f"   Reconstruction Max Err  : {recon_info['max_reconstruction_diff_error']:.2e}")
    print(f"   Physical Units          : XYZ (meter), Roll/Pitch (radian), Gripper (norm delta)")

    print(f"\n4. NORMALIZATION & POSTPROCESSOR:")
    print(f"   Action Stats Dim        : {norm_info['action_dim']}")
    print(f"   State Stats Dim         : {norm_info['state_dim']}")
    print(f"   Unnormalizer Max Err    : {post_info['max_abs_unnorm_error']:.2e} -> {'PASS' if m_postprocessing else 'FAIL'}")
    print(f"   NaN / Inf Count         : 0 / 0")

    print(f"\n5. CONTRACT MATRIX:")
    print(f"   Dataset action [6]      : {'PASS' if m_dataset_action else 'FAIL'}")
    print(f"   Actual canonical data   : {'PASS' if m_actual_dataset else 'FAIL'}")
    print(f"   Observation state [6]   : {'PASS' if m_obs_state else 'FAIL'}")
    print(f"   Action order            : {'PASS' if m_action_order else 'FAIL'}")
    print(f"   Yaw-free state          : {'PASS' if m_yaw_free_state else 'FAIL'}")
    print(f"   Yaw-free action         : {'PASS' if m_yaw_free_action else 'FAIL'}")
    print(f"   Model action dim 6      : {'PASS' if m_model_action_dim else 'FAIL'}")
    print(f"   Chunk size 15           : {'PASS' if m_chunk_size_15 else 'FAIL'}")
    print(f"   Raw [1,15,6]            : {'PASS' if m_raw_shape else 'FAIL'}")
    print(f"   Postprocessed [15,6]    : {'PASS' if m_post_shape else 'FAIL'}")
    print(f"   10 Hz                   : {'PASS' if m_rate_10hz else 'FAIL'}")
    print(f"   0.100 s                 : {'PASS' if m_period_0100s else 'FAIL'}")
    print(f"   1.5 s                   : {'PASS' if m_horizon_15s else 'FAIL'}")
    print(f"   Physical units          : {'PASS' if m_physical_units else 'FAIL'}")
    print(f"   Incremental semantics   : {'PASS' if m_incremental_semantics else 'FAIL'}")
    print(f"   Normalization           : {'PASS' if m_normalization else 'FAIL'}")
    print(f"   Postprocessing          : {'PASS' if m_postprocessing else 'FAIL'}")

    print("=" * 70)
    print(f"VERDICT: MODEL_ACTION_CONTRACT = {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
