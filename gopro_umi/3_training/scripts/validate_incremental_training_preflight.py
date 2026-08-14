#!/usr/bin/env python3
"""Fail-closed preflight for smoke or full incremental SmolVLA training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ACTION_ORDER = ["dx", "dy", "dz", "droll", "dpitch", "dgripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = {
        "info": args.dataset / "meta" / "info.json",
        "contract": args.dataset / "meta" / "incremental_action_contract.json",
        "resampling": args.dataset / "meta" / "resampling_validation.json",
        "incremental": args.dataset / "meta" / "incremental_validation.json",
        "step9a_summary": args.dataset / "meta" / "step9a_audit_summary.json",
        "step9a_mapping": args.dataset / "meta" / "step9a_source_to_10hz_mapping.parquet",
        "checkpoint_config": args.checkpoint / "config.json",
        "checkpoint_weights": args.checkpoint / "model.safetensors",
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("preflight missing required files: " + ", ".join(missing))
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite training output: {args.output}")
    info = json.loads(required["info"].read_text())
    contract = json.loads(required["contract"].read_text())
    resampling = json.loads(required["resampling"].read_text())
    incremental = json.loads(required["incremental"].read_text())
    step9a = json.loads(required["step9a_summary"].read_text())
    checkpoint = json.loads(required["checkpoint_config"].read_text())
    failures = []
    action = info.get("features", {}).get("action", {})
    if info.get("fps") != 10:
        failures.append(f"dataset fps={info.get('fps')} not 10")
    if info.get("total_episodes") != 77 or info.get("total_frames") != 12320:
        failures.append(
            f"dataset inventory episodes={info.get('total_episodes')} frames={info.get('total_frames')}"
        )
    if action.get("shape") != [6] or action.get("names", {}).get("axes") != ACTION_ORDER:
        failures.append(f"dataset action schema invalid: {action}")
    expected_contract = {
        "action_dim": 6,
        "action_order": ACTION_ORDER,
        "action_semantics": "STEP-TO-STEP INCREMENTAL",
        "action_rate_hz": 10,
        "action_dt_s": 0.1,
        "chunk_size": 15,
        "n_action_steps": 15,
        "horizon_s": 1.5,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            failures.append(f"contract {key}={contract.get(key)!r}, expected {expected!r}")
    if contract.get("status") != "VERIFIED":
        failures.append(f"dataset contract status={contract.get('status')}")
    recovery = contract.get("recovery", {})
    if recovery.get("status") not in {"VERIFIED", "DEFERRED"}:
        failures.append(f"recovery status={recovery.get('status')}")
    if recovery.get("status") == "DEFERRED" and recovery.get("baseline_training_blocker") is not False:
        failures.append("deferred recovery is not explicitly marked non-blocking for baseline")
    if contract.get("step9a_audit", {}).get("status") != "PASS":
        failures.append(f"contract STEP 9A status={contract.get('step9a_audit')}")
    structural = step9a.get("structural", {})
    if (
        step9a.get("status") != "PASS"
        or step9a.get("ten_hz_suitability_verdict") != "PASS"
        or structural.get("total_found") != 77
        or structural.get("total_valid") != 77
        or not structural.get("pass")
    ):
        failures.append("copied STEP 9A structural/suitability audit is not PASS for 77/77")
    if resampling.get("status") != "VERIFIED":
        failures.append(f"10 Hz resampling status={resampling.get('status')}")
    if incremental.get("status") != "VERIFIED" or incremental.get("random_samples", 0) < 100:
        failures.append(f"incremental unit validation invalid: {incremental}")
    if checkpoint.get("output_features", {}).get("action", {}).get("shape") != [6]:
        failures.append("source checkpoint action output is not 6D")
    if checkpoint.get("normalization_mapping", {}).get("ACTION") != "MEAN_STD":
        failures.append("source checkpoint action normalization is not MEAN_STD")
    if failures:
        raise RuntimeError("TRAINING BLOCKED:\n- " + "\n- ".join(failures))
    print("VERIFIED: baseline dataset/training preflight passed; recovery DEFERRED is non-blocking; output is [B,15,6]")


if __name__ == "__main__":
    main()
