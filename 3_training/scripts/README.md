# Canonical training launchers

The active launchers preserve the recovered 10 Hz, chunk-15, 6D
step-to-step incremental training contract while resolving project-owned paths
from their own source location.

Canonical inputs:

- dataset:
  `PROJECT_ROOT/7_storage/lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1`
- source checkpoint: `PROJECT_ROOT/7_storage/Delta_Weights`
- vendored LeRobot fallback: `PROJECT_ROOT/etc/third_party/lerobot/src`

Run the fail-closed validation without CUDA or training:

```bash
PYTHON_BIN=/path/to/python bash 3_training/scripts/train_delta.sh --preflight-only
```

The shell launcher uses `PYTHON_BIN` when supplied and otherwise uses the
active shell's `python`. The Python launcher first validates the exact
canonical dataset name, all five source-derived metadata artifacts, the real
Parquet rows, timestamp spacing, finite 6D state/actions, the 15-step contract,
and the checkpoint schema. It refuses a 60 Hz, yaw-free-only, reconstructed-v1,
or incomplete dataset before checking CUDA or starting a training process.

The original recovered file hashes and historical remote paths remain recorded
in `etc/docs/CHECKPOINTS.md` and the archive; the active launchers are now
portable path-repaired copies. Importing these modules never starts training.
