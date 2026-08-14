# Recovered final training sources

These files were copied byte-for-byte from the training server on
2026-08-14. They are recovered historical sources, not reconstructed
replacements.

## Canonical recovered launcher

- `train_delta.py`: primary Python launcher for the final 10 Hz, chunk-15,
  6D step-to-step incremental SmolVLA run.
- `validate_incremental_training_preflight.py`: fail-closed dataset and source
  checkpoint preflight invoked by the launcher.
- `train_delta.sh`: original shell launcher retained as companion provenance.

The recovered launchers contain the original server paths rooted at
`/home/kimminje/gopro_umi`. They are intentionally preserved without local
path rewrites so their SHA256 values continue to match the server originals.
No standalone final YAML/JSON training config was found: the final invocation
settings are embedded in the launcher arguments.

The shell launcher's smoke-only path references three auxiliary server tools
that were not part of this retrieval. The full Python launcher is the
canonical recovered entry point; do not claim the shell smoke path is locally
self-contained.

## Contract companion

- `incremental_action_contract.py`: the six-dimensional incremental action
  contract and reconstruction helpers.
- `inference_delta_snippet.py`: inference-side example using that contract.
- `tests/test_incremental_action_contract.py`: recovered contract unit test.

The two recovered pipeline guides described older absolute/chunk-50 and
delta/chunk-60 experiments. They are preserved under
`archive/legacy_training/documentation/` and are not current contracts.

Training is never started by importing these modules. Use the exact server
environment and paths before attempting a future training run.
