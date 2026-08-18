# STEP 3 execution manifest

The detailed pre-move hashes are in `MIGRATION_BASELINE_STEP3.md`.

- Canonical extraction: `deploy_canonical.py` and validated helpers became
  inference, trajectory, control, telemetry, and thin `run_live.py` modules.
- Trusted moves: IK v7 and primary URDF/assets moved under `4_deploy/ik/`;
  overlap and timestamp sampler moved under `4_deploy/trajectory/` with their
  baseline hashes unchanged before interface-only integration.
- Vendor: LeRobot moved to `third_party/lerobot`; its six-line local port diff
  is preserved in `archive/patch_history/vendor/`, and the vendor worktree is
  clean at its original HEAD.
- Legacy runners moved to `archive/legacy_deploy/runners/`; the near-complete
  orchestration source moved to `archive/patch_history/orchestration/` only
  after replacement unit and motorless tests passed.
- Experiments, patch/debug source, old checkpoints, generated outputs,
  evaluations, benchmarks, and manual tools were separated by their explicit
  final taxonomy.
- The actioncam tracked-data split was rolled back due to the Git/source
  conflict documented in the baseline. No alternative architecture was
  improvised.
- Delete candidates were not deleted. Zero-byte generated outputs remain in
  their original locations.
