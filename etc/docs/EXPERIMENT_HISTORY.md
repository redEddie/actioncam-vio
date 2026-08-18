# Experiment history

Historical sources and outputs are retained under `etc/archive/`; none of the
values below override `CONTRACT.md`.

- Early policy paths interpreted actions as absolute targets, used yaw, and/or
  consumed one queued action at a time.
- Intermediate yaw-free work included 50-step and 60-step chunks, 60 Hz
  high-level assumptions, and anchor-relative delta experiments.
- Dataset work progressed from absolute next-state converters to the final
  10 Hz, 15-step, 6D step-to-step incremental run. Exact final converter and
  launch provenance remain missing rather than reconstructed from memory.
- Controller experiments explored larger external correction, clamp sweeps,
  P-gain changes, feedforward, tracking, and floor-contact-contaminated tests.
  The historical external gain 1.1 was superseded by accepted 0.5.
- The fixed full support bias `[0.264,0.615,2.286,0.703,0.352]` was retained as
  the exact reference-pose anchor, then split into static pan/roll support and
  URDF pose-dependent pitch gravity support. V1 uses a ±1.5 gravity-ratio
  clamp and does not fit gains or stiffness from the historical approximately
  7-degree elbow sag observation.
- The exact user-supplied `so101_gravity_compensation_v1.py` prototype was not
  present in the accessible workspace during STEP 3C, so no reconstructed file
  was mislabelled as that original archive provenance. When supplied, its
  reserved destination is
  `etc/archive/patch_history/controller/gravity/so101_gravity_compensation_v1.py`.
- Start-pose experiments covered elevated +30/+50 mm variants and multiple
  shoulder-pan RAW values. The v2 physical start is the only runtime start.
- Latency work progressed through request/response timing, fractional
  reanchor, old/new overlap, and true-pipeline benchmark scripts.
- Historical gripper endpoint experiments and comments conflicted. The
  canonical physical endpoints were frozen on 2026-08-13 as open RAW 600 and
  closed RAW 3000 while preserving `G=0` closed / `G=1` open semantics.

Archive taxonomy separates legacy deployment/training/dataset code, patch
history, controller/start experiments, old checkpoints, generated results,
upstream snapshots, and backups.
