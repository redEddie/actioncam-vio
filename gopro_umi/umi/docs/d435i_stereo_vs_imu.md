# D435i: STEREO vs IMU_STEREO for the SLAM anchor

Quick comparison on one D435i recording (origin marker id 13 visible, with the
gripper mounted, enough motion to initialize the IMU). We measure the
**world-anchor precision** (`calibrate_slam_tag` translation-spread of the origin
tag; lower = better).

## Result

| mode | trajectory source | anchor spread (median) |
|---|---|---|
| **STEREO** | live `_frames.txt` | **0.43 cm** |
| IMU_STEREO | post-opt `_full.txt` | 0.63 cm |

Both are well under 1 cm. **IMU did not improve anchor accuracy here** — STEREO was
slightly better and is simpler to run.

(Note on the trajectory source: in IMU_STEREO the map is reset/rescaled mid-sequence
during IMU init (VIBA), so the live `_frames.txt` capture mixes pre/post-VIBA frames
and must not be used directly — only the post-optimization `_full.txt` is consistent.
STEREO has no such reset, so its live `_frames.txt` is already clean.)

## Decision

- **Use STEREO for data collection.** Same-or-better anchor accuracy, simpler
  pipeline (live per-frame trajectory works directly, no VIBA-rescale handling).
- We are **not root-causing why IMU came out worse right now** — our IMU know-how is
  still thin (the cam-IMU calibration via OpenICC never converged, so we fall back to
  the factory extrinsic). We will revisit IMU when polishing the full pipeline,
  **especially if demonstrations involve fast / dynamic motions** (e.g. throwing a
  ball) where stereo-only tracking may struggle and inertial constraints should help.
