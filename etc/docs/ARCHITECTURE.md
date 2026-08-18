# Canonical deployment architecture

`4_deploy/run_live.py` is the sole production entry point and contains only
lifecycle/orchestration. Importing it does not connect hardware, open a camera,
start a thread, create a log, connect SSH, or load the policy.

The runtime dependency chain is:

1. `config/runtime_config.py` loads and cross-validates deployment, physical
   start, joint mapping, calibration, and immutable model references.
2. `control/motor_io.py` is the sole canonical motor side-effect boundary.
3. `inference/camera.py` supplies a latest RGB 256×256 frame and monotonic
   timestamp; `inference/observation.py` pairs it with fresh RAW→URDF→FK state.
4. `inference/remote_smolvla.py` retains request IDs and transports immutable
   request snapshots through a bounded asynchronous client.
5. `inference/action_postprocess.py` validates `[1,15,6]`/`[15,6]` without
   renormalization.
6. `trajectory/incremental.py` reconstructs all 15 step-to-step increments;
   `trajectory/reanchor.py` uses explicit physical arrival state.
7. `trajectory/chunk_scheduler.py` owns request contexts and old/new chunk
   lifecycle; `trajectory/overlap.py` preserves blend mathematics.
8. `trajectory/interpolate.py` samples timestamped 10 Hz anchors at the 30 Hz
   control clock with explicit BEFORE_START/VALID/EXPIRED states.
9. `ik/ik_solver_v7.py` solves XYZ/Roll/Pitch with FK-derived actual yaw.
10. `control/gravity_compensation.py` evaluates cached URDF inertials and the
    fixed camera/custom-jaw payload using fresh actual URDF joint state.
    `control/bp_delta_controller.py` applies only the change from onset support
    during trajectory, preserving exact TICK0 continuity;
    `control/safety.py` blocks invalid/nonfinite/out-of-limit commands before
    `control/motor_io.py` can write.
11. `telemetry/runtime_logger.py` opens output only on explicit runtime entry.

Every response is matched by request ID, followed immediately by injected
fresh state acquisition, RAW→URDF, FK, and actual-arrival reanchor. Planned
trajectory state is not accepted as physical arrival state.

The gravity model is initialized once in `build_parts`; URDF parsing and
reference-torque calculation never occur at 30 Hz. The IK model continues to
use `ik/urdf/so_arm_with_gopro_final.urdf`. Gravity separately uses
`control/gravity_assets/so_arm101_stock_inertial_gravity_v1.urdf` because the
IK URDF's custom end-effector subtree does not contain the V1 stock
`gripper_frame_link`/`moving_jaw_so101_v1_link` contract. Production imports
only `control/gravity_compensation.py`, never archive provenance.

Control-tick telemetry includes actual/nominal/command joints, gravity and
reference torques, unclipped/clipped ratios, gravity/static/total/delta support,
external correction, and clamp flags. Logging remains explicit-lifecycle.

Modes are `static`, `shadow`, `preview`, and `execute`. STEP 3 ran only
`static`; live modes additionally require the explicit live-I/O gate and remain
STEP 4 validation work.

## Capture/data ownership

`1_capture/actioncam-vio/` is the complete upstream Git repository and remains
the physical owner of its tracked `Episode/`, `Episode_result/`, and nested
`gopro_umi/` subtrees. Canonical data views are relative, non-duplicating
references:

- `1_capture/recordings/gopro/Episode` → `1_capture/actioncam-vio/Episode`
- `2_dataset/source/actioncam_vio/episode_results` →
  `1_capture/actioncam-vio/Episode_result`

The nested `1_capture/actioncam-vio/gopro_umi/` directory is an upstream,
Git-tracked historical subtree. It is not the canonical project root and is not
a runtime import source. No symlink points back into the project from within
the upstream repository.
