# Canonical runtime contract

This document contains only the frozen production contract.

## Policy and actions

- Model: SmolVLA.
- Remote raw batch: `[1,15,6]`; local postprocessed chunk: `[15,6]`.
- Action order: `dX,dY,dZ,dRoll,dPitch,dGripper`; yaw is excluded.
- High-level rate: 10 Hz; period: 0.100 s; chunk: 15; horizon: 1.5 s.
- Actions are step-to-step increments: `S[k+1] = S[k] + A[k]`.
- XYZ units are metres; Roll/Pitch are radians.
- Gripper state is clipped to `[0,1]` after every incremental step.
- Normalized gripper semantics are `G=0` closed and `G=1` open. At the single
  deployment motor-mapping boundary, `OPEN_RAW=600` and `CLOSED_RAW=3000`;
  inputs are clipped to `[0,1]` before interpolation.

## Frame and vision

- Robot frame: `+X forward`, `+Y left`, `+Z up`.
- Deployment yaw comes from fresh actual joint state through FK; it is never
  forced to world zero.
- Dataset/source imagery is 224×224. The model processor/transport image is
  256×256. One real logical camera is encoded; two logical cameras are empty.

## Dataset conversion

- Canonical storage root: `PROJECT_ROOT/7_storage`.
- Source Zarr: `7_storage/zarr/replay_buffer.zarr`; yaw-free Zarr:
  `7_storage/zarr/replay_buffer_yawfree.zarr`; final training dataset:
  `7_storage/lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1`.
- Canonical converter:
  `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py`.
- Provenance: `RECONSTRUCTED_CANONICAL_IMPLEMENTATION`; the historical final
  converter remains unrecovered.
- Each LeRobot row stores `observation.state: [6]` and `action: [6]` in
  `X,Y,Z,Roll,Pitch,Gripper` order. The action is the next sampled state minus
  the current sampled state within the same episode. The terminal state is
  only the target of the final transition; no synthetic zero-action row is
  stored.
- SmolVLA forms the `[15,6]` future window from row-wise actions. A `[15,6]`
  chunk is not duplicated into every dataset row.
- The stored dataset camera key is `observation.images.top` at RGB 224×224;
  the frozen processor renames it to `observation.images.camera1`, resizes for
  the model, and supplies the two missing logical-camera masks.
- The current canonical dataset has 77 episodes and 12,320 real transition
  rows. Its five validation artifacts under `meta/` are derived from a full
  comparison with the 77-episode yaw-free source.
- The yaw-free transform is active at
  `2_dataset/conversion/build_yawfree_zarr.py`, using the audited historical
  Euler convention. Current `replay_buffer.zarr` has 56 episodes while the
  preserved yaw-free artifact has 77, so recreating that exact 77-episode
  artifact from scratch remains blocked on the missing 77-episode source.

## Start and mapping

Joint order is `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll`.

- RAW start degrees: `[-7.1208791208791204,-79.47252747252747,
  81.84615384615384,46.417582417582416,1.3626373626373627]`.
- URDF start degrees: `[1.4945054945054945,-83.51648351648352,
  79.82417582417582,45.36263736263736,0.0]`.
- Encoder zero degrees: `[-8.615384615384615,4.043956043956044,
  2.021978021978022,1.054945054945055,1.3626373626373627]`.
- Signs: `[1,1,1,1,1]`.
- Mapping: `RAW = URDF * sign + encoder_zero`.

Runtime ordering is move to RAW start, settle, read fresh present position,
map RAW→URDF, compute actual FK/state, then construct the observation.

## Controller

- Low-level rate: 30 Hz.
- Motor PID: P=64, I=0, D=32.
- External gain: `K_ext=0.5`; correction clamp: ±2 degrees.
- Static support degrees: `[0.264,0,0,0,-0.352]`.
- Reference gravity compensation degrees: `[0,-0.615,-2.286,-0.703,0]`.
- Gravity ratios use the fresh actual URDF pose, reference torque at the
  physical-start URDF pose, and a fixed ±1.5 ratio clamp.

`gravity_torque()` is the generalized torque exerted by gravity
(`tau_g = -dU/dq`), not counter-gravity actuator torque. Reference gravity
values are signed direct position commands and oppose `tau_ref` joint by
joint; the torque ratio transports that signed compensation to other poses.

The pose-dependent support is `b_support(q_actual) = b_static + b_g(q_actual)`.
HOLD computes `q_cmd_hold = q_nom_hold + b_support(q_actual) +
clip(K_ext*(q_nom_hold-q_actual), ±2°)`. On the one true HOLD→TRAJECTORY
transition, `q_actual_0`, `q_nom_0`, `q_cmd_hold`, and `b_support_0` are saved.
Trajectory commands are `q_cmd_hold + Δq_nom +
(b_support(q_actual)-b_support_0) +
clip(K_ext*(Δq_nom-Δq_actual), ±2°)`. Full support, static bias, and HOLD
correction are not added again. Tick zero equals `q_cmd_hold` exactly, and a
new policy chunk does not reset the trajectory origin.

Gravity payload assumptions are camera mass 0.240 kg at 0.030 m backward and
0.055 m upward from the gripper motor center, plus one complete custom jaw
assembly of 0.280 kg. The jaw length scale is 2.0 and its CoM is one third of
custom length from the base. The stock 0.012 kg moving jaw is excluded when
the custom jaw is enabled. The historical observed 7-degree elbow sag is not
used anywhere in the numerical model or gain selection.

## Scheduler, IK, and artifacts

- Inference request stride: 0.9 s; requested overlap: 0.2 s.
- Sufficient-overlap new weights: `0, 1/3, 2/3, 1`; short overlap uses the
  preserved compressed linear blend.
- IK implementation: `ik/ik_solver_v7.py`, class
  `DLSInverseKinematicsV7`; position is metres, seed/output are radians, and
  gripper is never passed to arm IK.
- Gravity implementation: `control/gravity_compensation.py`. It uses the
  pinned inertial model
  `control/gravity_assets/so_arm101_stock_inertial_gravity_v1.urdf`; this is
  separate from and does not alter the IK URDF.
- Local immutable model bundle: `7_storage/Delta_Weights/`.
- Server checkpoint: `/home/kimminje/gopro_umi/3_training/smolvla_10hz_chunk15_incremental_baseline_v1/checkpoints/020000/pretrained_model`, step 20000.
- Remote endpoint: `kimminje@155.230.189.77:17970`; credentials are never
  stored in project source or configuration.
