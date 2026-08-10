# Why D405 gives lower SLAM-anchor accuracy than D435i

Record of an experiment in this project (2026-06). We built the full UMI pipeline
(map → world-anchor via ArUco → camera/TCP → gripper action) and compared the
**world-anchor precision** (`calibrate_slam_tag` translation-spread of the origin
tag id 13; lower = better) across cameras and conditions.

## Results

| Camera | baseline | IMU | gripper in view | origin tag | mode | **anchor spread (median)** |
|---|---|---|---|---|---|---|
| D435i | 50 mm | yes | **no** (handheld) | 160 mm | stereo | 0.38 cm |
| D405  | 18 mm | no  | yes | 160 mm @0.7 m | stereo | 7.1 cm |
| D405  | 18 mm | no  | yes | 160 mm @0.4 m | stereo | 11.9 cm |
| D405  | 18 mm | no  | yes | **100 mm** | stereo | 5.5 cm |
| **D435i** | 50 mm | yes | **yes** | **100 mm** | stereo | **0.94 cm** |

The last two rows are the **fair, apples-to-apples comparison** (both with the gripper
mounted, same 100 mm marker, same STEREO mode): **D435i 0.94 cm vs D405 5.5 cm (~6x)**.
Tracking was robust on all (single map, no track-loss) — the difference is **metric
pose accuracy**, not tracking stability.

What each variable contributed:
- **Camera (D405→D435i): the dominant factor** (5.5 → 0.94 cm, same conditions).
- **Gripper occlusion: minor** (D435i 0.38 gripper-free → 0.94 with gripper, +~0.5 cm).
  A smaller marker (160→100 mm) helped D405 (7→5.5 cm) by reducing occlusion, but did
  not close the camera gap.

## Root cause — it is the (short baseline → short usable stereo-depth range), not "baseline" alone

We run **STEREO** (ORB-SLAM3 triangulates from the IR pair) — **not RGB-D** (the depth
map is unused). The mechanism:

```
short baseline -> short range over which stereo depth is reliable.
ORB-SLAM3 treats a feature as having valid (metric) depth only within
   Stereo.ThDepth (=40) x baseline:
     D405 : 0.0182 m x 40 = 0.73 m
     D435i: 0.0502 m x 40 = 2.01 m
At a typical mapping distance (~0.5 m and beyond) many features on D405 fall near/
past 0.73 m -> treated as monocular (no metric depth) -> camera-pose scale is poorly
constrained -> drift/scale error -> the 5-12 cm anchor spread.
+ D405 has NO IMU -> no inertial constraint to damp that drift.
```

So "short baseline" is only the *starting point*; the operative cause is **the fraction
of scene features that lack reliable stereo depth** at the working distance. (Switching
to RGB-D would not save D405: its depth sensor is itself a close-range device, unreliable
past ~0.5 m for this kind of scene.)

## Takeaway / decision

- **Environment-SLAM 6DoF pose tracking needs a long-enough usable stereo-depth range
  (i.e. enough baseline) + IMU.** Use **D435i / D455** for the SLAM anchor.
- **D405's strength is close-range (7-50 cm) depth + RGB-stereo observation**, not
  wide-area metric SLAM.
- **Decision: D435i is the SLAM/anchor camera** (0.94 cm with the gripper; IMU still
  available to improve further). The pipeline is camera-agnostic (`--cam d405|d435i`);
  only accuracy differs.

## Sample footage
~6 s clips (2x downscaled) from the recordings:
- D405 (color): [`clips/d405_sample.mp4`](clips/d405_sample.mp4) — close-range, marker
  fills much of the frame, little far background.
- D435i (left IR): [`clips/d435i_sample.mp4`](clips/d435i_sample.mp4) — wider context
  with distant background structure.
