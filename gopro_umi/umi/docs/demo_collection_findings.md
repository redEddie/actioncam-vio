# Demo collection findings (D435i, SO-ARM small workspace)

Status after building the full map-once + relocalize-demos pipeline (workflow A)
and testing it on a SO-ARM (short-reach, small workspace) setup.

## What works
- **Relocalization patch** (see `slam/patches/0001-...`): localization mode now
  relocalizes into the prebuilt map instead of starting a fresh map. Self-localize
  went 1 -> 934 tracked frames; all demos now relocalize into a single map.
- **Full extraction pipeline** end-to-end: localize -> `T_world_map` (id13 anchor)
  -> `T_cam_tcp` -> per-frame TCP pose + gripper width -> episode CSV.
- **Cleaning** (`dataset/clean_episode.py`): a physical per-frame step cap removes
  transient SLAM glitches robustly (better than a fixed q99, which assumes a
  constant outlier rate). Timestamps fixed to nanoseconds in `extract_episode`.
- **Visualization**: `plot_episode[_interactive].py`, `plot_map.py`, and a Rerun
  logger `log_rerun.py` (camera frustums + point cloud + frame timeline).

## What does NOT work yet — the core problem
On 10 pick-and-place demos, **relocalization gets an initial fix but cannot sustain
tracking**: after the first relocalize the camera pose drifts/jumps (e.g. demoA01
excursions of ~30 cm) and most of each demo is lost. Only 2/10 (A01, A09) captured a
recognizable (but tiny) pick-and-place; the rest were truncated.

Root cause — **the map is too sparse to track against in this setup**:
- The map had only ~11 keyframes / ~1400 points, clustered.
- The ArUco landmark cubes (ids 0-4) placed beside the workspace were **never even
  detected** in the map clip -> they were barely in the camera's view.
- The real bottleneck is **field of view**: the D435i is ~87 deg horizontal, much
  narrower than UMI's fisheye GoPro (~150 deg+). In a small workspace the narrow FOV
  sees too little stable background to build a dense map or keep frame-to-frame
  tracking locked. UMI uses a fisheye precisely to capture wide background context.

## Directions (decision pending)
1. **Wider FOV camera / fisheye lens** (user's leaning, and what UMI does). A wide/
   fisheye view captures far more background features even in a confined workspace,
   which should fix both map density and sustained tracking. (Intel's fisheye option
   was the T265 tracking cam; or add a wide-angle/fisheye lens; or a separate fisheye
   for SLAM + D435i for depth/gripper.)
2. **Marker-based pose** (workflow B): keep a marker board in view during demos and
   get TCP pose directly from the marker, using SLAM only to bridge brief occlusions.
   More robust for a confined workspace where SLAM can't see enough to track.

Either way, the SLAM/relocalization machinery itself is now correct; the limitation
is sensing (FOV) for this small-workspace rig.
