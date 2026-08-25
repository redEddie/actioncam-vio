#!/usr/bin/env bash
# Run only after run_additional_orbslam_75_95.sh completed for every episode.
# This does NOT rerun ORB-SLAM or rebuild the map.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ADD_DIR="/home/kimminje/add episode"
MAP_TX="$ROOT/Episode_result/map_result/world/tx_slam_tag.json"
CALIB="$ROOT/cameras/hero13black/calibration/intrinsics.json"

[[ -f "$MAP_TX" ]] || { echo "ERROR: existing map transform missing: $MAP_TX" >&2; exit 1; }
[[ -f "$CALIB" ]] || { echo "ERROR: calibration missing: $CALIB" >&2; exit 1; }

for ep in $(seq 57 77); do
    n=$((ep + 18))
    video="$ADD_DIR/GX0100${n}.MP4"
    out="$ROOT/Episode_result/episode_${ep}"
    trajectory="$out/slam/camera_trajectory.csv"

    [[ -f "$video" ]] || { echo "ERROR: missing video: $video" >&2; exit 1; }
    [[ -f "$trajectory" ]] || {
        echo "ERROR: SLAM is not complete for episode_${ep}: $trajectory" >&2
        exit 1
    }

    echo "===== episode_${ep}: world align -> robot-base TCP ====="
    python -m gopro_vio.world_align --apply "$MAP_TX" "$trajectory" -o "$out/world"
    python -m gopro_vio.tcp_robot_transform "$out/world/trajectory_world.csv" -o "$out/world"

    python -m gopro_vio.aruco_detect "$video" --calib "$CALIB" \
        -o "$out/tags.pkl" --step 1 --ids 0 1
    python -m gopro_vio.gripper_width "$out/tags.pkl" -o "$out/gripper"

    python -m gopro_vio.validate_umi \
        --trajectory "$trajectory" \
        --world "$out/world/trajectory_world.csv" \
        --tx "$MAP_TX" \
        --kind demo \
        --slam-log "$out/slam/slam_log.txt" \
        --min-tracked-ratio 0.85 \
        --min-contiguous-ratio 0.70 \
        --max-world-diagonal-m 1.0 \
        --max-step-m 0.15 \
        --max-p99-step-m 0.05 \
        -o "$out/validation.json"
done

echo "TCP conversion, gripper extraction, and validation complete for episodes 57 through 77."
