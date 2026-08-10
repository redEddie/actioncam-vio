#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ADD_DIR="/home/kimminje/add episode"
FFMPEG_BIN="/home/kimminje/miniconda3/bin/ffmpeg"
MAP_ATLAS="$ROOT/Episode_result/map_result/slam/map_atlas.osa"
CALIB="$ROOT/cameras/hero13black/calibration/intrinsics.json"
EXTR="$ROOT/cameras/hero13black/calibration/imu_extrinsics.json"
MASK="$ROOT/cameras/hero13black/calibration/gripper_mask.png"

[[ -f "$MAP_ATLAS" ]] || { echo "ERROR: existing map atlas missing: $MAP_ATLAS" >&2; exit 1; }
[[ -x "$FFMPEG_BIN" ]] || { echo "ERROR: ffmpeg missing: $FFMPEG_BIN" >&2; exit 1; }
[[ -f "$CALIB" ]] || { echo "ERROR: calibration missing: $CALIB" >&2; exit 1; }
[[ -f "$EXTR" ]] || { echo "ERROR: IMU extrinsics missing: $EXTR" >&2; exit 1; }
[[ -f "$MASK" ]] || { echo "ERROR: mask missing: $MASK" >&2; exit 1; }
# gopro_vio.slam invokes the literal command `ffmpeg`.  The active
# lerobot/venv combination can omit miniconda's base bin directory, so make
# the known installed binary visible without changing the user's shell setup.
export PATH="$(dirname "$FFMPEG_BIN"):$PATH"

for n in $(seq 75 95); do
    ep=$((n - 18))
    video="$ADD_DIR/GX0100${n}.MP4"
    out="$ROOT/Episode_result/episode_${ep}"

    [[ -f "$video" ]] || { echo "ERROR: missing video: $video" >&2; exit 1; }
    if [[ -f "$out/slam/camera_trajectory.csv" ]]; then
        echo "SKIP: episode_${ep} already has a camera trajectory"
        continue
    fi

    echo "===== GX0100${n}.MP4 -> episode_${ep} ====="
    mkdir -p "$out"
    python -m gopro_vio.extract "$video" -o "$out"
    python -m gopro_vio.slam "$video" \
        --imu "$out/imu.csv" \
        -o "$out/slam" \
        --calib "$CALIB" \
        --extr "$EXTR" \
        --mask "$MASK" \
        --load-map "$MAP_ATLAS" \
        --init_tag_size 0.10 \
        --width 960 \
        --features 2500 \
        --fps-div 2
    [[ -f "$out/slam/camera_trajectory.csv" ]] || {
        echo "ERROR: trajectory missing for episode_${ep}" >&2
        exit 1
    }
    echo "DONE: episode_${ep}"
done

echo "ORB-SLAM complete for GX010075 through GX010095."
