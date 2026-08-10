#!/usr/bin/env bash
# Build the offline .db3 SLAM driver (links against the prebuilt ORB-SLAM3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_SLAM3_ROOT="${ORB_SLAM3_ROOT:-$HOME/rs_slam/ORB_SLAM3}"
PREFIX="${PANGOLIN_PREFIX:-$HOME/rs_slam/install}"   # local Pangolin install

if [ ! -f "$ORB_SLAM3_ROOT/lib/libORB_SLAM3.so" ]; then
  echo "ERROR: ORB-SLAM3 not built at $ORB_SLAM3_ROOT/lib/libORB_SLAM3.so"
  echo "       Run ~/rs_slam/setup_2_build.sh first."
  exit 1
fi

cmake -S "$HERE" -B "$HERE/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DORB_SLAM3_ROOT="$ORB_SLAM3_ROOT" \
  -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build "$HERE/build" -j"$(nproc)"
echo "Built: $HERE/build/stereo_inertial_db3"
