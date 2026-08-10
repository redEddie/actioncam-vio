#!/usr/bin/env bash
# Run the offline ORB-SLAM3 driver on a .db3 recording, handling all paths.
#
#   slam/run_slam.sh mapping   <recording.db3> [--viewer]   # build + save map_d435i.osa
#   slam/run_slam.sh localize  <recording.db3> [--viewer]   # load map_d435i.osa + relocalize
#   slam/run_slam.sh odom      <recording.db3> [--viewer]   # plain stereo-inertial odometry
#
# Maps and trajectories are written under data/maps/. The viewer (Pangolin)
# defaults OFF; pass --viewer to show it on the desktop (DISPLAY :1 over SSH).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: run_slam.sh <mapping|localize|odom> <recording.db3> [--viewer]}"
BAG="${2:?path to .db3 recording}"
VIEWER="--no-viewer"; [ "${3:-}" = "--viewer" ] && VIEWER=""

BAG="$(readlink -f "$BAG")"
[ -f "$BAG" ] || { echo "recording not found: $BAG"; exit 1; }
ORB="$HOME/rs_slam/ORB_SLAM3"
VOCAB="$ORB/Vocabulary/ORBvoc.txt"
DRIVER="$REPO/slam/offline_driver/build/stereo_inertial_db3"
[ -x "$DRIVER" ] || { echo "driver not built: run slam/offline_driver/build.sh"; exit 1; }

# Camera: --cam d435i (default) or d405. D405 has no IMU -> always STEREO.
CAM="d435i"; for a in "$@"; do case "$a" in --cam=*) CAM="${a#--cam=}";; esac; done
# Default to non-inertial STEREO: robust, and required for the D405 (no IMU).
# IMU_STEREO is enabled with --imu (d435i/d455 only, after cam-IMU calib).
SENSOR="--stereo"; for a in "$@"; do [ "$a" = "--imu" ] && SENSOR=""; done
[ "$CAM" = "d405" ] && SENSOR="--stereo"   # D405 has no IMU

case "$MODE" in
  mapping)  CFG="$REPO/slam/configs/${CAM}_mapping.yaml";       EXTRA="$SENSOR" ;;
  localize) CFG="$REPO/slam/configs/${CAM}_localization.yaml";  EXTRA="--localize $SENSOR" ;;
  odom)     CFG="$REPO/slam/configs/${CAM}_stereo$([ "$CAM" = d435i ] && echo _inertial).yaml"; EXTRA="$SENSOR" ;;
  *) echo "mode must be mapping | localize | odom"; exit 1 ;;
esac
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }

mkdir -p "$REPO/data/maps"
NAME="$(basename "$BAG" .db3)"
TRAJ="$REPO/data/maps/${MODE}_${NAME}_traj.txt"

cd "$REPO/data/maps"     # atlas map_d435i.osa is saved/loaded relative to here
export LD_LIBRARY_PATH="$HOME/rs_slam/install/lib:${LD_LIBRARY_PATH:-}"
export DISPLAY="${DISPLAY:-:1}"

echo ">> mode=$MODE  bag=$NAME  viewer=${VIEWER:-on}"
"$DRIVER" "$VOCAB" "$CFG" "$BAG" "$TRAJ" $EXTRA $VIEWER
echo ">> done. trajectory: $TRAJ"
ls -la "$REPO"/data/maps/*.osa 2>/dev/null || true
