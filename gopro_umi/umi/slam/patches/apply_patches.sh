#!/usr/bin/env bash
# Reproduce our patched ORB-SLAM3 from a clean upstream clone.
#   - clones UZ-SLAMLab/ORB_SLAM3 next to the workspace (if missing)
#   - applies our patch (C++14 build fix for Ubuntu 22.04 + Settings/System
#     SIGSEGV fix for "Rectified" stereo such as the RealSense D435i)
#
# Usage:  bash apply_patches.sh [<path-to-ORB_SLAM3>]
# Default target: ~/rs_slam/ORB_SLAM3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB="${1:-$HOME/rs_slam/ORB_SLAM3}"
PATCH="$HERE/0001-orbslam3-d435i-ubuntu2204.patch"

if [ ! -d "$ORB" ]; then
  echo "==> ORB_SLAM3 not found at $ORB — cloning upstream"
  git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORB"
fi

echo "==> Checking patch against $ORB"
if git -C "$ORB" apply --check "$PATCH" 2>/dev/null; then
  git -C "$ORB" apply "$PATCH"
  echo "==> Patch applied."
elif git -C "$ORB" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "==> Patch already applied (skipping)."
else
  echo "ERROR: patch does not apply cleanly. Upstream may have changed."
  echo "       Inspect $PATCH manually."
  exit 1
fi
echo "Done. Now build with: bash ~/rs_slam/setup_2_build.sh"
