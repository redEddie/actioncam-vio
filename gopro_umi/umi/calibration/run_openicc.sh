#!/usr/bin/env bash
# Run OpenICC camera + camera-IMU calibration on a converted RealSense dataset.
# Dataset is produced by calibration/db3_to_openicc.py and must contain:
#   <dataset>/cam/cam0/*.png         (intrinsics clip)
#   <dataset>/cam_imu/cam0/*.png     (cam-IMU clip)
#   <dataset>/cam_imu/imu0.csv       (EuRoC, no header)
#
# Board: charuco 10x8, square = 0.023 m, dict = DICT_ARUCO_ORIGINAL.
#   Usage:  bash calibration/run_openicc.sh <dataset_dir>
set -euo pipefail
RS="$HOME/rs_slam"
DATASET="${1:?usage: run_openicc.sh <dataset_dir>}"
DATASET="$(readlink -f "$DATASET")"
OICC="$RS/OpenImuCameraCalibrator"
APPS="$OICC/build/applications"
[ -x "$APPS/calibrate_camera" ] || { echo "OpenICC not built: run calibration/build_openicc.sh"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openicc                      # py3.10 env; provides 'python' for sub-scripts
export LD_LIBRARY_PATH="$RS/install/lib:${LD_LIBRARY_PATH:-}"

# Drop stale corner/pose artifacts so corners are re-extracted on the full clip.
rm -f "$DATASET"/cam/cam_corners.uson "$DATASET"/cam/cam_calib_*.json \
      "$DATASET"/cam_imu/cam_imu_corners.uson "$DATASET"/cam_imu/pose_calib.calibdata 2>/dev/null || true

cd "$OICC/python"
python run_mynteye_calibration.py \
  --path_calib_dataset="$DATASET" \
  --path_to_build="$APPS" \
  --camera_model=PINHOLE \
  --board_type=charuco \
  --checker_size_m=0.023 \
  --num_squares_x=10 \
  --num_squares_y=8 \
  --image_downsample_factor=1 \
  --recompute_corners=1 \
  --voxel_grid_size=0.02 \
  --verbose=1

echo
echo "=== results ==="
echo "  camera intrinsics : $DATASET/cam/cam_calib_*.json"
echo "  cam-IMU calib     : $DATASET/cam_imu/cam_imu_calib_result.json"
echo "Next: feed T_cam_imu + intrinsics into slam/configs (IMU.T_b_c1, Camera1.*)."
