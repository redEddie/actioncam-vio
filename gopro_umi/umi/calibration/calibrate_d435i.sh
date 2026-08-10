#!/usr/bin/env bash
# End-to-end D435i camera + camera-IMU calibration (OpenICC), self-contained.
# Uses the NEWEST calcam / calcamimu .db3 in data/recordings.
#   - binaries run with SYSTEM libs (LD_LIBRARY_PATH=install/lib, NOT conda)
#   - python helper steps run with the openicc env
#   - calibrate_camera voxel size is retried until it converges
# Run from the repo root:  bash calibration/calibrate_d435i.sh
set -uo pipefail   # not -e: report failures, keep going
RS="$HOME/rs_slam"; PIPE="$RS/umi_pipeline"; OICC="$RS/OpenImuCameraCalibrator"
APPS="$OICC/build/applications"
UMIPY="$HOME/miniconda3/envs/umi/bin/python"
OICCPY="$HOME/miniconda3/envs/openicc/bin/python"
export LD_LIBRARY_PATH="$RS/install/lib"

CAM_DB=$(ls -t "$PIPE"/data/recordings/d435i_slam_calcam_*.db3 | head -1)
CI_DB=$(ls -t "$PIPE"/data/recordings/d435i_slam_calcamimu_*.db3 | head -1)
D="$PIPE/data/calib/d435i_openicc"
echo ">>> cam clip     : $(basename "$CAM_DB")"
echo ">>> cam_imu clip : $(basename "$CI_DB")"
rm -rf "$D"

echo "### [1] convert .db3 -> dataset (rs-convert)"
"$UMIPY" "$PIPE/calibration/db3_to_openicc.py" --cam "$CAM_DB" --cam_imu "$CI_DB" --out "$D"

CAMD="$D/cam"; CI="$D/cam_imu"; CAL="$CAMD/cam_calib_ph_1.0.json"
B="--board_type=charuco --checker_square_length_m=0.023 --num_squares_x=10 --num_squares_y=8 \
   --aruco_detector_params=$OICC/resource/charuco_detector_params.yml \
   --downsample_factor=1 --recompute_corners=1 --logtostderr=0"

echo "### [2] cam corners + camera intrinsics (retry voxel)"
$APPS/extract_board_to_json --input_path="$CAMD/cam0" --save_corners_json_path="$CAMD/cam_corners.uson" $B >/dev/null 2>&1
for g in 0.02 0.012 0.008 0.005 0.003; do
  $APPS/calibrate_camera --input_corners="$CAMD/cam_corners.uson" \
    --save_path_calib_dataset="$CAMD/cam_calib_ph_1.0" --camera_model_to_calibrate=PINHOLE \
    --grid_size=$g --optimize_board_points=0 --logtostderr=0 2>&1 | grep -iE "Using|reproj error" | sed "s/^/   [voxel=$g] /"
  [ -f "$CAL" ] && break
done
if [ -f "$CAL" ]; then
  echo "   INTRINSICS:"; "$OICCPY" -c "import json;d=json.load(open('$CAL'));i=d['intrinsics'];print('     focal=%.2f cx=%.2f cy=%.2f reproj_err=%.3f'%(i['focal_length'],i['principal_pt_x'],i['principal_pt_y'],d['final_reproj_error']))"
else
  echo "   !! intrinsic calibration FAILED (not enough diverse views)"; exit 1
fi

echo "### [3] cam_imu corners + telemetry"
$APPS/extract_board_to_json --input_path="$CI/cam0" --save_corners_json_path="$CI/cam_imu_corners.uson" $B >/dev/null 2>&1
"$OICCPY" -c "import sys;sys.path.insert(0,'$OICC/python');from telemetry_converter import TelemetryConverter;TelemetryConverter().convert_csv_telemetry_file('$CI/imu0.csv','$CI/imu.json')"
"$OICCPY" "$OICC/python/get_sew_for_dataset.py" --input_json_path="$CI/imu.json" --output_path="$CI/spline_info.json" --q_so3=0.99 --q_r3=0.97 >/dev/null 2>&1

echo "### [4] board poses"
$APPS/estimate_camera_poses_from_checkerboard --input_corners="$CI/cam_imu_corners.uson" \
  --camera_calibration_json="$CAL" --output_pose_dataset="$CI/pose_calib.calibdata" \
  --optimize_board_points=0 --logtostderr=0 >/dev/null 2>&1
echo "   poses: $([ -f "$CI/pose_calib.calibdata" ] && echo OK || echo FAIL)"

echo "### [5] IMU->cam rotation init"
$APPS/estimate_imu_to_camera_rotation --telemetry_json="$CI/imu.json" \
  --input_pose_calibration_dataset="$CI/pose_calib.calibdata" \
  --imu_rotation_init_output="$CI/imu_to_cam_calibration.json" --logtostderr=0 >/dev/null 2>&1

echo "### [6] continuous-time spline fusion"
$APPS/continuous_time_imu_to_camera_calibration \
  --gyro_to_cam_initial_calibration="$CI/imu_to_cam_calibration.json" --telemetry_json="$CI/imu.json" \
  --input_pose_dataset="$CI/pose_calib.calibdata" --input_corners="$CI/cam_imu_corners.uson" \
  --camera_calibration_json="$CAL" --imu_bias_file="$CI/imu_bias.json" --output_path="$CI" \
  --spline_error_weighting_json="$CI/spline_info.json" --result_output_json="$CI/cam_imu_calib_result.json" \
  --reestimate_biases=1 --global_shutter=1 --logtostderr=0 2>&1 | grep -iE "T_i_c|offset" | head

echo "### ===== CAM-IMU RESULT ====="
[ -f "$CI/cam_imu_calib_result.json" ] && "$OICCPY" -c "
import json;d=json.load(open('$CI/cam_imu_calib_result.json'))
print(' q_i_c (w,x,y,z):',[round(d['q_i_c'][k],5) for k in 'wxyz'])
print(' t_i_c (m)      :',[round(d['t_i_c'][k],5) for k in 'xyz'])
print(' time_offset_s  :',round(d['time_offset_imu_to_cam_s'],5))
print(' final_reproj_px:',round(d['final_reproj_error'],3),'   (<1 good, >>1 = did NOT converge)')
" || echo " FAILED to produce result"
