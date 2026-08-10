# Calibration (Phase 1)

## Stage A1 - factory parameters  (done)
`python calibration/dump_factory_calib.py` -> `factory_calib_<serial>.json` for every
connected camera; the reliable intrinsics/baseline are already applied to slam/configs.

## Stage B - camera-IMU calibration (OpenICC)  [D435i / D455 only]
Tool: OpenImuCameraCalibrator (built to `~/rs_slam/install`, binaries in
`~/rs_slam/OpenImuCameraCalibrator/build/applications`). Python orchestration in the
`openicc` conda env.

**Board:** charuco 10 x 8, square = 0.023 m, dict = `DICT_ARUCO_ORIGINAL`.
(If detection fails the board uses another aruco dict -> tell me which.)

### 1. Record two clips (IR + IMU, emitter off)
```bash
conda activate umi
cd ~/rs_slam/umi_pipeline
# (a) intrinsics clip: board fills the view, move SLOWLY (no motion blur),
#     cover all corners + tilt angles, ~20-30 s
python recording/record.py --profile configs/cameras/d435i_slam.yaml --tag calcam
# (b) cam-IMU clip: board visible, excite ALL 6 axes (3 translation + 3 rotation),
#     move briskly but NO blur, ~30-60 s
python recording/record.py --profile configs/cameras/d435i_slam.yaml --tag calcamimu
```

### 2. Convert -> OpenICC dataset
```bash
python calibration/db3_to_openicc.py \
  --cam     data/recordings/d435i_slam_calcam_*.db3 \
  --cam_imu data/recordings/d435i_slam_calcamimu_*.db3 \
  --out     data/calib/d435i_openicc
```

### 3. Calibrate
```bash
bash calibration/run_openicc.sh data/calib/d435i_openicc
# -> data/calib/d435i_openicc/cam_imu/cam_imu_calib_result.json  (T_cam_imu, time offset)
#    data/calib/d435i_openicc/cam/cam_calib_*.json               (intrinsics)
```

### 4. Apply
Feed the calibrated intrinsics + `T_cam_imu` into `slam/configs` (`Camera1.*`,
`IMU.T_b_c1`), then enable IMU mode: `slam/run_slam.sh mapping <bag> --imu`.

## Stage B2 - IMU noise (optional, for tuning)
`fit_allan_variance` / `static_imu_calibration` binaries are built; needs a long
static IMU recording. Start with current generic noise values and refine later.
