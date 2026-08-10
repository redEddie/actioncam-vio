#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gopro-vio

VIDEO="/home/kimminje/Desktop/GX010013.MP4"
OUT="output/GX010013"

echo "[1/2] IMU 데이터 추출 중..."
python -m gopro_vio.extract "$VIDEO" -o "$OUT"

echo "[2/2] ORB-SLAM3 (VIO) 가동 중..."
# cameras/hero13black 에 캘리브레이션 데이터가 이미 존재함!
python -m gopro_vio.slam "$VIDEO" \
    --imu "$OUT/imu.csv" \
    -o "$OUT/slam" \
    --calib cameras/hero13black/calibration/intrinsics.json \
    --extr cameras/hero13black/calibration/imu_extrinsics.json \
    --width 960 --features 2500 --fps-div 2

echo "🎉 슬램 완료! 결과물 위치: $OUT/slam"
