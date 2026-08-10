#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "❌ 오류: 비디오 파일 이름을 입력하지 않으셨습니다!"
    echo "사용법: bash slam.sh <비디오파일명>"
    exit 1
fi

VIDEO_NAME="$1"
TEST_VIDEO="video/${VIDEO_NAME}"
CAMERA_MODEL="hero13black"
OUT_DIR="result/${VIDEO_NAME%.MP4}_test"
mkdir -p "$OUT_DIR"
CALIB_DIR="cameras/${CAMERA_MODEL}/calibration"

echo "========================================================="
echo "ORB-SLAM3 실시간 특징점 확인 (GUI 켜짐, 맵 렌더링 없음)"
echo "========================================================="

echo "[1/2] 영상에서 IMU 데이터 추출 중..."
python -m gopro_vio.extract "$TEST_VIDEO" -o "$OUT_DIR"

echo "[2/2] ORB-SLAM3 궤적 추출 진행 중 (실시간 창 띄움)..."
# -g 옵션을 포함하여 실시간 특징점 창(Pangolin GUI)을 띄워줍니다.
python -m gopro_vio.slam "$TEST_VIDEO" --imu "$OUT_DIR/imu.csv" -o "$OUT_DIR/slam" \
    --calib "$CALIB_DIR/intrinsics.json" --extr "$CALIB_DIR/imu_extrinsics.json" \
    --width 960 --features 2500 --fps-div 2 -g

echo "✅ 실시간 특징점 확인 종료!"
