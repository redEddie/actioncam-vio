#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "❌ 오류: 비디오 파일 이름을 입력하지 않으셨습니다!"
    echo "사용법: bash map.sh <비디오파일명>"
    exit 1
fi

VIDEO_NAME="$1"
TEST_VIDEO="video/${VIDEO_NAME}"
CAMERA_MODEL="hero13black"
OUT_DIR="result/${VIDEO_NAME%.MP4}_test"
mkdir -p "$OUT_DIR"
CALIB_DIR="cameras/${CAMERA_MODEL}/calibration"

echo "========================================================="
echo "ORB-SLAM3 맵 데이터 자동 생성 (GUI 끄고 알아서 연산 + 맵 렌더링)"
echo "결과 저장 위치: $OUT_DIR"
echo "========================================================="

echo "[1/3] 영상에서 IMU 데이터 추출 중..."
python -m gopro_vio.extract "$TEST_VIDEO" -o "$OUT_DIR"

echo "[2/3] ORB-SLAM3 백그라운드 연산 중 (창이 뜨지 않고 알아서 특징점 잡습니다)..."
# -g 옵션이 없으므로 실시간 창 없이 훨씬 빠르게 연산만 진행합니다.
python -m gopro_vio.slam "$TEST_VIDEO" --imu "$OUT_DIR/imu.csv" -o "$OUT_DIR/slam" \
    --calib "$CALIB_DIR/intrinsics.json" --extr "$CALIB_DIR/imu_extrinsics.json" \
    --width 960 --features 2500 --fps-div 2

echo "[3/3] 3D 맵 시각화 데이터 렌더링 중..."
python -m gopro_vio.rerun_viz "$OUT_DIR/slam" --video "$OUT_DIR/slam/video_slam_960x720_30fps.mp4"

# 맵 보기 실행 파일(더블클릭용) 생성
ABS_OUT_DIR=$(realpath "$OUT_DIR")
cat <<EOF > "$OUT_DIR/view_map.c"
#include <stdlib.h>
#include <unistd.h>
int main() {
    chdir("$ABS_OUT_DIR");
    system("bash -c 'source ~/miniconda3/bin/activate gopro-vio && rerun slam/map.rrd'");
    return 0;
}
EOF
gcc "$OUT_DIR/view_map.c" -o "$OUT_DIR/맵_바로보기"
rm "$OUT_DIR/view_map.c"

echo "✅ 3D 맵 생성 완료!"
echo "▶ 맵을 보시려면 해당 폴더에 들어가서 '맵_바로보기' 실행파일을 더블클릭 하세요!"
