#!/bin/bash
set -e

# ========================================================
# [사용법] bash test_slam.sh <비디오파일명>
# ========================================================

if [ -z "$1" ]; then
    echo "❌ 오류: 비디오 파일 이름을 입력하지 않으셨습니다!"
    echo "사용법: bash test_slam.sh <비디오파일명>"
    echo "예시: bash test_slam.sh GX010002.MP4"
    exit 1
fi

VIDEO_NAME="$1"
TEST_VIDEO="video/${VIDEO_NAME}"
CAMERA_MODEL="hero13black"

# 🌟 고프로 폴더(현재 폴더) 내부에 result 디렉토리 생성
OUT_DIR="result/${VIDEO_NAME%.MP4}_test"
mkdir -p "$OUT_DIR"

CALIB_DIR="cameras/${CAMERA_MODEL}/calibration"

echo "========================================================="
echo "ORB-SLAM3 단독 구동 테스트 시작 (입력: $TEST_VIDEO)"
echo "결과 저장 위치: $OUT_DIR"
echo "========================================================="

echo "[1/3] 영상에서 IMU 데이터 추출 중..."
python -m gopro_vio.extract "$TEST_VIDEO" -o "$OUT_DIR"

echo "[2/3] ORB-SLAM3 궤적 추출 진행 중 (도커 컨테이너 사용)..."
python -m gopro_vio.slam "$TEST_VIDEO" --imu "$OUT_DIR/imu.csv" -o "$OUT_DIR/slam" \
    --calib "$CALIB_DIR/intrinsics.json" --extr "$CALIB_DIR/imu_extrinsics.json" \
    --width 960 --features 2500 --fps-div 2 -g

echo "[3/3] 3D 궤적 및 맵 시각화 데이터 생성 중..."
python -m gopro_vio.rerun_viz "$OUT_DIR/slam" --video "$OUT_DIR/slam/video_slam_960x720_30fps.mp4"

# 맵 보기 실행 파일(더블클릭용) 생성
# 우분투 탐색기 기본 보안 설정상 텍스트(.sh, .desktop) 더블클릭이 막혀있어서,
# 더블클릭으로 바로 실행되도록 진짜 어플리케이션(바이너리)으로 즉석에서 컴파일하여 만듭니다.
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
rm -f "$OUT_DIR/맵_바로보기.desktop"

echo "ORB-SLAM3 테스트 완료! 모든 결과가 $OUT_DIR 폴더에 저장되었습니다."
echo "▶ 맵을 보시려면 언제든지 해당 폴더에 들어가서 '맵_바로보기' 실행파일을 더블클릭 하세요!"
