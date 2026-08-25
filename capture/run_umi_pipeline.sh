#!/bin/bash
set -e

# ========================================================
# [사용법] bash run_umi_pipeline.sh <맵_비디오파일명> <데모_비디오파일명>
# ========================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "❌ 오류: 비디오 파일 이름을 2개(맵 영상, 데모 영상) 입력하셔야 합니다!"
    echo "사용법: bash run_umi_pipeline.sh <맵_비디오파일명> <데모_비디오파일명>"
    echo "예시: bash run_umi_pipeline.sh map.MP4 demo.MP4"
    exit 1
fi

MAP_VIDEO="video/$1"
DEMO_VIDEO="video/$2"
CAMERA_MODEL="hero13black"

# 🌟 고프로 폴더 내부에 result 디렉토리 생성
MAP_OUT="result/${1%.MP4}_map"
DEMO_OUT="result/${2%.MP4}_demo"
mkdir -p "$MAP_OUT" "$DEMO_OUT"

CALIB_DIR="cameras/${CAMERA_MODEL}/calibration"

echo "========================================================="
echo "UMI 그리퍼 너비 측정 및 궤적 추출 파이프라인 시작"
echo "- 맵 영상: $MAP_VIDEO"
echo "- 데모 영상: $DEMO_VIDEO"
echo "- 결과 저장 폴더: result/ 폴더 내부"
echo "========================================================="

# 0. IMU 데이터 추출
echo "[0단계] 고프로 영상에서 IMU 센서 데이터 추출 중..."
python -m gopro_vio.extract "$MAP_VIDEO" -o "$MAP_OUT"
python -m gopro_vio.extract "$DEMO_VIDEO" -o "$DEMO_OUT"

# 1. 맵 영상 처리 (월드 앵커 및 맵 생성)
echo "[1단계] 맵 영상 분석 (SLAM 맵 생성 및 id13 마커 원점 잡기)..."
python -m gopro_vio.slam "$MAP_VIDEO" --imu "$MAP_OUT/imu.csv" -o "$MAP_OUT/slam" \
    --calib "$CALIB_DIR/intrinsics.json" --extr "$CALIB_DIR/imu_extrinsics.json" \
    --mask "$CALIB_DIR/gripper_mask.png" --init_tag_size 0.10 --width 960 --features 2500 --fps-div 2 -g

python -m gopro_vio.aruco_detect "$MAP_VIDEO" --calib "$CALIB_DIR/intrinsics.json" -o "$MAP_OUT/tags.pkl" --step 2 --ids 13
python -m gopro_vio.world_align "$MAP_OUT/tags.pkl" "$MAP_OUT/slam/camera_trajectory.csv" -o "$MAP_OUT/world" \
    --scale-range 0.5 1.5 --min-inliers 20 --max-median-residual-cm 2.0
python -m gopro_vio.tcp_robot_transform "$MAP_OUT/world/trajectory_world.csv" -o "$MAP_OUT/world"

# 2. 데모 영상 처리 (생성된 맵을 불러와서 궤적 추출)
echo "[2단계] 데모 영상 분석 (생성된 맵을 불러와서 정확한 궤적 복원)..."
python -m gopro_vio.slam "$DEMO_VIDEO" --imu "$DEMO_OUT/imu.csv" -o "$DEMO_OUT/slam" \
    --calib "$CALIB_DIR/intrinsics.json" --extr "$CALIB_DIR/imu_extrinsics.json" \
    --load-map "$MAP_OUT/slam/map_atlas.osa" \
    --mask "$CALIB_DIR/gripper_mask.png" --init_tag_size 0.10 --width 960 --features 2500 --fps-div 2 -g

python -m gopro_vio.world_align --apply "$MAP_OUT/world/tx_slam_tag.json" "$DEMO_OUT/slam/camera_trajectory.csv" -o "$DEMO_OUT/world"
python -m gopro_vio.tcp_robot_transform "$DEMO_OUT/world/trajectory_world.csv" -o "$DEMO_OUT/world"

# 3. 그리퍼 손가락 너비 측정
echo "[3단계] 데모 영상에서 그리퍼 너비(id0, id1 마커) 측정 중..."
python -m gopro_vio.aruco_detect "$DEMO_VIDEO" --calib "$CALIB_DIR/intrinsics.json" -o "$DEMO_OUT/tags.pkl" --step 1 --ids 0 1
python -m gopro_vio.gripper_width "$DEMO_OUT/tags.pkl" -o "$DEMO_OUT/gripper"

echo "========================================================="
echo "완료! 모든 Odom(궤적)과 그리퍼 측정 데이터가 result/ 폴더에 저장되었습니다."
echo "========================================================="
