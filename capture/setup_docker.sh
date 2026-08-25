#!/bin/bash
set -e

# 이전에 실패해서 남은 찌꺼기 컨테이너가 있다면 미리 삭제
echo "기존 찌꺼기 컨테이너를 강제 삭제합니다..."
docker rm -f patch1 patch2 2>/dev/null || true

echo "1단계: 베이스 이미지 확인 및 다운로드..."
docker pull chicheng/orb_slam3:latest

echo "2단계: [패치 1+3] 초기화 및 LOST 조건 완화 (gate28 생성 중)..."
docker run --name patch1 chicheng/orb_slam3:latest bash -c '
  cd /ORB_SLAM3 &&
  sed -i "s/mnMatchesInliers<30){/mnMatchesInliers<20){/" src/Tracking.cc &&
  sed -i "s/} else if (mnMatchesInliers<50 \&\& !mpAtlas->isImuInitialized()){/} else if (mnMatchesInliers<28 \&\& !mpAtlas->isImuInitialized()){/" src/Tracking.cc &&
  sed -i "s/time_recently_lost(0.2)/time_recently_lost(2.0)/" src/Tracking.cc &&
  cd build && make -j16 gopro_slam'

docker commit patch1 orb_slam3:gate28
docker rm patch1

echo "3단계: [패치 2+4] 크래시 방어 및 전체 맵 출력 코드 덮어쓰기 (gate28-rescue 생성 중)..."
docker run --name patch2 \
  -v $PWD/docker/gopro_slam_patched.cc:/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam.cc \
  -v $PWD/docker/System_patched.cc:/ORB_SLAM3/src/System.cc \
  -v $PWD/docker/System_patched.h:/ORB_SLAM3/include/System.h \
  orb_slam3:gate28 \
  bash -c 'cd /ORB_SLAM3/build && make -j16 gopro_slam'

docker commit patch2 orb_slam3:gate28-rescue
docker rm patch2

echo "=========================================================="
echo "모든 빌드가 완벽하게 성공했습니다! (orb_slam3:gate28-rescue)"
echo "=========================================================="
