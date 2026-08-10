#!/bin/bash
set -e

echo "기존 찌꺼기 컨테이너를 삭제합니다..."
docker rm -f patch2 2>/dev/null || true

echo "[패치 2] 크래시 방어 및 전체 서브맵 출력 패치 적용 후 재빌드..."
docker run --name patch2 \
  -v $PWD/docker/gopro_slam_patched.cc:/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam.cc \
  -v $PWD/docker/System_patched.cc:/ORB_SLAM3/src/System.cc \
  -v $PWD/docker/System_patched.h:/ORB_SLAM3/include/System.h \
  orb_slam3:gate28 \
  bash -c 'cd /ORB_SLAM3/build && make -j16 gopro_slam'

docker commit patch2 orb_slam3:gate28-rescue
docker rm patch2
echo "[패치 2] 완료! 최종 이미지 orb_slam3:gate28-rescue 생성 성공!"
