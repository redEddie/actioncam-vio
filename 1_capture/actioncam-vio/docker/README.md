# 패치된 ORB-SLAM3 도커 이미지 (`orb_slam3:gate28-rescue`)

베이스: UMI의 `chicheng/orb_slam3:latest` (ORB-SLAM3 v1.0 포크, gopro_slam 바이너리 포함)

## 적용된 패치

1. **`src/Tracking.cc` — IMU 초기화 전 인라이어 게이트 완화**
   - `mnMatchesInliers<50 && !isImuInitialized` → `<28`
   - 재로컬라이즈 직후 게이트 `<30` → `<20`
   - 이유: 전진 보행 영상에서 단안 2-뷰 초기화의 시차가 ~0.3-1°에 불과해
     초기 맵의 인라이어가 31~48에 머물러 50 게이트에서 무한 리셋됨.
     게이트를 넘기면 키프레임이 쌓이며 맵이 강해지고 ~2초 후 IMU 초기화(VIBA)
     이후에는 원래 게이트(15)로 동작.

2. **`Examples/Monocular-Inertial/gopro_slam.cc`** (`gopro_slam_patched.cc` 참조)
   - 영상 끝에서 IMU 타임스탬프 배열 인덱스가 범위를 벗어나는 버그 수정
   - 1500프레임마다 궤적 체크포인트 저장 (`*.csv.ckpt`)
   - SIGSEGV/SIGABRT/SIGBUS 시 궤적 구조(rescue) 저장 (`*.csv.rescue`)
     — ORB-SLAM3의 알려진 스레드 경쟁 크래시 대비

## 재현 방법

```bash
docker run --name p chicheng/orb_slam3:latest bash -c '
  cd /ORB_SLAM3 &&
  sed -i "s/mnMatchesInliers<30){/mnMatchesInliers<20){/" src/Tracking.cc &&
  sed -i "s/} else if (mnMatchesInliers<50 \&\& !mpAtlas->isImuInitialized()){/} else if (mnMatchesInliers<28 \&\& !mpAtlas->isImuInitialized()){/" src/Tracking.cc &&
  cd build && make -j16 gopro_slam'
docker commit p orb_slam3:gate28 && docker rm p
# 그 다음 gopro_slam_patched.cc를 마운트해 재빌드:
docker run --name p2 -v $PWD/docker/gopro_slam_patched.cc:/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam.cc orb_slam3:gate28 \
  bash -c 'cd /ORB_SLAM3/build && make -j16 gopro_slam'
docker commit p2 orb_slam3:gate28-rescue && docker rm p2
```

3. **`src/Tracking.cc` — RECENTLY_LOST 유지시간 복원**
   - 포크가 0.2초로 줄여놓은 `time_recently_lost`를 2.0초로 복원
   - 이유: IMU 초기화 후 저텍스처 구간(무지 벽 근접 등)에서 순간적으로
     인라이어가 15 미만으로 떨어질 때, IMU 예측으로 버티며 재포착할 시간을
     확보. 0.2초면 즉시 LOST → 맵 리셋 → 아틀라스 분절.
   - 재현: `sed -i "s/time_recently_lost(0.2)/time_recently_lost(2.0)/" src/Tracking.cc`

4. **`src/System.cc` — 전체 서브맵 궤적 내보내기 (`SaveTrajectoryAllMapsCSV`)**
   - 기존 `SaveTrajectoryCSV`는 "최대 맵"의 프레임만 내보내고 나머지를 전부
     is_lost 처리 → 아틀라스가 분절된 영상에서 궤적이 사실상 빈 파일이 됨.
   - `map_id` 컬럼을 추가해 모든 서브맵의 프레임을 각자의 좌표계(각 맵 첫
     키프레임 원점, 중력 정렬·미터 스케일)로 내보내는 함수를 추가하고
     gopro_slam.cc의 정상 저장/rescue 경로에서 호출 (`*.allmaps`).
   - 소스: `System_patched.cc` / `System_patched.h`

## 이미지 계보

- `orb_slam3:gate28` — 패치 1만
- `orb_slam3:gate28-rescue` — 패치 1+2+3+4 (**기본 사용**)
- `orb_slam3:gate28-coast` — (실험) 추가로 IMU 초기화 후 추적 실패 시 맵 리셋
  대신 관성 코스팅: `sed -i "s/if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())/if(!pCurrentMap->isImuInitialized())/" src/Tracking.cc`
  - 짧은 가림에는 유효하나, 무지 벽 구간이 수 초 이상 이어지는 환경(GX014217)
    에서는 드리프트된 포즈로 재포착하며 **맵 자체가 오염**됨(40m 홀이 567m로
    폭발). 기본 이미지에서 제외.

5. **`Examples/Monocular-Inertial/export_map.cc` — 저장된 맵(.osa) → PLY 내보내기**
   - `map_atlas.osa`를 로드해 전체 서브맵의 3D 포인트(map_id 포함)와 키프레임
     위치를 PLY/CSV로 내보내는 별도 바이너리. SLAM 재실행 없이 저장된 맵을
     시각화할 때 사용. 좌표계는 `SaveTrajectoryCSV`와 동일(최대 맵 첫 KF 원점).
   ```bash
   docker run --rm -v $PWD/output/GX014220/slam:/work orb_slam3:gate28-rescue \
       /ORB_SLAM3/Examples/Monocular-Inertial/export_map \
       -s /work/orbslam3_settings.yaml -l /work/map_atlas.osa \
       -o /work/map_points.ply -k /work/keyframes.csv
   ```

## ⚠️ 빌드 시 주의: cmake 재구성 금지

이 이미지 계보에서 `cmake ..`를 다시 실행하거나 전체 클린 재빌드를 하면
(Thirdparty 포함 전체를 재빌드해도) **초기화 직후 전역 BA에서 힙 손상으로
결정적 크래시**가 발생한다(gdb: `EdgeSE3ProjectXYZ::linearizeOplus` SIGSEGV
/ `g2o::HyperGraph::clear` double free). 원저자 빌드 트리의 구성 상태와
새 configure 결과 사이의 미규명 불일치가 원인으로 추정.
**항상 기존 build/ 디렉토리에서 증분 `make`만 사용할 것.**
새 실행 파일 추가는 gopro_slam의 flags.make/link.txt를 재사용해 수동
컴파일·링크로 해결(export_map가 그 예).
