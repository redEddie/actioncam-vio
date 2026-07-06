# gopro_slam — GoPro 액션캠 기반 VIO 파이프라인 (UMI 스타일)

[Universal Manipulation Interface(UMI)](https://umi-gripper.github.io/)처럼 GoPro
액션캠 하나로 촬영한 MP4에서 RGB + IMU를 뽑아 캘리브레이션하고, ORB-SLAM3
monocular-inertial로 맵을 만들고 카메라 궤적(미터 스케일)을 복원하는 파이프라인입니다.

- 카메라: **GoPro HERO7 Black**, 2704×2028(4:3) @ 59.94 fps, IMU(GPMF) ≈ 197.7 Hz
- 캘리브레이션 보드: ChArUco 10×8칸, 정사각형 0.023 m
  (`DICT_ARUCO_ORIGINAL`, legacy 패턴 — 파이프라인이 자동 탐지)
- SLAM: UMI와 동일한 [chicheng/ORB_SLAM3](https://hub.docker.com/r/chicheng/orb_slam3)
  도커 이미지 + 로컬 패치(`docker/README.md` 참조)
- 환경: miniconda / **Python 3.13**, OpenCV 5.0, NumPy 2.5 (`environment.yml`)

## 설치

```bash
conda env create -f environment.yml
conda activate gopro-vio
docker pull chicheng/orb_slam3:latest
# 패치 이미지 빌드(필수): docker/README.md의 재현 명령 실행 → orb_slam3:gate28-rescue
```

## 파이프라인

```
MP4 ─┬─> gopro_vio.extract   GPMF 파싱 → imu.csv / imu_data.json / video_info.json
     │
     ├─> gopro_vio.charuco   (보드 영상) ChArUco 검출 → KB4 피쉬아이 내부파라미터
     │                        + 프레임별 보드 포즈 타임라인 (detection 캐시 지원)
     │
     ├─> gopro_vio.imu_sync  gyro↔영상 각속도 정렬 → 시간 오프셋 + R_imu_cam
     │                        + 중력 일관성 검증
     │
     ├─> gopro_vio.slam      ORB-SLAM3 yaml 생성 + 영상 트랜스코딩(30fps/960px)
     │                        + IMU 리샘플 → docker SLAM → camera_trajectory.csv
     │
     └─> gopro_vio.viz       궤적 3D/탑뷰/시계열 플롯 + 통계
```

### 사용 예

```bash
# 1. 텔레메트리 추출 (모든 영상)
python -m gopro_vio.extract data/GX014222.MP4 -o output/GX014222

# 2. 카메라 내부파라미터 (ChArUco 영상)
python -m gopro_vio.charuco data/GX014222.MP4 -o calibration \
    --squares 10 8 --square-size 0.023

# 3. IMU-카메라 시간/회전 캘리브레이션
python -m gopro_vio.imu_sync calibration/board_poses.npz \
    output/GX014222/imu.csv -o calibration

# 4. SLAM (걷기 영상)
python -m gopro_vio.slam data/GX014219.MP4 \
    --imu output/GX014219/imu.csv -o output/GX014219/slam \
    --width 960 --features 2500 --fps-div 2
#   --start-s 30 : 앞부분 N초(저텍스처 구간 등)를 건너뛰고 시작
#   --features/--ini-fast/--min-fast : 특징점 빈곤 환경에서 상향/하향 조정

# 5. 궤적 시각화 (정적 PNG)
python -m gopro_vio.viz output/GX014219/slam/camera_trajectory.csv

# 6. 맵 시각화 — 저장된 맵(.osa)에서 점군 추출 후 rerun 인터랙티브 뷰어
docker run --rm -v $PWD/output/GX014220/slam:/work orb_slam3:gate28-rescue \
    /ORB_SLAM3/Examples/Monocular-Inertial/export_map \
    -s /work/orbslam3_settings.yaml -l /work/map_atlas.osa \
    -o /work/map_points.ply -k /work/keyframes.csv
python -m gopro_vio.rerun_viz output/GX014220/slam \
    --video output/GX014220/slam/video_slam_960x720_30fps.mp4
rerun output/GX014220/slam/map.rrd   # 뷰어 열기 (맵+궤적+카메라+영상 타임라인)
```

## 캘리브레이션 결과 (이 데이터셋)

| 항목 | 값 |
|---|---|
| KB4 fisheye (2704×2028) | fx=1215.0, fy=1217.3, cx=1343.3, cy=1030.4 |
| 왜곡 k1~k4 | 0.0476, 0.0191, -0.0109, 0.0023 |
| 재투영 RMS | **0.826 px** (60 뷰) |
| IMU-영상 시간 오프셋 | **-53 ms** (IMU가 앞섬, 반복 추정 ±2 ms) |
| R_imu_cam | cam x→imu y, cam y→-imu z, cam z→-imu x (± <0.6°) |
| 중력 검증 | 보드 좌표계에서 \|g\|=9.49 m/s², 방향 편차 2.3% |

## 구현 노트

- **GPMF 파서는 순수 Python** (`gopro_vio/mp4.py`, `gopro_vio/gpmf.py`).
  MP4 `gpmd` 트랙의 stts/stsc/stco를 직접 파싱해 페이로드별 시간을 얻고,
  샘플 인덱스↔페이로드 시작시간의 최소제곱 직선 적합으로 샘플별 타임스탬프를
  복원합니다(gpmf-parser의 `GetGPMFSampleRate` 방식).
- **IMU 축 규약**: GPMF 원시 축 순서를 그대로 사용하고 R_imu_cam을 같은
  데이터로 직접 추정 → 규약이 파이프라인 전체에서 자기일관적.
- **IMU-카메라 회전/시간 정렬**: ChArUco 보드 포즈 시퀀스에서 카메라 각속도를
  계산하고 자이로와 (8 Hz 저역통과 후) 교차상관 → 시간 오프셋,
  Kabsch/Wahba → R_imu_cam. 병진(lever arm)은 자이로만으로는 관측 불가라 0으로 둠.
- **`gopro_slam` 바이너리 특성** (소스 분석):
  - IMU json의 첫 ACCL 샘플 cts를 0으로 재정렬하고 ACCL/GYRO를 인덱스로 페어링
    → 두 스트림을 **video t=0 기준 공통 균일 그리드(200 Hz)로 리샘플**해 공급
    (`gopro_vio.slam`이 수행, 끝부분 0.5 s 패딩 포함)
  - yaml의 `Camera.width/height`로 입력 프레임 리사이즈, `Camera.fps`는 정수 필수
  - 프레임 타임스탬프는 `frame_idx / fps` (컨테이너의 구형 FFmpeg가 2.7K GoPro
    원본을 못 읽어 사전 트랜스코딩 필수)
- Hero7에는 CORI/GRAV 스트림이 없지만(HERO8+ 전용) 바이너리는 없어도 동작.
- 전 영상에서 **EIS(HyperSmooth) OFF 확인**(udta GPMF `EISE=N`) — VIO 기하 보존.
  촬영 시 반드시 끄고 찍을 것.

## TODO

- [ ] **분절된 아틀라스에서 전체 궤적 복원**: 현재 `SaveTrajectoryCSV`는 "최대
  맵" 기준으로만 내보내서, 맵이 조각난 영상(GX014221, GX014217)은 추적이
  잘 되던 구간도 궤적에서 `is_lost`로 빠짐. 서브맵별 궤적 내보내기 →
  IMU 적분/중력 정렬로 이어붙이는 후처리 필요 (연구노트 §3)
- [ ] ORB-SLAM3 LOST/리셋 경로의 비결정적 세그폴트 backtrace 확보 및 근본 수정
  (현재는 rescue 핸들러로 궤적만 보존; GX014219는 맵(.osa) 저장 불가)
- [ ] IMU-카메라 병진(lever arm) 캘리브레이션 (가속도계 활용, Kalibr/OpenICC 방식)
- [ ] 가속도계 스케일 보정 (측정 |g|=9.49 vs 9.81, -3.3%)
- [ ] 롤링 셔터 모델링 (캘리브레이션 + SLAM)
- [ ] 특징점 빈곤 환경 대응: 학습 특징점(SuperPoint+LightGlue 등) 프런트엔드 검토
- [ ] 촬영 프로토콜 정립: 시작 시 옆걸음/회전으로 시차 확보, EIS·자동저조도 OFF
- [ ] 미래지향 백엔드 검토: DPVO/DROID-SLAM(GPU) + IMU 융합으로 교체 실험
- [ ] **Ace Pro 2 다른 VIO 백엔드**: Basalt / OKVIS / Kimera-VIO 비교 실험 (현재 ORB-SLAM3만)
- [ ] Ace Pro 2 994 Hz IMU를 200 Hz로 리샘플할 때 안티에일리어싱 저역통과 추가
- [ ] Kalibr 포맷 내보내기(rosbag/yaml)로 Basalt·OKVIS 계열 캘리브레이션 호환

## 연구 필요사항 (Research Notes)

실행 과정에서 확인된, "기존 알고리즘이 잘 안 되는" 지점들입니다.

### 1. 전진 보행 영상에서 단안-관성 초기화의 구조적 취약성 ★핵심
- ORB-SLAM3 mono-inertial의 2-뷰 초기화는 최소 시차 1°를 요구하는데, 전방 주시
  보행(광축 방향 이동 + 원거리 장면)에서는 인접 프레임 시차가 0.3~1°에 불과.
  초기화가 "성공"해도 깊이 불확실성이 큰 맵이 만들어져 다음 프레임에서
  인라이어가 30~48개로 붕괴 → 50개 게이트에 걸려 무한 리셋.
- 60fps/30fps/10fps, 960/1280px, 2000~3500 특징점 어느 조합에서도 동일하게 재현.
  파이썬으로 동일 지오메트리를 재구성해 코드 버그가 아닌 **데이터 영역(regime)
  문제**임을 검증함 (E-행렬 인라이어 90%인데 삼각측량 시차 중앙값 0.32°).
- **UMI가 ArUco 큐브 태그 초기화를 쓰는 실질적 이유로 추정** — 태그가 없으면
  이 취약성이 그대로 드러남.
- 임시 해결: IMU 초기화 전 인라이어 게이트 50→28 완화(패치 이미지). IMU 초기화
  (VIBA) 후에는 원래 게이트(15)로 동작하므로 안정성 손실 최소.
- 연구 방향: (a) IMU-우선 초기화(자이로 적분으로 회전 제거 후 병진 시차만 평가),
  (b) 깊이 사전(prior)을 주는 학습 기반 초기화, (c) UMI처럼 시작 구간에
  태그/보드를 두는 운용 프로토콜.

### 2. ORB-SLAM3(chicheng 포크)의 견고성 문제
- 추적 실패/리셋 경로와 영상 종료 시점에서 **재현 가능한 세그폴트**:
  - 영상 끝에서 IMU 타임스탬프 배열 범위 초과 접근(경계 검사 없는 while) → 수정
  - LOST→리셋/재로컬라이즈 경로의 스레드 경쟁으로 추정되는 크래시(비결정적,
    다른 프레임 위치에서 재현) → 근본 수정 대신 **주기적 궤적 체크포인트 +
    크래시 시그널 핸들러의 rescue 저장**으로 방어 (크래시 시에도 궤적 보존)
- 연구 방향: 크래시 backtrace 확보 후 업스트림 수정, 또는 유지보수되는 현대적
  VIO(OpenVINS, VINS-Fusion, DPVO/DROID-SLAM 계열)로 백엔드 교체 검토.

### 3-1. GX014217 "30초 이후" 실험 (2026-07-05)
앞 30초(반대편 공간 + 무지 벽 근접 통과)를 잘라내고 `--start-s 30`으로도
시도했으나 **세 가지 구성 모두 만족스러운 궤적을 얻지 못함**:

| 구성 | 결과 |
|---|---|
| 기본 (리셋 정책 그대로) | 129초 동안 맵 64회 조각남, 리셋된 맵의 프레임은 소실 → 사용 가능 궤적 없음 |
| + IMU 코스팅 2초 (`gate28-coast`) | 맵 3개로 유지 + IMU 초기화 성공. 그러나 코스팅 시간(42s)이 실제 추적(31s)보다 길어 드리프트된 포즈로 재포착 → **맵 기하 오염**(40m 홀이 567m로 폭발) |
| 코스팅 창 축소 | 무지 벽 구간이 수 초 이상이라 다시 조각남 (중간값 없음) |

원인: 이 홀은 무지 파티션이 시야를 가득 채우는 구간이 수 초씩 반복되어,
특징점 기반 추적이 끊기는 시간 동안 소비자급 IMU 단독 항법(미성숙한
bias/scale)으로 기하를 유지할 수 없음. **ORB 기반 mono-inertial의 명확한
한계 사례**로, 아래 연구 방향의 대표 벤치마크 영상으로 활용 가치가 있음.

### 3. 특징점 빈곤 환경 (GX014217 전시홀) + 아틀라스 분절
- 대형 무지 파티션 + 균일 카펫 환경에서는 게이트 완화 후에도 인라이어 확보가
  빠듯함(ORB 코너 기반의 한계). 프레임 단위 추적은 90.8% OK였으나 잦은 맵
  리셋으로 **아틀라스가 여러 소형 맵으로 분절**되어, "최대 맵" 기준으로
  내보내는 `SaveTrajectoryCSV`가 사실상 빈 궤적을 출력.
- GX014221(보도)에서도 약하게 발생 — 추적 96.7% OK인데 최대 맵 커버리지 53%.
- 연구 방향: (a) 학습 특징점(SuperPoint/DISK+LightGlue), 라인/평면 특징 병용,
  직접법(DSO 계열) 하이브리드, (b) 아틀라스 서브맵 병합(map merging) 후
  전체 궤적을 이어붙이는 후처리, (c) 분절 맵 각각을 내보내 IMU 적분으로
  연결하는 스티칭.

### 4. 미모델링 요소 (정밀도 개선 여지)
- **롤링 셔터**: Hero7 2.7K의 리드아웃(~15-30 ms)이 캘리브레이션·SLAM 모두에서
  무시됨. 빠른 회전 시 계통 오차 유발 → RS 보정 모델(예: Kalibr RS, RS-aware VIO) 필요.
- **IMU-카메라 병진(lever arm)**: 자이로 정렬만으로는 관측 불가라 0으로 가정
  (실제 1~2 cm). 가속도계까지 쓰는 완전 캘리브레이션(Kalibr/OpenICC 방식) 필요.
- **가속도계 스케일**: 측정 |g|=9.49 m/s² (명목 9.81 대비 -3.3%) — GoPro 공장
  캘리브레이션 한계. 6면 정적 캘리브레이션 또는 온라인 스케일 추정으로 개선 가능.
- **자동 저조도(VLTE=Y)**: 어두운 곳에서 셔터/게인 변동 → 모션 블러 증가 가능.
  촬영 설정에서 끄는 것을 권장.

### 5. 시간 오프셋의 출처
- GPMF cts와 영상 PTS 사이 **-53 ms** 오프셋이 일관되게 측정됨(원인: 노출 중점
  vs PTS, 인코더 지연 등 복합 추정). 카메라 개체/모드별로 다를 수 있어
  데이터셋마다 재추정 필요 → 본 파이프라인은 ChArUco 영상에서 자동 추정.

## Insta360 Ace Pro 2 지원 (data/acepro2)

GoPro(GPMF)와 달리 Insta360은 텔레메트리를 **MP4 트레일러**(파일 끝, ffprobe에
안 보임)에 독자 포맷으로 저장한다. 추출은
[telemetry-parser](https://github.com/AdrianEddy/telemetry-parser)를 사용
(`gopro_vio/insta360.py` — 단위 자동 감지: gyro deg/s→rad/s, accl g→m/s²).

```bash
# 설치 (pypi 버전은 깨져 있음 — git 소스 + 최신 maturin + pyo3 3.13 우회 필요)
pip install -U maturin
git clone --depth 1 https://github.com/AdrianEddy/telemetry-parser.git /tmp/tp
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 pip install /tmp/tp/bin/python-module --no-build-isolation

# 파이프라인 (GoPro와 동일 구조, 추출기만 다름)
python -m gopro_vio.insta360 data/acepro2/VID_..._527.mp4 -o output/acepro2_527
python -m gopro_vio.charuco data/acepro2/VID_..._527.mp4 -o calibration_acepro2/v527 --squares 10 8 --square-size 0.023
python -m gopro_vio.calib_report calibration_acepro2/v527       # 커버리지+FOV
python -m gopro_vio.imu_sync calibration_acepro2/v527/board_poses.npz output/acepro2_527/imu.csv -o calibration_acepro2
python -m gopro_vio.slam data/acepro2/VID_..._525.mp4 --imu output/acepro2_525/imu.csv \
    -o output/acepro2_525/slam --calib calibration_acepro2/v527/intrinsics.json \
    --extr calibration_acepro2/imu_extrinsics.json --width 960 --features 2500 --fps-div 2
```

### 측정된 특성 (vs GoPro HERO7)

| 항목 | Ace Pro 2 | HERO7 Black |
|---|---|---|
| 영상 | 2688×2016 @59.94, HEVC | 2704×2028 @59.94, H.264 |
| IMU | **994 Hz**, \|g\| 9.72-9.93 (스케일 오차 ~1%) | 197.7 Hz, \|g\| 9.49 (-3.3%) |
| FOV (H/D) | **130.1° / 152.3°** | 120.5° / 149.4° |
| 캘리브 RMS | 0.82 px (두 영상 fx 0.2% 일치) | 0.83 px |
| IMU-영상 오프셋 | **-2.3 ms** (telemetry-parser 정렬 우수) | -53 ms |
| 인카메라 안정화 | OFF 확인(배럴 왜곡 보존) | OFF 확인(EISE=N) |

### Ace Pro 2 검증 결과 (2026-07-06)

**근거리 ATE** (보드 영상, ChArUco mm급 기준 대비, `board_eval`):

| 영상 | 정렬 | ATE RMSE | 중앙값 | 스케일 계수 |
|---|---|---|---|---|
| VID_526 (36s 전체) | rigid (scale=1) | **1.23 cm** | 0.98 cm | — |
| VID_526 | similarity | 0.47 cm | 0.21 cm | **1.067** |
| VID_527 (건강 구간 2-24s) | rigid | 3.36 cm | 2.95 cm | — |

- GoPro(1.88 cm, 스케일 1.126) 대비 **절대 오차·스케일 오차 모두 개선** —
  가속도계 품질(~1% vs -3.3%) 차이와 일관.
- VID_527의 24-42s 보드 초근접 구간에서 GoPro와 동일한 스케일 폭주 발생 후
  회복 — **근접-평면 한계는 카메라와 무관한 mono-inertial 공통 현상**임을 확인.

**야외 드리프트** (VID_528, 399 m 공원 산책, 루프 클로저 미발동 = 순수 오도메트리):
재방문 일관성 중앙값 1.82 m / p90 2.71 m (상한선), 수직 드리프트 +0.8 m (0.2%).

## 근거리 정밀도 검증 (ChArUco 기준 대비 ATE)

캘리브레이션 영상(GX014222)은 보드 대비 mm급 기준 궤적(solvePnP, 3,299 포즈)을
공짜로 제공하므로, 같은 영상의 VIO 궤적과 정합해 근거리 정밀도를 정량화했다
(`gopro_vio.board_eval`, 보드 15~63 cm 거리에서 손으로 흔든 48초 구간):

| 정렬 | ATE RMSE | 중앙값 | p95 | 의미 |
|---|---|---|---|---|
| rigid (scale=1) | **1.88 cm** | 1.71 cm | 3.08 cm | 미터 스케일 그대로의 절대 오차 |
| similarity (scale 자유) | **0.34 cm** | 0.22 cm | 0.59 cm | 궤적 형상 자체는 mm급 |

- 잔여 오차의 지배 성분은 **단일 스케일 계수 1.126** (IMU 스케일 추정 오차 —
  가속도계 -3.3% 스케일, lever arm 미보정과 일관). 형상은 3 mm RMSE.
- t≈50 s 이후 보드 초근접 구간(화면 전체가 평면 + 고속 모션)에서는 스케일이
  드리프트해 궤적이 부풀어짐 → 연구노트 §4에 추가된 근접-평면 한계 사례.
- 재현: `python -m gopro_vio.board_eval calibration/board_poses.npz \
  output/GX014222/slam/camera_trajectory.csv -o output/GX014222/eval \
  --ok-only --t-min 2 --t-max 50`
- 시각화: `output/GX014222/eval/board_eval.png`, rerun `output/GX014222/slam/map.rrd`

## 이 데이터셋의 SLAM 결과

| 영상 | 환경 | 길이 | 결과 | 경로/범위 |
|---|---|---|---|---|
| GX014219 | 공원 산책로(야외) | 152 s | **전 구간 궤적** (99.7% 추적) | 172 m, 42×60 m |
| GX014220 | 연못 공원(야외) | 256 s | **클린 완주 + 맵 저장** (91.5%) | 307 m, 62×99 m |
| GX014221 | 보도/거리(야외) | 111 s | 부분 궤적 (최대 맵 53%) | 72 m |
| GX014217 | 전시홀(실내, 무지 파티션) | 159 s | **실패** — 맵 분절로 내보내기 불가 (프레임 추적은 90.8% OK) | — |

궤적은 중력 정렬 좌표계·미터 스케일이며 보행 속도가 1.18~1.2 m/s로 복원되어
IMU 스케일 추정이 물리적으로 일관됨을 확인했습니다.

## 결과물

각 `output/<VIDEO>/slam/`에:
- `camera_trajectory.csv` — 프레임별 포즈 (frame_idx, timestamp, state, is_lost,
  is_keyframe, x, y, z, q_x, q_y, q_z, q_w), 미터 스케일·중력 정렬 좌표계
- `camera_trajectory.tum` — TUM 포맷 (evo 등 평가 도구 호환)
- `map_atlas.osa` — ORB-SLAM3 맵(정상 종료 시)
- `trajectory.png`, `trajectory_stats.json` — 시각화·통계
