# actioncam-vio — 액션캠 기반 VIO 파이프라인 (UMI 스타일)

[Universal Manipulation Interface(UMI)](https://umi-gripper.github.io/)처럼 액션캠
하나로 촬영한 MP4에서 RGB + IMU를 뽑아 캘리브레이션하고, ORB-SLAM3
monocular-inertial로 맵을 만들고 카메라 궤적(미터 스케일·중력 정렬)을 복원합니다.

```
MP4 ─┬─> IMU 추출        gopro_vio.extract (GPMF) / gopro_vio.insta360 (Insta360)
     ├─> 카메라 캘리브    gopro_vio.charuco + calib_report  (KB4 fisheye, 커버리지/FOV)
     ├─> IMU-카메라 정렬  gopro_vio.imu_sync  (시간 오프셋 + R_imu_cam + 중력 검증)
     ├─> VIO             gopro_vio.slam  (ORB-SLAM3 mono-inertial, docker)
     ├─> 평가            gopro_vio.board_eval  (ChArUco 기준 근거리 ATE)
     ├─> 시각화          gopro_vio.viz / rerun_viz / map_viz / sphere_viz
     └─> 도메인 정렬      gopro_vio.remap  (녹화→웹캠/롤아웃 기하 변환, VLA용)
```

## 지원 카메라

| 카메라 | IMU | FOV(H) | 근거리 ATE† | 상세 페이지 |
|---|---|---|---|---|
| **Insta360 Ace Pro 2** | 994 Hz | 130.1° | **1.23 cm** | [cameras/acepro2](cameras/acepro2/README.md) |
| **GoPro HERO7 Black** | 197.7 Hz | 120.5° | 1.88 cm | [cameras/hero7black](cameras/hero7black/README.md) |

† ChArUco mm급 기준 궤적 대비 rigid ATE RMSE (보드 15~63 cm 근거리, 스케일 보정 없음)

새 카메라 추가 = IMU 추출 모듈 하나 작성 + ChArUco 보드 영상 1~2개 촬영이 전부입니다.
나머지(캘리브레이션→정렬→SLAM→평가)는 카메라 무관 공용입니다.

## 설치

```bash
conda env create -f environment.yml && conda activate gopro-vio
docker pull chicheng/orb_slam3:latest
# 패치 이미지 빌드(필수): docker/README.md 참조 → orb_slam3:gate28-rescue
# Insta360 카메라 사용 시: cameras/acepro2/README.md의 telemetry-parser 설치 참조
```

## 사용법

카메라별 전체 명령은 각 상세 페이지에 있습니다. 공통 형태:

```bash
# 1. IMU 추출 (카메라별 모듈)
python -m gopro_vio.extract <영상.MP4> -o output/<이름>            # GoPro
python -m gopro_vio.insta360 <영상.mp4> -o output/<이름>           # Insta360

# 2. 캘리브레이션 (ChArUco 보드 영상, 예: 10x8칸 0.023 m)
python -m gopro_vio.charuco <보드영상> -o cameras/<모델>/calibration \
    --squares 10 8 --square-size 0.023
python -m gopro_vio.calib_report cameras/<모델>/calibration        # 커버리지+FOV

# 3. IMU-카메라 시간/회전 정렬 (같은 보드 영상의 IMU 사용)
python -m gopro_vio.imu_sync cameras/<모델>/calibration/board_poses.npz \
    output/<보드영상이름>/imu.csv -o cameras/<모델>/calibration

# 4. VIO
python -m gopro_vio.slam <영상> --imu output/<이름>/imu.csv -o output/<이름>/slam \
    --calib cameras/<모델>/calibration/intrinsics.json \
    --extr cameras/<모델>/calibration/imu_extrinsics.json \
    --width 960 --features 2500 --fps-div 2
#   --start-s N : 앞부분 N초 스킵 / --features, --ini-fast: 저텍스처 환경 조정

# 5. 시각화 / 평가
python -m gopro_vio.viz output/<이름>/slam/camera_trajectory.csv
python -m gopro_vio.rerun_viz output/<이름>/slam --video <트랜스코딩된 영상>
python -m gopro_vio.board_eval cameras/<모델>/calibration/board_poses.npz \
    output/<보드영상이름>/slam/camera_trajectory.csv -o output/<보드영상이름>/eval --ok-only
```

맵 점군 내보내기(rerun 시각화용)는 도커의 `export_map` 사용 — `docker/README.md` 참조.

## 촬영 가이드 (실측 기반)

- **인카메라 안정화 반드시 OFF** (GoPro HyperSmooth / Insta360 FlowState) — 켜면 VIO 기하가 깨짐
- 시작 시 **옆걸음/회전으로 시차를 만들어** 초기화를 도울 것 (전진만 하면 초기화 취약 — 연구노트 §1)
- 캘리브레이션 보드 촬영 시 **보드가 화면을 가득 채울 만큼 다가가지 말 것** (근접-평면 스케일 폭주 — 연구노트 §4)
- 보드 영상은 이미지 모서리까지 보드를 움직여 커버리지 확보

## 구현 노트 (공통)

- **캘리브레이션·정렬은 전부 데이터 기반 자기일관**: 보드 사전(dictionary)·레이아웃 자동 탐지,
  IMU 축 규약은 원시값 그대로 두고 R_imu_cam을 직접 추정, 시간 오프셋은 gyro↔영상
  각속도 교차상관(8 Hz 공통대역 필터 + 포물선 정밀화)으로 추정.
- **ORB-SLAM3 도커**: UMI의 `chicheng/orb_slam3`에 로컬 패치(초기화 게이트 완화,
  크래시 rescue 저장, 전체 서브맵 내보내기, IMU 경계 버그 수정). 재현 방법과
  ⚠️ *cmake 재구성 금지* 주의사항은 [docker/README.md](docker/README.md).
- 입력은 960px/30fps로 트랜스코딩(기하 무손실 — 내부파라미터 동일 배율 스케일),
  IMU는 video t=0 기준 200 Hz 균일 그리드로 리샘플.

## 연구 필요사항 (Research Notes)

실행 과정에서 확인된, 기존 알고리즘이 잘 안 되는 지점들.

### 1. 전진 보행 영상에서 단안-관성 초기화의 구조적 취약성 ★핵심
전방 주시 보행(광축 방향 이동 + 원거리 장면)에서는 초기화 시차가 0.3~1°에 불과해
"성공" 판정에도 깊이가 엉망인 맵이 생기고, IMU 초기화 전 50-인라이어 게이트에서
무한 리셋됨. 파이썬 기하 재현으로 코드 버그가 아닌 **데이터 영역 문제**임을 검증.
UMI가 ArUco 큐브 태그 초기화를 쓰는 실질적 이유로 추정. 임시 해결: 게이트 완화
(50→28, 패치 이미지) — IMU 초기화(VIBA) 후에는 원래 게이트(15)로 동작.
연구 방향: IMU-우선 초기화, 학습 기반 깊이 사전, 시작 지점 태그 프로토콜.

### 2. ORB-SLAM3(chicheng 포크)의 견고성
LOST/리셋 경로와 영상 종료 시점의 비결정적 세그폴트 → 주기적 체크포인트 +
크래시 시그널 핸들러 rescue로 방어(궤적 보존). 근본 수정 또는 유지보수되는
백엔드(Basalt/OKVIS/DPVO 계열)로의 교체 검토 필요.

### 3. 특징점 빈곤 환경 + 아틀라스 분절 (HERO7 전시홀 사례)
무지 파티션이 시야를 수 초씩 채우는 환경에서 맵 분절 → "최대 맵"만 내보내는
`SaveTrajectoryCSV`가 빈 궤적 출력(→ `SaveTrajectoryAllMapsCSV` 패치로 서브맵별
내보내기 가능). IMU 코스팅 연장은 드리프트된 재포착으로 **맵을 오염**시켜 역효과
(40 m 홀이 567 m로 폭발). 연구 방향: 학습 특징점, 서브맵 병합 후처리, IMU 스티칭.

### 4. 근접-평면 스케일 폭주 (카메라 무관 공통 현상)
보드가 화면을 가득 채우는 초근접 구간에서 스케일 드리프트 — **HERO7과 Ace Pro 2
모두에서 동일 재현**되어 mono-inertial의 구조적 한계로 확인. 원거리 특징 부재 +
순수 평면 + 소진폭 고속 모션의 조합.

### 5. 미모델링 요소
롤링 셔터(캘리브·SLAM 모두), IMU-카메라 병진(lever arm, 자이로만으로는 관측 불가),
가속도계 스케일(6면 정적 캘리브레이션으로 개선 가능), 시간 오프셋의 클록 드리프트
(현재 상수 가정 — Kalibr류 연속시간 최적화가 다음 단계).

## TODO

- [ ] 분절된 아틀라스에서 전체 궤적 복원 (서브맵 IMU 스티칭)
- [ ] ORB-SLAM3 비결정적 세그폴트 근본 수정 (backtrace 확보됨: LOST 경로 스레드 경쟁)
- [ ] IMU-카메라 병진(lever arm) 캘리브레이션 / 가속도계 스케일 보정 / 롤링셔터 모델링
- [ ] 학습 특징점(SuperPoint+LightGlue 등) 프런트엔드 검토
- [ ] 다른 VIO 백엔드 비교: Basalt / OKVIS / Kimera-VIO / DPVO·DROID-SLAM(GPU)
- [ ] 994 Hz IMU 리샘플 안티에일리어싱 + 고레이트 직접 공급 실험
- [ ] Kalibr 포맷 내보내기 (타 백엔드 캘리브레이션 호환)
- [ ] 파이썬 패키지명 `gopro_vio` → `actioncam_vio` 개명

## 문서

- [관측 지연·지터 노이즈 모델](docs/observation_latency_model.md) — 실측 지연(카메라별)과
  VLA 학습용 노이즈 주입 구현 (sim2real)
- [도커 이미지 패치](docker/README.md) — ORB-SLAM3 수정 내역과 재현 방법

## 데이터·결과물

- **데이터셋**: [huggingface.co/datasets/chanwook/actioncam-vio](https://huggingface.co/datasets/chanwook/actioncam-vio)
  — 원본 영상 11편(10.8 GB) + SLAM 산출물 + 매니페스트. 다운로드:
  ```bash
  hf download chanwook/actioncam-vio --repo-type dataset --local-dir data
  ```
  받으면 `data/<모델>/…` 배치가 이 README의 명령과 그대로 호환됩니다.
- 파이프라인 산출물은 `output/<이름>/slam/`: `camera_trajectory.csv`(+`.allmaps`),
  `camera_trajectory.tum`, `map_atlas.osa`, `trajectory.png`, `map.rrd`
- 검증 결과 아티팩트(플롯/통계)는 각 `cameras/<모델>/results/`에 커밋되어 있음
