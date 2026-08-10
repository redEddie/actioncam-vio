# umi_pipeline

우리만의 **Universal Manipulation Interface (UMI)** 데이터 수집 파이프라인.
GoPro 대신 **Intel RealSense D435i / D405**를 사용한다.

원본 UMI: GoPro(어안)+IMU → ORB-SLAM3로 그리퍼 6DoF 궤적 복원, ArUco로 월드 원점·
그리퍼 폭 측정 → Zarr replay buffer 데이터셋. 우리는 RealSense로 대체하고 최종적으로
**LeRobot 포맷(HuggingFace Hub)**으로 공유하는 것을 목표로 한다.

## 카메라
| | D435i | D405 |
|---|---|---|
| IMU | 있음 | **없음** |
| SLAM 모드 | Stereo-Inertial | Stereo / RGB-D (비관성) |

레코딩·SLAM은 `configs/cameras/*.yaml` 프로파일로 **장치 비종속(device-agnostic)** 하게 동작.

## 디렉토리
```
configs/cameras/   # d435i.yaml, d405.yaml  (스트림/해상도/SLAM모드 선언)
recording/         # record.py(.bag 녹화), bag_to_folder.py(추출)
slam/
  patches/         # ORB-SLAM3 우리 수정분(diff) + apply_patches.sh
  configs/         # ORB-SLAM3 yaml (Atlas 저장/로드 포함)
calibration/       # 카메라-IMU(OpenICC), IMU noise
aruco/             # 마커 생성, 원점/그리퍼 검출
dataset/           # 중간 에피소드 스키마 → LeRobot 변환기
data/              # .bag 등 대용량 (git 제외)
```

## ORB-SLAM3 재현
```bash
bash slam/patches/apply_patches.sh   # upstream 클론 + 패치 적용
bash ~/rs_slam/setup_2_build.sh      # 빌드
```

## 데이터 흐름 (raw → 최종)
```
.bag (RealSense 원본, 무손실)
  → [SLAM] 카메라 궤적 + [ArUco] world 변환 + [calib] TCP 외부파라미터
  → 중간 에피소드 스키마 (per-frame: rgb, T_world_tcp, gripper_width, action)
  → [exporter] LeRobot 데이터셋 → HuggingFace Hub
```
초기 개발에는 raw `.bag` + 중간 스키마로 반복하고, 에피소드 포즈가 검증된 뒤
(Phase 5) LeRobot exporter를 붙인다.
