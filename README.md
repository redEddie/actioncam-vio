# GoPro-UMI

GoPro 시연 영상 → VIO 궤적 복원 → 데이터셋 → 정책 학습 → SO-101 실기 배포.
**코드(파이프라인 순서)와 산출물(영상·데이터셋·모델)을 분리**한 구조이며, 폴더 순서가 곧 파이프라인 순서다.

```
capture/     1. 수집·VIO — GoPro 영상에서 RGB+IMU 추출, ORB-SLAM3, 월드 정렬, TCP 궤적 (구 1_capture/actioncam-vio)
dataset/     2. 데이터셋 — zarr 빌드 → yaw-free → 10 Hz LeRobot 변환·검증 (구 2_dataset, 파일 평탄화)
training/    3. 학습 — SmolVLA 정식 런처 + variants/(안블레이션 V0–V7·최종 레시피) + umi_dp/(DP 기준선, 외부 패치 동봉)
deploy/      4. 배포 — run_live, inference/(관측·원격추론·SE(3) 계약), trajectory/, control/, ik/, config/, tests/
urdf/        로봇 모델 — SO-101+GoPro URDF, meshes/, assets/ (FK/IK·MuJoCo가 공용 참조)
eval/        오프라인 평가 — hold-out 궤적 지표(RMSE/DTW/DCT/FAST+), 예측 러너, 학습 곡선
benchmarks/  실기·시스템 벤치마크 01–10 (지연, 30 Hz 예산, IK 정확도, 폐루프 추종 …)
tools/       유틸 — calibration/ camera/ motor/ teleop/ visualization/(MuJoCo 리플레이·URDF 검증) diagnostics/ + 전체 파이프라인 오케스트레이터
docs/        문서 — pipeline.md(전체 흐름), 아키텍처·계약·실험 기록
artifacts/   산출물 자리 — git에는 manifest/config JSON만. 실제 영상·zarr·모델은 로컬 디스크 (아래 참조)
```

## 산출물(artifacts) 규칙
대용량 데이터는 git 밖, 로컬 `artifacts/`(구 `7_storage/`)에 둔다:
`datasets/<ID>/{00_source,01_orbslam3,02_zarr,03_lerobot,04_training,05_logs}`, `run/`(production 모델),
`run_candidates/`(승격 대기 — 현재 `v6_se3relwp_11k`), `Delta_Weights/`.
git에는 각 모델의 config/manifest JSON만 추적한다. 기존 로컬 트리(`~/GoPro_Umi/gopro_umi/7_storage`)를
쓰는 실행 스크립트는 해당 경로를 그대로 참조한다(각 README에 명시).

## 빠른 실행
```bash
python -m pytest                                  # 소프트웨어 테스트 (하드웨어 테스트는 deploy/tests/hardware)
python tools/run_full_dataset_pipeline.py --help  # 영상→모델 전체 오케스트레이션
python deploy/run_live.py --help                  # 실기 (config: deploy/config/deployment.yaml,
                                                  #        확정 후보: deployment_v6_candidate.yaml)
```

## 현재 상태 (2026-08-22)
hold-out(ep 90–98) 1.5 s 궤적 RMSE: 기존 계약 3.28 cm → **최종 레시피 V6(SE(3) 상대 웨이포인트+강한 증강) 2.92 cm**,
UMI-DP 60ep 2.86 cm. 최종 구성 = V6 + 런타임 관측 정렬(`deploy/config/deployment_v6_candidate.yaml`), 실기 검증 대기.
상세: `training/variants/RESULTS.md`, `tools/visualization/URDF_VALIDATION.md`, `docs/`.
