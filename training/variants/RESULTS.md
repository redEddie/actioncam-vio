# SmolVLA 안블레이션 및 최종 레시피 (2026-08-19 ~ 08-21)

공통: Delta_Weights 시작, batch 32, chunk 15, 학습 ep 0–89 / **hold-out ep 90–98**,
평가 = 1.5 s 궤적 TCP 위치 RMSE (오프라인, `5_benchmarks/11_offline_trajectory_eval`).

| 변형 | 변경 | hold-out best | 판정 |
|---|---|---|---|
| V0 baseline | 절대 state 6D, 1프레임 (기존 계약, 20k) | 3.28 cm (3k부터 정체, train 0.66) | 절대좌표 암기 |
| V1 relvel | state XYZ→이전 스텝 대비 변위 | 3.03 @5k | ✅ 관측 표현 레버 |
| V2 +prev image | V1+이전 프레임 camera2 | 3.01 @5k | △ +0.02, 기각 가능 |
| V3 +strong aug | V1+강한 증강 | 2.92 @6k (하강 중) | ✅ |
| V4 +freeze vision | V1+인코더 동결 | 3.26 | ❌ fisheye는 적응 필요 |
| V5 rel-waypoint | V1+상대 웨이포인트 액션 | 3.04 @6k | ✅ DTW/DCT/grip 최고 |
| **V6 = 최종 레시피** | SE(3) 상대 state/액션 + 강한 증강 (20k) | **2.92 @11k** | UMI-DP 60ep(2.86)와 동등 |
| V7 lat-aug | V6 + 지연 88±14ms 이미지 시프트 | 2.97 @12k | 정렬 불가 배포용 예비 |

**최종 구성**: V6 모델 + 런타임 UMI식 관측 정렬(`4_deploy/config/deployment_v6_candidate.yaml`).
V7은 정렬과 병용 금지(이중 보정). 참고: UMI-DP 원본 60 epoch 2.86 cm — 두 모델 모두
이 데이터셋 바닥(~2.9 cm) 도달 → 다음 병목은 데이터 양·다양성.

## 파일
- `make_variant_dataset.py` — canonical 10Hz 데이터셋에서 relvel/dropxyz/prev-image/relwp 변형 생성
- `make_se3_dataset.py` — base zarr의 완전 회전으로 SE(3) 10D state/action 데이터셋 (+`--latency-aug`)
- `relwp_step.py` / `relwp_se3_step.py` — lerobot 전처리 파이프라인에 삽입되는 상대 웨이포인트 변환 step
- `train_relwp.py` / `train_se3.py` / `train_strongaug.py` — lerobot_train 래퍼 (step 삽입 + 강한 증강)
- `run_variant.sh` / `eval_variant.sh` / `queues/` — 실행·체크포인트별 hold-out 평가 자동화
- 경로 가정: `$HOME/GoPro_Umi/gopro_umi` (프로젝트), `$HOME/work/eval` (평가), `$HOME/work/smolvla` (출력)
