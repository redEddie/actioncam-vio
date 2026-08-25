# Benchmark 11 — Offline Trajectory Evaluation (hold-out)

정책이 hold-out 에피소드(90–98)의 앵커 관측에서 예측한 1.5 s(15 step) 궤적을
GT(canonical 10 Hz LeRobot 상태)와 같은 yaw-free 6D 공간에서 비교한다.
손실값 대신 이걸 학습 건강 지표로 쓴다 (train/hold-out 격차로 과적합 판정).

## 지표 (`eval_common.chunk_metrics`)
- 궤적 위치 RMSE / 최종점 오차 (cm), roll·pitch RMSE, gripper RMSE, 지평선별 오차 곡선
- DTW(정규화 6D), 저주파 DCT 코사인(k=3/5) — 시간축 변화 압축 유사도
- FAST+ 토크나이저(physical-intelligence/fast): pre-BPE 양자화 DCT 일치율 권장
  (BPE 토큰 편집거리는 변별력 낮음 — 유사도 지표로 쓰지 말 것)
- 베이스라인: 정지 / 평균 액션 / 등속. 정지 = 8.25 cm.

## 사용
1. `predict_smolvla.py --ckpt <dir> --out preds/x.npz [--state-mode relvel] [--action-mode se3relwp]`
   (후배 remote 워커와 동일 전·후처리) / `predict_dp.py --ckpt <ckpt>` (UMI DP, relative repr 복원)
2. `compute_metrics.py --pred name=preds/x.npz ... --out results/` — 표·그림
3. `make_curves.py` — 체크포인트별 학습 곡선. `finalize.sh <step>` — 최종 리포트 조립.

경로 가정: 프로젝트 `$HOME/GoPro_Umi/gopro_umi`, 예측 출력 `preds/`.
환경: predict_smolvla→lerobot env, predict_dp/metrics→umi_dp env (transformers 4.48 for FAST).
