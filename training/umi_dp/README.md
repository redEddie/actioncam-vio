# 원본 UMI Diffusion Policy 학습 (비교 기준선)

real-stanford/universal_manipulation_interface의 DP를 canonical 데이터셋으로 학습해
SmolVLA와 동일 hold-out(ep 90–98)에서 비교. 결과: 8ep 3.09 → 12ep 2.95 → **60ep 2.86 cm**
(epoch ~35 수렴, train 0.55) — SmolVLA 최종 레시피 V6(2.92)와 동등.

## 구성
- `convert_gopro_zarr_to_umi.py` — canonical `02_zarr/replay_buffer.zarr`(UMI ReplayBuffer 스키마)
  → UMI가 요구하는 `dataset.zarr.zip` (+robot0_demo_start/end_pose 추가)
- UMI 저장소 쪽 변경은 해당 클론(`~/universal_manipulation_interface`)의
  **`theme/gopro-umi` 브랜치** 참조: task cfg `umi_gopro.yaml`(60→10 Hz 다운샘플 6,
  고정 hold-out `val_episodes`), UmiDataset val_episodes 옵션, resume `initial_lr` 수정.

## 실행 (conda env `umi_dp`: py3.10, torch 2.5.1, diffusers 0.18.2, timm 0.9.7, zarr 2.16)
```bash
python convert_gopro_zarr_to_umi.py --src .../02_zarr/replay_buffer.zarr --out data/gopro_umi.zarr.zip
cd ~/universal_manipulation_interface && git apply .../patches/universal_manipulation_interface.patch
python train.py --config-name=train_diffusion_unet_timm_umi_workspace task=umi_gopro \
  training.num_epochs=60 training.checkpoint_every=1 logging.mode=offline
```
평가는 `5_benchmarks/11_offline_trajectory_eval/predict_dp.py`.
주의: resume는 cosine 스케줄을 새로 만들므로 장기 학습은 처음부터 num_epochs를 크게.
