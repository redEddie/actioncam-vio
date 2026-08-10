# 🤖 로봇 학습 및 추론(Inference) 파이프라인 종합 가이드
이 문서는 **(A)절대 좌표 기반**과 **(B)상대 좌표 기반** 두 가지 버전에 대한 학습 파이프라인 및 로컬 컴퓨터에서의 로봇 제어(추론) 방법을 정리한 통합 문서입니다.

---

## [버전 A] 절대 좌표 (Absolute) 기반 파이프라인

전통적인 방식으로 공간상의 '정확한 위치'를 예측하도록 학습합니다.

### 1. 특징
- **데이터셋**: `lerobot_dataset_yawfree` (모델이 목표물의 절대 위치 X, Y, Z를 예측)
- **학습 스크립트**: `train_absolute.sh`
- **청크 사이즈**: 50 프레임
- **비전 인코더**: 해동됨 (Full Fine-tuning)

### 2. 로컬 컴퓨터(Inference) 로직
1. 카메라 이미지와 현재 로봇의 절대 위치(State)를 모델에 입력합니다.
2. 모델은 50개의 **"미래 절대 좌표(Absolute)"** 궤적을 예측합니다.
3. **시간 앙상블**: 과거 예측 궤적들과 현재 궤적을 겹쳐서 '평균 절대 좌표'를 도출합니다.
4. **역기구학 제어**: 이 평균 절대 좌표를 그대로 역기구학(IK) 솔버에 넣고 로봇을 구동합니다.

---

## [버전 B] 상대 좌표 (Delta Action) 기반 파이프라인 ⭐(추천)

0.1mm 단위의 미세 조향 부족(Overshooting) 현상과 오차 누적(Drift)을 방지하기 위해 특수 설계된 파이프라인입니다.

### 1. 특징
- **데이터셋**: `lerobot_dataset_delta` (모델이 이전 프레임 대비 미세한 '변화량 dx, dy, dz'만 예측)
- **학습 스크립트**: `train_delta.sh`
- **오차 복구 강화**: 이미지 변형(`image_transforms.enable=true`)을 통해 시각적 노이즈가 발생해도 궤도를 잃지 않고 복구(Recovery)하는 능력을 훈련합니다.
- **청크 사이즈**: 60 프레임 (로컬 컴퓨터의 6프레임 제어 주기에 10배수로 딱 맞아떨어지게 설계)

### 2. 로컬 컴퓨터(Inference) 로직 🚨 (버전 A와 다름)
1. 카메라 이미지와 현재 로봇의 절대 위치(State)를 모델에 입력합니다.
2. 모델은 60개의 **"미래 변화량(Delta dx, dy, dz)"** 궤적을 예측합니다.
3. **시간 앙상블**: 과거 예측한 변화량들과 현재 예측한 변화량을 겹쳐서 '평균 Delta'를 도출합니다.
4. **절대 좌표 복원**: 도출된 평균 Delta 값을 현재 로봇의 절대 위치에 **더해서** 최종 타겟 좌표를 만듭니다.
   👉 `최종 Target (X, Y, Z) = 현재 State (X, Y, Z) + 앙상블된 Delta (dx, dy, dz)`
5. **역기구학 제어**: 계산된 Target 절대 좌표를 역기구학 솔버에 넣고 로봇을 구동합니다.

---

## 🚀 파일 전송 가이드 (서버 ➡️ 로컬 데스크탑)

서버에서 학습이 완료되면, 로컬 데스크탑 터미널을 열고 아래 명령어를 입력하여 가중치와 정리 문서를 가져오십시오.

### 1. [버전 A] 절대 좌표 가중치 및 스크립트 복사
```bash
# 로컬에 폴더 생성
mkdir -p ~/Desktop/project/gopro_umi/yawfree_server_candidate_absolute

# 가중치 및 스크립트 복사 (scp -P 17970 사용)
scp -P 17970 -r kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/smolvla_yawfree_run_fullft/checkpoints/020000/pretrained_model/* ~/Desktop/project/gopro_umi/yawfree_server_candidate_absolute/
scp -P 17970 kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/train_absolute.sh ~/Desktop/project/gopro_umi/yawfree_server_candidate_absolute/
scp -P 17970 kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/pipeline_guide_comprehensive.md ~/Desktop/project/gopro_umi/yawfree_server_candidate_absolute/
```

### 2. [버전 B] 상대 좌표 가중치 및 스크립트 복사
```bash
# 로컬에 폴더 생성
mkdir -p ~/Desktop/project/gopro_umi/yawfree_server_candidate_delta

# 가중치 및 스크립트 복사
scp -P 17970 -r kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/smolvla_delta_run_fullft/checkpoints/020000/pretrained_model/* ~/Desktop/project/gopro_umi/yawfree_server_candidate_delta/
scp -P 17970 kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/train_delta.sh ~/Desktop/project/gopro_umi/yawfree_server_candidate_delta/
scp -P 17970 kimminje@155.230.189.77:/home/kimminje/gopro_umi/3_training/pipeline_guide_comprehensive.md ~/Desktop/project/gopro_umi/yawfree_server_candidate_delta/
```
