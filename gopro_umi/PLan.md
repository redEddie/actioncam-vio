# GoPro + UMI 데이터 파이프라인 및 SmolVLA 학습 마스터 플랜 & 서버 인수인계서 (Plan)

본 문서는 `actioncam-vio` 데이터 추출, Zarr 데이터셋 빌드, 그리고 서버 GPU 환경에서의 **SmolVLA (BC + LoRA)** 모델 학습을 위한 프로젝트 통합 규격서이자 **서버 Gemini 에이전트 인수인계 문서(Server Handover Specification)**입니다.

---

## 📌 주요 원칙 및 선배님 지침
1. **ArUco 마커 단계적 적용**:
   - **1단계 (현행)**: ArUco 마커 부착 데이터로 Zarr 패키징 및 SmolVLA 학습 검증 완료.
   - **2단계 (확장)**: 제어 성공 시 Zero-shot (Markerless) 영상 재촬영 및 파동 테스트 확장.
2. **독립 에피소드 관리**: 모든 비디오 및 데이터는 56개 에피소드 단위로 엄격 분리 관리.
3. **개발 수칙 준수**: 메인 개발 디렉토리는 청결하게 유지하며, 모든 임시 패치/실험 파일은 반드시 `TEST/` 디렉토리에 격리 작성.

---

## 📂 프로젝트 표준 디렉토리 구조 (Sitemap)

```text
project/gopro_umi/                      (메인 프로젝트 루트)
├── PLan.md                             (프로젝트 마스터 플랜 & 서버 인수인계서)
│
├── 1_data_pipeline/                    [단계 1: 원시 데이터 추출, 동기화 & Zarr 빌드]
│   ├── actioncam-vio/                  (GoPro VIO SLAM & 궤적 추출)
│   ├── build_umi_zarr.py              (56개 에피소드 -> replay_buffer.zarr 생성 빌더)
│   ├── gripper_width.py                (그리퍼 너비 계산 - w_min: 0.0482m, w_max: 0.1306m)
│   └── get_min_max.py                 (Min/Max 궤적 정규화 도구)
│
├── 2_dataset/                          [단계 2: Zarr 데이터셋 보관소]
│   ├── replay_buffer.zarr/            (생성된 UMI/SmolVLA 학습 데이터셋)
│   ├── TEST_Episode_v2.py              (에피소드 센서값 보간 & 동기화 검증)
│   └── umi_official/                   (UMI 공식 라이브러리 및 유틸리티)
│
├── 3_training/                         [단계 3: 서버 SmolVLA 학습 전용] 🚀 (서버 전송 대상)
│   ├── train_smolvla.py               (SmolVLA Behavioral Cloning + LoRA 학습 메인)
│   ├── configs/                        (Action Chunking=10, Frame Skip=10 설정 파일)
│   ├── models/                         (SmolVLA 모델 아키텍처 및 LoRA 레이어 정의)
│   └── checkpoints/                    (서버 학습 체크포인트 및 로그 저장소)
│
├── 4_deploy/                           [단계 4: 로봇 실기 배포 & 제어]
│   ├── custom_ik_teleop.py            (Yaw-Unconstrained Pitch/Roll Preserved IK 반영 SO-100/101 직진성 보정 제어기)
│   ├── URDF/                           (로봇 arm 3D 모델 및 Kinematic 오프셋)
│   └── Teleop/                         (원격 제어 모듈)
│
└── TEST/                               [격리 공간: 하드웨어 진단 & 임시 패치]
    ├── diag_motor.py                   (로봇 모터 진단 15개 스크립트)
    └── scan_all_motors.py ...
```

---

## 🤖 서버 Gemini 에이전트 작업 및 인수인계 지침 (Server Instructions)

> **서버 담당 Gemini 에이전트는 로컬 에이전트와 동일하게 아래 지침을 엄격히 준수하여 작업을 진행하십시오.**

### 1. 작업 및 환경 전제조건
- **서버 접속 정보**: `ssh -p 17970 kimminje@155.230.189.77` (`kimminje-T-Series`)
- **서버 프로젝트 루트**: `/home/kimminje/gopro_umi`
- **필수 의존성**: Python 3.10+, PyTorch (CUDA 사용 가능), `zarr`, `numpy`, `opencv-python`, `torchvision` (루트의 `requirements.txt` 이용)
- **임시 패치 파일 금지 규칙**: 서버 작업 중 발생하는 일회성 테스트 스크립트, 패치 파일, 실험용 파이썬 코드 등은 **절대 루트나 `3_training/` 메인 폴더에 다이렉트로 만들지 말고 `TEST/` 디렉토리에 생성하여 격리 실행**할 것.

### 2. 순차적 실행 단계 (Execution Pipeline)

#### Step 1: Zarr 데이터셋 생성 및 검증
```bash
# 1_data_pipeline의 build_umi_zarr.py를 실행하여 56개 에피소드를 Zarr 포맷으로 패키징
cd /home/kimminje/Desktop/project/gopro_umi
python 1_data_pipeline/build_umi_zarr.py
```
- **출력 확인**: `2_dataset/replay_buffer.zarr` 파일이 생성되었는지 확인.

#### Step 2: SmolVLA (BC + LoRA) 학습 실행
```bash
# 3_training 디렉토리의 학습 메인 스크립트 실행
python 3_training/train_smolvla.py --config 3_training/configs/smolvla_lora.yaml
```
- **핵심 하이퍼파라미터 세팅**:
  - **Model Architecture**: SmolVLA (Vision-Language-Action)
  - **Training Strategy**: Behavioral Cloning (BC) + LoRA (Low-Rank Adaptation)
  - **Action Chunk Size**: `10` (미래 10스텝 액션 일괄 예측)
  - **Frame Skip**: `10` (매 10스텝 중 1번째 관측값 기반 추론)
  - **Saved Checkpoints**: `3_training/checkpoints/`에 `.pt` 파일로 주기적 저장.

---

## 📦 서버 가상환경 설치 검증 패키지 및 버전 목록 (Installed Packages & Versions)

> **서버에서 가상환경 설치 시 아래 버전 목록 및 `requirements.txt`를 통해 동일한 패키지 환경을 재현하십시오.**

```text
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.38.0
accelerate>=0.27.0
peft>=0.9.0
zarr>=2.16.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scipy>=1.11.0
opencv-python>=4.8.0
pillow>=10.0.0
matplotlib>=3.8.0
rerun-sdk>=0.16.0
datasets>=2.19.0
huggingface-hub>=0.20.0
PyYAML>=6.0
tqdm>=4.66.0
```

---

## 🏁 다음 진행 단계
- [x] 디렉토리 파이프라인 4단계 구조화 완료 (`1_data_pipeline`, `2_dataset`, `3_training`, `4_deploy`, `TEST`)
- [x] 로컬 가상환경(`test_env`) 패키지 및 버전 `PLan.md` & `requirements.txt`에 기록 완결
- [x] 서버 Gemini 에이전트용 인수인계 및 실행 표준서 작성 완료
- [ ] 서버에 `project/gopro_umi` 폴더 동기화 및 `python 1_data_pipeline/build_umi_zarr.py` 실행 검증

