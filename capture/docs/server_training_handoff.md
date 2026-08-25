# UMI 서버 최종 상태 및 학습 handoff

작성일: 2026-08-03
목적: Hero13 SLAM/VIO 결과부터 Zarr, LeRobot 7D dataset, SmolVLA 학습까지의 최종 상태를 한 문서로 전달한다.

## 현재 상태

- 원본 영상은 유지했다: `Episode/map.MP4` + 학습 episode MP4 56개.
- `map_video.MP4`는 구형/별도 파일이라 삭제했다.
- 구형 `Episode_result/`, `result/`, Zarr, LeRobot dataset, 학습 checkpoint는 삭제 후 최종 결과로 교체했다.
- Hero13 SLAM/VIO 및 TCP 변환 결과 56개 episode를 수신·검증했다.
- Zarr 및 LeRobot 7D dataset 생성을 완료했다.
- SmolVLA 7D 학습은 20,000 step 설정으로 시작했고 step 200까지 forward/backward/optimizer update를 통과했다.

## 원본 및 SLAM provenance

학습 pose의 유일한 원천은 각 episode의 아래 파일이다.

```text
1_data_pipeline/actioncam-vio/Episode_result/episode_N/world/trajectory_tcp_robot.csv
```

이는 이미 `robot_base -> tcp_link` pose이다. Zarr 빌더에서 camera offset, tag→robot 회전, URDF root→tcp offset을 다시 적용하면 안 된다.

사용 금지 pose 입력:

- `world/trajectory_world.csv`: ArUco tag frame의 camera optical-center pose
- `slam/camera_trajectory.csv`: SLAM 내부 camera pose
- `TEST/build_umi_zarr.py`: 과거 중복 변환 로직

고정 변환(참고용, 재적용 금지):

```text
T_robot_tcp = T_robot_tag @ T_tag_camera @ T_camera_tcp
T_camera_tcp: t=[0, 0.09, 0.16] m, R=diag(-1,+1,-1)
T_robot_tag: t=[0.50, 0, 0] m, R=Rz(-90 deg)
```

## 입력 데이터 검증

- 학습 episode: 56개
- `trajectory_tcp_robot.csv`: 56개
- `tcp_robot_validation.json`: 56개, `passed=false` 0개
- 원본 MP4 FPS: 모든 episode가 59.94006 FPS
- `width_norm` 유효 범위: `0.0` ~ `0.5366375`

`map.MP4`는 현재 atlas 생성의 원본이므로 반드시 유지한다. `.THM` 파일은 파이프라인에 필요 없다.

## Zarr 생성 결과

생성 파일:

```text
2_dataset/replay_buffer.zarr/
```

실제 검증값:

```text
frames: 55,751
episodes: 56
data/camera0_rgb:                  (55751, 224, 224, 3) uint8
data/robot0_eef_pos:               (55751, 3) float32
data/robot0_eef_rot_axis_angle:    (55751, 3) float32
data/robot0_gripper_width:         (55751, 1) float32
meta/episode_ends:                 (56,) int64
pose_source: trajectory_tcp_robot.csv
pose_frame: robot_base_tcp_link_pose
coordinate_transforms_applied_by_builder: none
```

각 배열 길이는 55,751이고 마지막 `episode_ends` 값도 55,751이다. gripper는 이미 `width_norm`이므로 Zarr 이후 다시 미터 단위 정규화를 적용하면 안 된다.

## Zarr → LeRobot 변환 변경사항

변환 스크립트:

```text
3_training/convert_zarr_to_lerobot.py
```

수정·검증된 사항:

- `width_norm`을 그대로 보존한다. 이전의 m→[0,1] 재정규화는 제거했다.
- source 영상 59.94006 FPS에 맞춰 LeRobot metadata FPS를 60으로 기록한다.
- Zarr provenance와 배열 길이, 56 episode, 7D schema를 변환 전에 검증한다.
- metadata robot type은 `umi_tcp`이다.

LeRobot 출력:

```text
2_dataset/lerobot_dataset/
episodes: 56
frames: 55,751
fps: 60
observation.images.top: (224, 224, 3) uint8 image
observation.state:      (7,) float32
action:                 (7,) float32
```

7D 순서:

```text
[x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper_norm]
```

`action`은 다음 frame의 absolute 7D state이며, episode 마지막 frame만 자기 자신의 state를 action으로 사용한다.

## SmolVLA 학습 변경사항

학습 진입점:

```text
3_training/train_smolvla.py
```

수정 내용:

- LeRobot root는 `2_dataset/lerobot_dataset/` 자체를 사용하도록 수정했다.
- 학습 전 `meta/info.json`에서 56 episode, nonzero frame, 7D state/action schema를 검사한다.
- output root는 `3_training/smolvla_run/`이며 checkpoint는 그 아래 `checkpoints/`에 생성된다.
- Hugging Face Hub push는 `false`다.
- pretrained `lerobot/smolvla_base`의 기본 ALOHA 6D state/action을 전체 feature dictionary override로 7D UMI schema에 맞춘다.

`smolvla_lora.yaml`은 현재 실제 CLI config가 아니라 프로젝트 reference 문서다. 구형 `camera_trajectory`, `gripper_width` key는 제거했다.

## 최종 학습 명령

`gopro_env` 활성화 상태에서 실행한다.

```bash
PYTHONUNBUFFERED=1 lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=gopro_umi/smolvla_finetuned \
  --policy.push_to_hub=false \
  --policy.input_features='{"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]},"observation.state":{"type":"STATE","shape":[7]}}' \
  --policy.output_features='{"action":{"type":"ACTION","shape":[7]}}' \
  --dataset.repo_id=gopro_umi/smolvla_dataset \
  --dataset.root=/home/kimminje/gopro_umi/2_dataset/lerobot_dataset \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --batch_size=32 \
  --steps=20000 \
  --output_dir=/home/kimminje/gopro_umi/3_training/smolvla_run
```

정상 시작 기준:

```text
observation.state: shape [7]
output_features.action: shape [7]
dataset.num_frames=55751
dataset.num_episodes=56
Start offline training on a fixed dataset
```

확인된 첫 학습 결과:

```text
step:200 smpl:6K ep:6 epch:0.11 loss:0.170 grdn:1.181 lr:1.5e-05 updt_s:0.395 data_s:0.152
```

이는 7D 입력으로 forward/backward/optimizer update가 정상 수행됐음을 의미한다.

## 남은 확인 항목

1. step 20,000 완료 여부 확인
2. `3_training/smolvla_run/checkpoints/020000/` 생성 확인
3. 최종 checkpoint로 1~2 episode inference/deployment smoke test 수행
