# UMI 서버 이관용 handoff

작성일: 2026-08-03
목적: 서버에서 Zarr 생성 후 별도 학습 코드를 실행하기 위한 입력·변경사항·검증 기준 전달

## 현재 결론

로컬에서 Hero13 SLAM/VIO 재처리, ArUco ID 13 world 정렬, TCP 좌표 변환,
캘리브레이션/좌표계 검증은 완료했다. 로컬에는 Zarr를 만들지 않았다.
서버에서 아래 빌더로 Zarr만 생성하면 된다.

## 절대 사용하지 않을 입력

- `world/trajectory_world.csv`: ArUco tag frame의 카메라 광학 중심 pose이다.
- `slam/camera_trajectory.csv`: SLAM 내부 카메라 pose이며 학습용 TCP pose가 아니다.
- `TEST/build_umi_zarr.py`의 과거 변환 로직: camera offset과 회전을 중복 적용하므로 사용하지 않는다.

## 서버에서 사용할 단일 pose 입력

각 episode의 다음 파일만 학습 pose의 원천으로 사용한다.

```
Episode_result/episode_N/world/trajectory_tcp_robot.csv
```

이 CSV는 이미 `robot_base -> tcp_link` pose이다. position은 robot base 좌표,
quaternion은 robot base에서 본 tcp_link 자세이다. 서버의 Zarr 빌더는 이 pose에
어떤 camera→TCP, tag→robot, URDF root→tcp 변환도 다시 적용하면 안 된다.

## 적용된 고정 변환 (참고용, 재적용 금지)

좌표계:

- Camera/OpenCV: +X 오른쪽, +Y 아래, +Z 앞
- TCP/tcp_link: +X 왼쪽, +Y 아래, -Z 앞
- ArUco tag: +X 오른쪽, +Y 위, +Z 법선/위
- Robot base: +X 앞, +Y 왼쪽, +Z 위

변환:

```
T_robot_tcp = T_robot_tag @ T_tag_camera @ T_camera_tcp
T_camera_tcp: t=[0, 0.09, 0.16] m, R=diag(-1,+1,-1)
T_robot_tag: t=[0.50, 0, 0] m, R=Rz(-90 deg)
```

camera→TCP 실측값은 사용자가 장착 상태에서 확인한 승인값이다. URDF의
root→tcp_link 14 cm를 추가 적용하지 않는다.

## 반드시 서버로 복사할 파일

가장 안전한 방법은 `1_data_pipeline`과 `actioncam-vio`를 그대로 복사하는 것이다.
최소 구성은 다음과 같다.

```
1_data_pipeline/build_umi_zarr.py
1_data_pipeline/requirements.txt
1_data_pipeline/actioncam-vio/configs/umi_coordinate_frames.yaml
1_data_pipeline/actioncam-vio/gopro_vio/tcp_robot_transform.py
1_data_pipeline/actioncam-vio/batch_reprocess.py
1_data_pipeline/actioncam-vio/Episode/*.MP4
1_data_pipeline/actioncam-vio/Episode_result/episode_mapping_log.json
1_data_pipeline/actioncam-vio/Episode_result/episode_*/world/trajectory_tcp_robot.csv
1_data_pipeline/actioncam-vio/Episode_result/episode_*/world/tcp_robot_validation.json
1_data_pipeline/actioncam-vio/Episode_result/episode_*/gripper/gripper_width.csv
```

`build_umi_zarr.py`의 기본 경로는 다음 구조를 가정한다.

```
<project>/1_data_pipeline/build_umi_zarr.py
<project>/1_data_pipeline/actioncam-vio/Episode/
<project>/1_data_pipeline/actioncam-vio/Episode_result/
<project>/2_dataset/
```

## 파일 무결성 확인

서버에서 코드 파일을 복사한 뒤 다음 SHA-256이 일치해야 한다.

```
configs/umi_coordinate_frames.yaml
  644fa39eb501e393cb8cc2574999991d53e460c685be0660e55273b5d863d9d9
gopro_vio/tcp_robot_transform.py
  dde6dd0b4048524121449a3f67aa1a1af0244ee6bb253c3e595ead08520dfc5c
batch_reprocess.py
  7b5069cce093a829d76e7d86a657e668d7c3a0eeb9f3006adb3846023e744ff9
build_umi_zarr.py
  4b9d7fecaa3e6067d5b96825e451783926488471e4a88862a8fac03ccdd89c57
```

## 서버 실행 순서

Zarr v2가 필요하다. Zarr v3는 `DirectoryStore`가 없어 이 빌더와 호환되지 않는다.

```bash
conda env update -n gopro-vio -f 1_data_pipeline/actioncam-vio/environment.yml
conda run -n gopro-vio python 1_data_pipeline/build_umi_zarr.py --dry-run
conda run -n gopro-vio python 1_data_pipeline/build_umi_zarr.py \
  -o 2_dataset/replay_buffer.zarr
```

dry-run은 56개 episode의 pose provenance, timestamp, gripper, 영상 경로를 확인한다.
`Validated 56 episode(s)`가 나와야 한다.

## 빌더가 만드는 Zarr schema

```
data/camera0_rgb                    (N,224,224,3) uint8
data/robot0_eef_pos                 (N,3) float32
data/robot0_eef_rot_axis_angle      (N,3) float32
data/robot0_gripper_width           (N,1) float32
meta/episode_ends                   (56,) int64
```

root attrs에는 다음 provenance가 기록된다.

```
pose_source = trajectory_tcp_robot.csv
pose_frame = robot_base_tcp_link_pose
coordinate_transforms_applied_by_builder = none
```

## 서버 GPT가 확인해야 할 체크리스트

1. `trajectory_tcp_robot.csv` 외 pose를 사용하지 않는가?
2. camera offset, tag 회전, URDF root→tcp offset을 중복 적용하지 않는가?
3. 56개 episode가 모두 `tcp_robot_validation.json: passed=true`인가?
4. Zarr의 각 배열 길이와 `episode_ends`가 일치하는가?
5. 학습 코드는 이 schema를 읽는가? 기존의 `image`, `camera_trajectory`,
   `gripper_width`라는 오래된 키를 그대로 사용하면 안 된다.
6. 학습 전 최소 1~2 episode, 1 epoch smoke test에서 forward/backward와 checkpoint
   저장이 통과하는가?

## 검증 기준값

57개 산출물, 총 28,479 pose 행을 독립 재합성한 결과:

- camera→TCP 위치 최대 오차: `1.61e-16 m`
- tag→robot 위치 최대 오차: `1.67e-16 m`
- 자세 합성 최대 오차: `9.39e-16 rad`
- manifest/calibration 불일치: `0/57`
- ORB-SLAM Hero13 KB4 설정 불일치: `0/57`
- ArUco tag 정렬 residual: median `0.192 cm`, max `0.995 cm`

## 알려진 기록 사항

IMU→camera 병진은 gyro-only 정렬에서 관측되지 않아 `[0,0,0] m`으로 기록되어
있다. 이는 현재 TCP 변환에 재적용하지 않는다. camera→TCP `[0,9,16] cm`은
사용자가 장착 상태에서 확인한 실측 승인값이다.
