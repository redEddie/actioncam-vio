전체 파이프라인
     1.에피소드 수집 : RGB데이터 +IMU 데이터 수집
     2.ORB Slam3 구동: RGB데이터 + IMU데이터를 6DOF 궤적으로 복원
     3.YAW FREE 변환: 6DOF 궤적에서  YAW를 제외한 5DOF 궤적으로 변환
     4. Source Zarr 생성 :RGB + TCP trajectory + Gripper + episode 형태로 Zarr에 저장
     5.10HZ 변환: RGB 데이터 + 궤적 을 0.1초간격으로 저장 [10Hz]
     6.Action 추출: S[1]-S[0] 형태로 t마다의 A[t] 추출
     7.lerobot 변환: 학습용으로 lerobot 형태 데이터 변환  RGB + observation.state [6] + action [6] + timestamp + episode data
     8. Chunk 생성: SmolVLA 학습 시현재 시점부터 미래의 action [6]을 15개 읽어서 [A[t], A[t+1], ... , A[t+14]] 형태의 [15,6] action window 생성 -> 10Hz × 15step = 1.5초 horizon
     9.검증 + 학습
    
    추론 파이프라인
    1.smolvla 출력 : chunking step 15, Delta 형태로 데이터 출력 즉 shape = [15,6] 형태로 변화량 출력
    2.Smolvla는 lerobot 변환 값 학습  -> 우리 형태인 m , radian등으로 변환하기 위해 lerobot postprocessser 사용
    3.TCP 기준 재정립: 목표 시작 pose =! 실제 시작 pose 즉 여러 오차로 인해 목표 포즈에서 시작지 못해 초기 tcp 위치가 다름 현재 joint상태를 fk 시켜 실제 tcp 위치로 좌표 재정립
    4.Action 생성: 누적식 생성을 통해 액션 생성 즉 S0-> S0+A0 ->S0+A0+A1 ---
    5.추론 Latency 보정: 추론 시작 시각 observation =! 추론 완료 시각 observation  -> Action은 유지 즉 tcp의 위치 이동량은 보존하며 기준위치만 observation 완료시 tcp 위치로 재정립
    6.10hz 제어: chunking 된 Action을 10hz제어로 0.1초마다 행동 입력
    7.30hz 제어 변환: 오차 보정용 하위 모터 제어기는 10hz가 아닌 30hz로 동작 즉 액션입력은 10hz주기로 모터 입력은 30hz주기로 입력되며 모터터 입력은 액션 간격사이를 보간 하는 형식으로 동작
    8.ik 변환: 6DOF를 joint 각도 형태로 변환  iK를 통해 q_nom=[shoulder_pan , shoulder_lift , elbow_flex, wrist_flex , wrist_roll] 형태로 변형
    9.오차 보정: 중력 + 마찰 + 엔코더 오차등으로 인해 목표 joint 위치 =! 실제 joint 위치 즉 이를 보정하기 위한 작업 필요
                1.q_actual:실제 joint 각
                2.q_nom   :목표 joint 각   -> q_actual =! q_nom 이므로 모터에 적절한 q_cmd를 입력해줘 q_actual == q_nom을 입력하도록 해줘야함
                3.q_cmd   :입력 joint 각
    
                1.q_actual0:실제 시작 joint 각            Del_q_nom = q_nom[t] - q_nom[0] (목표각도 변화량)
                2.q_nom0   :목표 시작 joint 각   ->  Del_q_actual = q_actual[t] - q_actual[0] (실제 각도 변화량)
                3.q_cmd0   :입력 시작 joint 각            실제 각도 변화량 =! 목표 각도량 -> E_delta = Del_q_nom - Del_q_actual 오차 보정을 위한 보상은 두개가 존재
                
                1.K_ext   : E_delat X K_ext를 q_cmd에 더해줘 오차 보정
                2.중력보상 : 관절 위치에 따른 모터 무게 부하를 계산해 그보다 더 높은 joint각도를 주도록 연산
                --> q_cmd = q_cmd_hold + Del_q_nom + 중력보상 변화량 + K_ext 보정  (무게오차는 중력 보상으로 나머지 오차는 K_Ext로 보정하는 형태)
                [최종 q_cmd = 시작 유지 명령+ 원하는 이동량 + 중력보상 변화량 + K_ext 추종오차 보정]
    10.영점 보정: URDF상 원점 =! 실제 모터 원점  -> 원점 보정 필요
              실제 로봇을 사람손으로 0도로 맞추고 raw값 측정 -> Encoder 가 읽은 Raw값 = Offset값             [우리기준 -8.6154,4.0440,2.0220,1.0549,1.3626]
              실제 로봇 입력 모터값은 1~9단계에서 계산한 URDF ik 값 + Offset으로 진행  --> 최종 Raw값 입력
    
    11.Feetech 변환:최종 Raw값을 Motor Bus를 통해 Motor에 입력
    
    
    
    Action Chunking
    1.조건: 10hz 제어 / 1.5s chunk -> 15step chunksize
    2.overlap 과정 : 15step 중 6step을 overlap 구간으로 지정 15step중 초기 4step은 추론 latency 를 고려해 이전 chunk 궤적 입력추론 
                     latency 이후는 앙상블 기법 사용 -> 2/3 , 1/3 형식으로 overlap 구간의 두 궤적을 합침       결과적으로 궤적이 멈추는 구간이 없도록 형성




















# Canonical GoPro-UMI 파이프라인

여러 관련 파일은 `///`로 구분한다.

## 학습 데이터 및 학습 파이프라인

1. 에피소드 수집
   - GoPro 영상에 RGB와 IMU/GPMF 데이터를 함께 기록한다. 이후 영상에서 RGB 프레임과 IMU 데이터를 추출하고 timestamp를 정렬한다.
   - 관련 파일: `1_capture/actioncam-vio/run_umi_pipeline.sh` /// `1_capture/actioncam-vio/gopro_vio/extract.py` /// `1_capture/actioncam-vio/gopro_vio/gpmf.py`

2. ORB-SLAM3/VIO 및 좌표계 변환
   - RGB와 IMU로 6DOF camera trajectory를 복원한다. 이어서 world alignment를 수행하고 최종 pose를 `robot_base -> tcp_link` 좌표계로 변환한다.
   - 관련 파일: `1_capture/actioncam-vio/gopro_vio/slam.py` /// `1_capture/actioncam-vio/gopro_vio/world_align.py` /// `1_capture/actioncam-vio/gopro_vio/tcp_robot_transform.py`

3. Yaw-free source Zarr 생성
   - RGB, TCP position, rotation, normalized gripper, episode boundary를 full-rate source Zarr로 만든다.
   - rotation vector를 회전행렬과 Euler ZYX로 변환한 뒤 Yaw를 제외하고 `[X,Y,Z,Roll,Pitch,Gripper]` 6D state를 만든다. arm pose만 보면 5DOF이고 gripper를 합치면 최종 state는 6D다.
   - 현재 보존된 yaw-free builder는 56-episode historical candidate이다. 최종 77-episode yaw-free Zarr를 생성했던 정확한 historical source는 아직 복구되지 않았으므로 이 과거 builder를 canonical 실행 파일로 사용하면 안 된다.
   - 관련 파일: `2_dataset/conversion/build_umi_zarr.py` /// `docs/CHECKPOINTS.md`
   - 최종 historical yaw-free builder와 source Zarr는 현재 source-only Git 업로드에 포함하지 않는 로컬 provenance다.

4. 10 Hz sampling
   - episode별 source 시간을 `source_index / (60000/1001)`로 계산한다.
   - target time `0.0, 0.1, 0.2, ...`마다 가장 가까운 source frame을 deterministic half-up 규칙으로 선택한다.
   - 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `2_dataset/validation/test_convert_to_lerobot_10hz_incremental.py`

5. Step-to-step incremental action 추출
   - 각 episode 내부에서만 `A[t] = S[t+1] - S[t]`를 계산한다.
   - action 순서는 `[dX,dY,dZ,dRoll,dPitch,dGripper]`이고 Yaw는 포함하지 않는다.
   - `A[1]=S[2]-S[0]` 형태의 anchor-relative action이나 absolute target을 사용하지 않는다. episode 마지막 row는 6D zero terminal action을 사용한다.
   - 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `3_training/scripts/incremental_action_contract.py` /// `3_training/scripts/tests/test_incremental_action_contract.py`

6. LeRobot dataset 직렬화
   - 각 LeRobot row에는 224×224 RGB, `observation.state [6]`, `action [6]`, timestamp와 episode metadata를 저장한다.
   - 각 row에 `[15,6]` action chunk를 저장하지 않는다.
   - 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `2_dataset/validation/test_convert_to_lerobot_10hz_incremental.py` /// `docs/CHECKPOINTS.md`

7. 15-step action window 구성
   - LeRobot training sampler가 row-wise `action [6]`에서 미래 15개 action을 읽어 `[15,6]` window를 만든다.
   - `chunk_size=15`, `n_action_steps=15`, action rate 10 Hz이므로 horizon은 1.5초다.
   - window의 의미는 `[S[t+1]-S[t], S[t+2]-S[t+1], ..., S[t+15]-S[t+14]]`이다.
   - 관련 파일: `3_training/scripts/train_delta.py` /// `3_training/scripts/train_delta.sh` /// `Delta_Weights/config.json`

8. 학습 전 fail-closed 검증
   - dataset fps, episode/frame 수, 6D action schema, step-to-step contract, chunk 15, source checkpoint와 output overwrite 조건을 검사한다.
   - 관련 파일: `3_training/scripts/validate_incremental_training_preflight.py` /// `3_training/scripts/incremental_action_contract.py` /// `3_training/scripts/README.md`

9. SmolVLA 학습
   - final recovered launcher는 10 Hz dataset, chunk 15, `n_action_steps=15`, 6D incremental action, batch size 32, 20,000 steps를 사용한다.
   - dataset의 `observation.images.top`은 학습 입력에서 `observation.images.camera1`로 rename되고 logical empty camera 2개가 설정된다.
   - 관련 파일: `3_training/scripts/train_delta.py` /// `3_training/scripts/train_delta.sh` /// `3_training/scripts/validate_incremental_training_preflight.py`

## 실시간 추론 및 제어 파이프라인

1. 실제 observation 생성 및 SmolVLA 출력
   - 실제 RGB frame과 fresh Present_Position을 하나의 immutable request context로 묶는다.
   - Present_Position은 RAW→URDF 변환 후 FK하여 `[X,Y,Z,Roll,Pitch,Gripper]` state를 만든다.
   - remote SmolVLA wire output은 `[1,15,6]`이고 batch 제거 후 local action chunk는 `[15,6]`이다.
   - 관련 파일: `4_deploy/inference/camera.py` /// `4_deploy/inference/observation.py` /// `4_deploy/inference/remote_smolvla.py`

2. SmolVLA pre/postprocess
   - official LeRobot preprocessor와 postprocessor는 remote model worker에서 실행된다.
   - postprocessor가 normalized model output을 physical `m/radian/dGripper` action으로 복원한다. local 코드는 다시 normalize하지 않고 `[1,15,6]` 또는 `[15,6]` shape와 finite 여부만 검사한다.
   - 관련 파일: `4_deploy/inference/remote_smolvla.py` /// `4_deploy/inference/action_postprocess.py` /// `Delta_Weights/policy_postprocessor.json`

3. 실제 TCP 기준 상태 설정
   - 로봇이 nominal start와 정확히 일치한다고 가정하지 않는다.
   - fresh Present_Position을 RAW→URDF로 변환하고 FK하여 실제 TCP position, Roll, Pitch와 별도 Yaw를 얻는다. Yaw는 model state/action에는 포함되지 않지만 IK target rotation에는 actual FK Yaw를 사용한다.
   - 관련 파일: `4_deploy/inference/observation.py` /// `4_deploy/control/joint_mapping.py` /// `4_deploy/ik/ik_solver_v7.py`

4. Incremental trajectory 복원
   - observation state `S0`에서 `S[k+1]=S[k]+A[k]`로 A0–A14를 순차 누적한다.
   - gripper는 각 increment를 더한 직후 `[0,1]`로 clip한다.
   - 관련 파일: `4_deploy/trajectory/incremental.py` /// `3_training/scripts/inference_delta_snippet.py` /// `4_deploy/inference/action_postprocess.py`

5. 추론 latency와 arrival-state reanchor
   - response가 도착하면 해당 request ID의 immutable `s_obs/t_obs`를 찾는다.
   - response 도착 직후 fresh Present_Position을 다시 읽고 RAW→URDF→FK하여 `s_actual_arrival`을 만든다.
   - elapsed 구간의 action은 버리고, fractional arrival time을 보간한 뒤 남은 future trajectory를 실제 arrival state로 이동시킨다. planned trajectory를 실제 arrival state 대신 사용하지 않는다.
   - 관련 파일: `4_deploy/trajectory/reanchor.py` /// `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/inference/observation.py`

6. Chunk scheduler와 overlap
   - request stride는 0.9초, chunk horizon은 1.5초, requested overlap은 0.2초다.
   - old trajectory는 timestamp-valid한 동안만 계속 사용한다. 충분한 overlap에서는 new-chunk weight `0, 1/3, 2/3, 1`을 사용하고 짧은 overlap은 실제 usable 구간에 맞춰 연속적으로 압축한다.
   - expired target을 extrapolate하거나 stale last point를 반복하지 않는다.
   - 관련 파일: `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/trajectory/overlap.py` /// `4_deploy/config/deployment.yaml`

7. 10 Hz Cartesian anchor를 30 Hz target으로 보간
   - 10 Hz policy anchor 사이를 absolute timestamp 기반으로 선형보간한다.
   - 30 Hz controller는 `BEFORE_START/VALID/EXPIRED` 상태를 명시적으로 처리하며, 놓친 tick을 재생하거나 마지막 target을 extrapolate하지 않는다.
   - 관련 파일: `4_deploy/trajectory/interpolate.py` /// `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/evaluation/tests/unit/test_interpolate_regression.py`

8. IK 변환
   - target의 X/Y/Z/Roll/Pitch와 fresh/current actual-q FK에서 얻은 Yaw로 rotation target을 만든다.
   - DLS IK가 이를 `[shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll]` 5개 arm joint의 `q_nom`으로 변환한다.
   - gripper는 arm IK에 넣지 않으며 별도 normalized trajectory로 처리한다.
   - 관련 파일: `4_deploy/ik/ik_solver_v7.py` /// `4_deploy/run_live.py` /// `4_deploy/control/safety.py`

9. BP+delta 및 pose-dependent gravity support
   - controller 값은 `P=64, I=0, D=32, K_ext=0.5, q_corr clamp=±2°`다.
   - HOLD:
     `b_support_hold=b_static+b_g(q_actual)`
     `q_corr_hold=clip(K_ext*(q_nom_hold-q_actual),±2°)`
     `q_cmd_hold=q_nom_hold+b_support_hold+q_corr_hold`
   - 진짜 HOLD→TRAJECTORY에서 `q_actual_0, q_nom_0, q_cmd_hold, b_support_0`를 한 번만 저장한다.
   - TRAJECTORY:
     `Delta_q_nom=q_nom-q_nom_0`
     `Delta_q_actual=q_actual-q_actual_0`
     `q_corr_delta=clip(K_ext*(Delta_q_nom-Delta_q_actual),±2°)`
     `Delta_b_support=b_support(q_actual)-b_support_0`
     `q_cmd=q_cmd_hold+Delta_q_nom+Delta_b_support+q_corr_delta`
   - full support, static bias, q_corr_hold를 trajectory에서 다시 더하지 않는다. 새 chunk, overlap, reanchor는 controller origin을 reset하지 않는다.
   - 관련 파일: `4_deploy/control/bp_delta_controller.py` /// `4_deploy/control/gravity_compensation.py` /// `4_deploy/config/deployment.yaml`

10. URDF→RAW 영점/부호 보정과 gripper 변환
    - arm mapping 일반식은 `RAW=URDF*sign+encoder_zero`이며 현재 signs는 모두 +1이다.
    - encoder zero는 `[-8.6154,+4.0440,+2.0220,+1.0549,+1.3626]°`다.
    - gripper는 `G=0 CLOSED→3000`, `G=1 OPEN→600`으로 변환하고 입력 G를 `[0,1]`로 clip한다.
    - 관련 파일: `4_deploy/control/joint_mapping.py` /// `4_deploy/config/motor_mapping.json` /// `4_deploy/config/physical_start.json`

11. Safety gate와 Feetech MotorIO 경계
    - q_cmd finite, IK convergence, physical/safe joint limits, gripper 범위와 raw tick 범위를 먼저 검사한다.
    - arm RAW degree는 pinned calibration을 이용해 Feetech raw tick으로 변환한다.
    - 모든 검증을 통과한 명령만 `MotorIO.write_goal_ticks()`의 단일 `Goal_Position` 경계로 전달한다. 다른 canonical runtime module은 motor bus를 직접 쓰지 않는다.
    - 관련 파일: `4_deploy/control/safety.py` /// `4_deploy/control/joint_mapping.py` /// `4_deploy/control/motor_io.py`

## 전체 연결

1. Canonical production orchestration
   - 관련 파일: `4_deploy/run_live.py` /// `4_deploy/config/runtime_config.py` /// `4_deploy/config/deployment.yaml`

2. End-to-end validation
   - 관련 파일: `4_deploy/evaluation/tests/integration/test_motorless_end_to_end.py` /// `4_deploy/evaluation/live_shadow/step4_full_shadow.py` /// `docs/ARCHITECTURE.md`
