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
                     
                     
검증

1. **Remote Pipeline Latency**
   카메라 → observation → SSH → 서버 SmolVLA → `[1,15,6]` 응답까지 전체 지연시간 측정.

2. **Local 30 Hz Budget**
   scheduler → interpolation → IK → gravity → BP+delta → safety 계산이 **33.333ms 안에 끝나는지** 확인.
   [검증]

3. **Model Action Contract**
   SmolVLA 출력이 정말 **10Hz / 15-step / 6D / incremental delta / yaw 제외** 계약대로 동작하는지 검증.
   [검증]

4. **IK / FK Accuracy**
   Cartesian target을 IK로 joint로 바꾼 뒤 FK로 다시 복원해서 **위치·자세 정확도, 성공률, 속도, joint continuity** 확인.
   [검증]
   
5. **Camera-State Sync**
   GoPro 이미지와 `Present_Position`이 시간적으로 충분히 가까운지, 즉 **RGB와 robot state 동기화** 검증.
   [검증] - 큰 문제는 없으나 모터 읽기와 rgb데이터 도착 간 차이가 생기면 같은 프레임으로 동작 가능성 5%이하 

6. **Motor Bus Latency**
   실제 `Present_Position READ → 계산 → Goal_Position WRITE` 통신 속도가 30Hz 운용에 충분한지 확인.
   [검증]

7. **Closed-Loop Tracking**
   `q_nom`, `q_cmd`, `q_actual`을 비교해서 **실제 로봇이 목표 joint trajectory를 얼마나 잘 따라가는지** 측정.

8. **Gravity Tracking Ablation**
   gravity OFF / fixed support / dynamic gravity를 비교해서 **현재 중력보상이 실제 tracking을 개선하는지** 검증.

9. **Joint Mapping Accuracy**
   `RAW ↔ URDF ↔ encoder_zero` 변환과 FK 입력이 정확한지 확인. 즉 **좌표/엔코더 offset 오류 방지**.
   [검증]

10. **Trajectory Reconstruction**
    15-step delta 누적 `S[k+1]=S[k]+A[k]`과 **10Hz → 30Hz interpolation**이 수학적으로 정확한지 검증.
    [검증]
    
    # Canonical GoPro-UMI 파이프라인

여러 관련 파일은 `///`로 구분한다.

## 학습 데이터 및 학습 파이프라인

1. 에피소드 수집

   * GoPro 영상에 RGB와 IMU/GPMF 데이터를 함께 기록한다. 이후 영상에서 RGB 프레임과 IMU 데이터를 추출하고 timestamp를 정렬한다.
   * 원본 영상은 GoPro 8:7 Native, 3840×3360, 약 59.94 fps HEVC를 사용한다.
   * 관련 파일: `1_capture/actioncam-vio/run_umi_pipeline.sh` /// `1_capture/actioncam-vio/gopro_vio/extract.py` /// `1_capture/actioncam-vio/gopro_vio/gpmf.py`

2. ORB-SLAM3/VIO 및 좌표계 변환

   * RGB와 IMU로 6DOF camera trajectory를 복원한다.
   * 이어서 world alignment를 수행하고 최종 pose를 `robot_base -> tcp_link` 기준으로 변환한다.
   * 최종 Robot Base Cartesian convention은 `+X=Forward`, `+Y=Left`, `+Z=Up`이다.
   * 관련 파일: `1_capture/actioncam-vio/gopro_vio/slam.py` /// `1_capture/actioncam-vio/gopro_vio/world_align.py` /// `1_capture/actioncam-vio/gopro_vio/tcp_robot_transform.py`

3. Yaw-free source Zarr 생성

   * RGB, TCP position, rotation, gripper, episode boundary를 source Zarr로 만든다.
   * rotation을 Rotation Matrix 및 Euler ZYX로 변환한 뒤 Yaw를 Policy State에서 제외하여 `[X,Y,Z,Roll,Pitch,Gripper]` 6D state를 만든다.
   * Arm pose만 보면 5DOF이고 Gripper를 합치면 최종 Policy State는 6D다.
   * 현재 Canonical Dataset은 `202608161903`이며 99개 Episode 전체의 Raw MP4와 Reconstruction Pipeline이 보존되어 있어 `FULLY REGENERABLE` 상태다.
   * `1_capture/actioncam-vio/Episode_result`의 78 Episode는 초기 Working Reconstruction이며, Canonical 99-Episode Dataset 전체와 동일한 범위가 아니다.
   * 관련 파일: `2_dataset/conversion/build_umi_zarr.py` /// `2_dataset/conversion/build_yawfree_zarr.py` /// `7_storage/datasets/202608161903/02_zarr/replay_buffer_yawfree.zarr`

4. 10 Hz sampling

   * source timestamp를 기준으로 target time `0.0, 0.1, 0.2, ...`에 대응하는 source frame/state를 선택한다.
   * 최종 Policy Rate는 10 Hz이며 Action 간격은 100 ms이다.
   * 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `2_dataset/validation/test_convert_to_lerobot_10hz_incremental.py`

5. Step-to-step incremental action 추출

   * 각 Episode 내부에서만 `A[t] = S[t+1] - S[t]`를 계산한다.
   * Action 순서는 `[dX,dY,dZ,dRoll,dPitch,dGripper]`이고 Yaw는 포함하지 않는다.
   * `A[1]=S[2]-S[0]` 형태의 anchor-relative action이나 absolute target을 사용하지 않는다.
   * Reconstruction은 반드시 `S[k+1]=S[k]+A[k]` 순차 누적 방식으로 수행한다.
   * 마지막 State는 마지막 Action의 Target으로만 사용하며 synthetic zero terminal row는 저장하지 않는다.
   * 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `3_training/scripts/incremental_action_contract.py` /// `3_training/scripts/tests/test_incremental_action_contract.py`

6. LeRobot dataset 직렬화

   * 각 LeRobot row에는 RGB, `observation.state [6]`, `action [6]`, timestamp와 episode metadata를 저장한다.
   * 각 row에 `[15,6]` Action Chunk를 직접 저장하지 않는다.
   * 현재 Canonical Dataset:
     `7_storage/datasets/202608161903/03_lerobot/lerobot_dataset_10hz_chunk15_incremental_baseline_v1`
   * 현재 Dataset은 99 Episodes, 99,850 Frames이다.
   * 관련 파일: `2_dataset/conversion/convert_to_lerobot_10hz_incremental.py` /// `2_dataset/validation/test_convert_to_lerobot_10hz_incremental.py`

7. 15-step action window 구성

   * LeRobot Training Sampler가 row-wise `action [6]`에서 미래 15개 Action을 읽어 `[15,6]` window를 만든다.
   * `chunk_size=15`, `n_action_steps=15`, Action Rate 10 Hz이므로 Horizon은 1.5초다.
   * Window 의미:
     `[S[t+1]-S[t], S[t+2]-S[t+1], ..., S[t+15]-S[t+14]]`
   * 관련 파일: `3_training/scripts/train_delta.py` /// `3_training/scripts/train_delta.sh` /// `7_storage/Delta_Weights/config.json`

8. 학습 전 fail-closed 검증

   * Dataset FPS, Episode/Frame 수, 6D State/Action Schema, Step-to-step Contract, Chunk 15, Timestamp, finite 여부, Source→10 Hz mapping, Model/Output Path를 검증한다.
   * Placeholder validation artifact를 허용하지 않는다.
   * 현재 Preflight 기준 Canonical Training Path와 Dataset Path는 모두 정상 resolve된다.
   * 관련 파일: `2_dataset/validation/validate_canonical_10hz_dataset.py` /// `3_training/scripts/validate_incremental_training_preflight.py` /// `3_training/scripts/incremental_action_contract.py`

9. SmolVLA 학습

   * Canonical Training Entry Point는 별도 One-shot Script가 아니라 `3_training/scripts/train_delta.py`로 단일 통합되어 있다.
   * 10 Hz Dataset, Chunk 15, `n_action_steps=15`, 6D Incremental Action, Batch Size 32, 20,000 Steps를 사용한다.
   * Dataset의 `observation.images.top`은 Policy Config에서 `observation.images.camera1`로 정확히 Mapping된다.
   * State/Action Normalization은 `MEAN_STD`, Visual은 `IDENTITY`이다.
   * 현재 Canonical Training에는 별도 Image Augmentation, Numerical State Noise, Numerical Action Noise가 구현되어 있지 않다.
   * 관련 파일: `3_training/scripts/train_delta.py` /// `3_training/scripts/train_delta.sh` /// `3_training/scripts/validate_incremental_training_preflight.py`

---

## 실시간 추론 및 제어 파이프라인

1. 실제 observation 생성 및 SmolVLA 출력

   * 실제 RGB Frame과 Fresh `Present_Position`을 하나의 immutable request context로 묶는다.
   * `Present_Position`은 RAW→URDF 변환 후 FK하여 `[X,Y,Z,Roll,Pitch,Gripper]` State를 만든다.
   * 저장된 `physical_start.json`을 현재 Robot State Truth로 사용하지 않는다.
   * Remote SmolVLA Wire Output은 `[1,15,6]`이고 Batch 제거 후 Local Action Chunk는 `[15,6]`이다.
   * 관련 파일: `4_deploy/inference/camera.py` /// `4_deploy/inference/observation.py` /// `4_deploy/inference/remote_smolvla.py`

2. SmolVLA pre/postprocess

   * Official LeRobot Preprocessor와 Postprocessor는 Remote Model Worker에서 실행된다.
   * Postprocessor가 normalized model output을 Physical `meter/radian/dGripper` Action으로 복원한다.
   * Local Runtime은 다시 Normalize/Unnormalize하지 않고 Shape와 finite 여부를 검사한다.
   * Remote Worker는 Persistent SSH 연결 위에서 `python -u -c` Inline Worker 방식으로 동작한다.
   * 관련 파일: `4_deploy/inference/remote_smolvla.py` /// `4_deploy/inference/action_postprocess.py` /// `7_storage/Delta_Weights/policy_postprocessor.json`

3. 실제 TCP 기준 상태 설정

   * Robot이 Nominal Start와 정확히 일치한다고 가정하지 않는다.
   * Fresh `Present_Position`을 RAW→URDF로 변환하고 FK하여 실제 TCP Position, Roll, Pitch와 별도 Yaw를 얻는다.
   * Yaw는 Model State/Action에는 포함되지 않지만 IK Target Rotation에는 Current Actual FK Yaw를 사용한다.
   * 관련 파일: `4_deploy/inference/observation.py` /// `4_deploy/control/joint_mapping.py` /// `4_deploy/ik/ik_solver_v7.py`

4. Incremental trajectory 복원

   * Observation State `S0`에서 `S[k+1]=S[k]+A[k]`로 A0–A14를 순차 누적한다.
   * Absolute target 또는 `S0+A[k]` 방식으로 복원하지 않는다.
   * Gripper도 `G[k+1]=G[k]+dG[k]` 형태로 순차 누적하고 Runtime 허용 범위 내에서 Clip한다.
   * 관련 파일: `4_deploy/trajectory/incremental.py` /// `3_training/scripts/inference_delta_snippet.py` /// `4_deploy/inference/action_postprocess.py`

5. 추론 latency와 arrival-state reanchor

   * Response가 도착하면 해당 Request ID의 immutable `s_obs/t_obs`를 사용한다.
   * Response 도착 직후 Fresh `Present_Position`을 다시 읽고 RAW→URDF→FK하여 `s_actual_arrival`을 만든다.
   * 이미 지나간 Trajectory Segment는 제거하고 Fractional Arrival Time은 보간한다.
   * 남아 있는 Future Displacement만 보존하여 실제 Arrival State를 새로운 Trajectory Origin으로 사용한다.
   * Planned Arrival State를 실제 `s_actual_arrival` 대신 사용하지 않는다.
   * 관련 파일: `4_deploy/trajectory/reanchor.py` /// `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/inference/observation.py`

6. Chunk scheduler와 overlap

   * Chunk Horizon은 1.5초, 15 Step이다.
   * Request Stride는 0.9초, 9 Step이다.
   * 따라서 Old/New Chunk가 시간상 동시에 존재 가능한 Nominal Temporal Overlap은 `15-9=6 Step`, 즉 0.6초다.
   * 그러나 실제 Weight Blending은 6 Step 전체에 적용하지 않고 Handoff의 2 Step, 즉 0.2초에만 적용한다.
   * Blend Step 1: `Old 2/3 + New 1/3`
   * Blend Step 2: `Old 1/3 + New 2/3`
   * 이후: `New 100%`
   * 즉 `6-Step Temporal Overlap != 2-Step Actual Blend`이다.
   * 실제 usable blend가 1 Step이면 1-Step Transition으로 축소하고, 0 Step이면 Old Trajectory/HOLD/Fresh Inference 계열 Late Fallback을 사용한다.
   * Expired Target을 extrapolate하거나 stale last point를 반복하지 않는다.
   * 관련 파일: `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/trajectory/overlap.py` /// `4_deploy/config/deployment.yaml`

7. 10 Hz Cartesian anchor를 30 Hz target으로 보간

   * 10 Hz Policy Anchor 사이를 Absolute Timestamp 기준으로 보간한다.
   * XYZ는 Linear Interpolation을 사용한다.
   * Roll/Pitch는 `unwrap -> interpolation -> wrap`으로 Angle Discontinuity를 방지한다.
   * 30 Hz Controller는 `BEFORE_START/VALID/EXPIRED` 상태를 명시적으로 처리하며, 놓친 Tick을 뒤늦게 재생하거나 마지막 Target을 무한 Extrapolation하지 않는다.
   * 관련 파일: `4_deploy/trajectory/interpolate.py` /// `4_deploy/trajectory/chunk_scheduler.py` /// `4_deploy/evaluation/tests/unit/test_interpolate_regression.py`

8. IK 변환

   * Target X/Y/Z/Roll/Pitch와 Fresh/Current Actual Joint의 FK에서 얻은 Yaw로 Target Rotation을 만든다.
   * Euler Convention은 ZYX이며 `R_target=Rz(Current_Yaw)*Ry(Target_Pitch)*Rx(Target_Roll)`이다.
   * DLS IK가 이를 `[shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll]` 5개 Arm Joint의 `q_nom`으로 변환한다.
   * `q_nom` 단위는 radian이다.
   * Gripper는 Arm IK에 넣지 않고 별도로 처리한다.
   * 관련 파일: `4_deploy/ik/ik_solver_v7.py` /// `4_deploy/run_live.py` /// `4_deploy/control/safety.py`

9. BP+delta 및 pose-dependent gravity support

   * Controller 값은 `P=64, I=0, D=32, K_ext=0.5, q_corr clamp=±2°`이다.
   * 현재 Production Gravity Support는 ENABLED이며 30 Hz Position-domain 방식으로 동작한다.
   * HOLD:
     `b_support_hold = b_static + b_g(q_actual)`
     `q_corr_hold = clip(K_ext*(q_nom_hold-q_actual), ±2°)`
     `q_cmd_hold = q_nom_hold + b_support_hold + q_corr_hold`
   * 실제 HOLD→TRAJECTORY 전환 시 `q_actual_0, q_nom_0, q_cmd_hold, b_support_0`를 한 번만 저장한다.
   * TRAJECTORY:
     `Delta_q_nom = q_nom - q_nom_0`
     `Delta_q_actual = q_actual - q_actual_0`
     `q_corr_delta = clip(K_ext*(Delta_q_nom-Delta_q_actual), ±2°)`
     `Delta_b_support = b_support(q_actual)-b_support_0`
     `q_cmd = q_cmd_hold + Delta_q_nom + Delta_b_support + q_corr_delta`
   * Full Support, Static Bias, `q_corr_hold`를 Trajectory에서 다시 중복 추가하지 않는다.
   * 새 Chunk, Overlap, Re-anchor는 Controller Origin을 Reset하지 않는다.
   * 현재 Static Bias는 `[-0.264°, 0.0°, 0.0°, 0.0°, -0.352°]`이다.
   * Camera Payload는 `0.240 kg`, Gripper/Custom Jaw Payload는 `0.280 kg`이며 Mass/COM 기반 Pose-dependent Support를 사용한다.
   * 직접 Joint Torque Command는 사용하지 않는다.
   * 관련 파일: `4_deploy/control/bp_delta_controller.py` /// `4_deploy/control/gravity_compensation.py` /// `4_deploy/config/deployment.yaml`

10. URDF→RAW 영점/부호 보정과 gripper 변환

    * Arm Mapping 일반식은 `RAW = URDF*sign + encoder_zero`이다.
    * Reverse Mapping은 `URDF = (RAW-encoder_zero)/sign`이다.
    * 현재 Joint Order:
      `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]`
    * 현재 Signs:
      `[+1,+1,+1,+1,+1]`
    * 현재 승인된 Encoder Zero:
      `[+6.6813,+4.0879,+3.3846,+0.1319,+5.6703]°`
    * 과거 `[-8.6154,+4.0440,+2.0220,+1.0549,+1.3626]°` 값은 현재 Production Mapping이 아니므로 사용하지 않는다.
    * Gripper Dataset Closed는 `0.0`, Open Reference는 `0.441305`이다.
    * Physical Motor 기준 Full Open은 약 `600 tick`, Full Closed는 약 `3000 tick`이다.
    * `0.441305`는 물리적으로 44.13% 열린 상태를 의미하지 않고 Dataset Semantic Open Reference이다.
    * 관련 파일: `4_deploy/control/joint_mapping.py` /// `4_deploy/config/motor_mapping.json` /// `4_deploy/config/physical_start.json`

11. Safety gate와 Feetech MotorIO 경계

    * `q_cmd` finite, NaN/Inf, IK convergence, Physical/Safe Joint Limit, Joint Continuity, Gripper Range, RAW/Tick Range, Expired Trajectory, Communication Error를 검사한다.
    * Arm RAW Degree는 현재 Calibration을 이용해 Feetech Raw Tick으로 변환한다.
    * 모든 검증을 통과한 Command만 `MotorIO.write_goal_ticks()`의 단일 `Goal_Position` 경계로 전달한다.
    * 다른 Canonical Runtime Module은 Motor Bus를 직접 쓰지 않는다.
    * 관련 파일: `4_deploy/control/safety.py` /// `4_deploy/control/joint_mapping.py` /// `4_deploy/control/motor_io.py`

## 전체 연결

1. Canonical production orchestration

   * Canonical Runtime Entry Point는 `4_deploy/run_live.py`이다.
   * Local Project Root는 `/home/kimminje/Desktop/project/gopro_umi`, Remote Project Root는 `/home/kimminje/gopro_umi`이며 서로 다른 Absolute Path를 의도적으로 분리하여 사용한다.
   * Local Production Model은 `7_storage/run/pretrained_model`, Remote Production Model은 `/home/kimminje/gopro_umi/7_storage/datasets/202608161903/04_training/final/pretrained_model`이다.
   * 현재 Path/Name Audit 결과 Local/Remote Dataset, Model, Config, URDF, Import, Module Name, Case, Hyphen/Underscore Mapping은 모두 PASS이며 Required Fix는 없다.
   * 관련 파일: `4_deploy/run_live.py` /// `4_deploy/config/runtime_config.py` /// `4_deploy/config/deployment.yaml` /// `4_deploy/inference/remote_smolvla.py`

2. End-to-end validation

   * Live Shadow Pipeline은 현재 사용하지 않으며 관련 Entry Point는 Archive되었다.
   * 따라서 제거된 `4_deploy/evaluation/live_shadow/step4_full_shadow.py`를 현재 End-to-End Validation 경로로 사용하지 않는다.
   * 현재 Canonical Test Baseline은 `67 collected / 67 passed / 0 failed`이다.
   * Static Compile은 100 Python Files PASS, Production Core Import는 15/15 PASS, Broken Symlink는 0이다.
   * `etc/archive/patch_history/pipeline/replay_yawfree_shadow.py`는 이름과 달리 현재 Shadow Runtime이 아니라 `evaluate_10hz_direction.py`의 `ZarrV2Array` Offline Evaluation Dependency로만 유지한다.
   * 관련 파일: `4_deploy/evaluation/tests/integration/test_motorless_end_to_end.py` /// `4_deploy/evaluation/model/evaluate_10hz_direction.py` /// `etc/archive/patch_history/pipeline/replay_yawfree_shadow.py`

3. 현재 최종 상태

   * Canonical Dataset ID: `202608161903`
   * Canonical Dataset: `99 Episodes / 99,850 Frames`
   * Dataset Regeneration: `FULLY REGENERABLE`
   * Policy: `10 Hz / Chunk 15 / Horizon 1.5 s / 6D Step-to-Step Incremental`
   * Scheduler: `Stride 9 Step = 0.9 s`
   * Temporal Overlap: `6 Step = 0.6 s`
   * Actual Weighted Blend: `2 Step = 0.2 s`
   * Production Model Step: `20,000`
   * Production `model.safetensors` SHA256:
     `35ee5629268aa5bb5c62c99fc932a269e390bc24c23c6594bea9d040cf276298`
   * Training Path Audit: `PASS`
   * Local Inference Path Audit: `PASS`
   * Remote Inference Path Audit: `PASS`
   * Dataset/Model Contract: `PASS`
   * Required Fixes: `NONE`
   * Project Cleanup Status: `FROZEN`
























/live_shadow/step4_full_shadow.py` /// `etc/docs/ARCHITECTURE.md`
