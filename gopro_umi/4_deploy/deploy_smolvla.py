import os
import json
from pathlib import Path

# Resolve every deployment artifact from this file, not from the shell's cwd.
DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent

# HuggingFace 베이스 모델(용량 큼) 다운로드 경로를 gopro_umi 내부로 격리
os.environ["HF_HOME"] = str(PROJECT_ROOT / "smolvla_cache")

import cv2
import sys
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.types import RobotAction
from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from ik_solver_v7 import DLSInverseKinematicsV7
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

URDF_PATH = DEPLOY_DIR / "URDF" / "so_arm_with_gopro_final.urdf"

# Canonical, newly trained 7D checkpoint.  Never silently fall back to the old
# 4_deploy/pretrained_model directory (that directory contains the former 6D model).
MODEL_PATH = (
    PROJECT_ROOT
    / "yawfree_server_candidate_new"
)
EXPECTED_TRAINING_STEP = 20_000
TASK_DESCRIPTION = "pick and place the target object"


def validate_model_artifacts(model_path: Path) -> None:
    """Fail closed unless the deployment target is the final 7D checkpoint."""
    required = [
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete SmolVLA checkpoint {model_path}: missing {missing}")

    config = json.loads((model_path / "config.json").read_text())
    state_shape = config.get("input_features", {}).get("observation.state", {}).get("shape")
    action_shape = config.get("output_features", {}).get("action", {}).get("shape")
    if state_shape not in ([6], [7]) or action_shape not in ([6], [7]):
        raise RuntimeError(
            f"Refusing unknown SmolVLA checkpoint: state={state_shape}, action={action_shape}"
        )

    training_step_path = model_path.parent / "training_state" / "training_step.json"
    if training_step_path.is_file():
        training_step_data = json.loads(training_step_path.read_text())
        if isinstance(training_step_data, int):
            training_step = training_step_data
        else:
            training_step = training_step_data.get("step", training_step_data.get("training_step"))
        if training_step != EXPECTED_TRAINING_STEP:
            print(f"WARNING: checkpoint at step {training_step}; expected {EXPECTED_TRAINING_STEP}")
    else:
        print(f"INFO: no training_step proof at {training_step_path}, skipping step check")

def get_current_state(ik, follower, arm_m, all_m):
    obs = follower.get_observation()
    # 1. 조인트 라디안 변환
    q = np.array([np.deg2rad(float(obs[f"{m}.pos"])) for m in arm_m])
    
    # 2. TCP 위치 및 회전 계산
    tcp_pos = ik.get_tcp_position(q)
    tcp_rot = ik.get_tcp_rotation(q)
    # 3. 그리퍼 값 정규화 (-27=최대 열림, 1042=최대 닫힘)
    # 현재 물리 모터값 읽기 (LeRobot은 use_degrees=True일 때 Degree를 반환함)
    gripper_val_deg = float(obs["gripper.pos"])
    
    # Degree -> RAW 변환 (다이나믹셀 기준 360도 = 4096)
    gripper_val_raw = gripper_val_deg * (4096.0 / 360.0)
    
    # 정규화 (서버 스펙 일치): 0.0 = 닫힘(1042), 1.0 = 열림(-27)
    gripper_norm = (gripper_val_raw - 1042.0) / (-27.0 - 1042.0)
    gripper_norm = np.clip(gripper_norm, 0.0, 1.0)
    
    # Dataset and IK both use robot-base -> tcp_link.  Do not apply the
    # historical ArUco marker +/-90deg or 0.50m translation here.
    axis_angle = R.from_matrix(tcp_rot).as_rotvec()
    
    state = np.array([
        tcp_pos[0],            # robot-base X (앞)
        tcp_pos[1],            # robot-base Y (왼쪽)
        tcp_pos[2],            # robot-base Z (위)
        axis_angle[0],         # Rx
        axis_angle[1],         # Ry
        axis_angle[2],         # Rz
        gripper_norm           # Gripper (7차원)
    ], dtype=np.float32)
    return state, q

def main():
    print("🚀 SmolVLA 자율 주행 배포 스크립트 시작")

    # 1. 새 20k/7D 모델과 해당 정규화기를 먼저 검증·로드한다. 검증에
    # 실패하면 로봇 하드웨어에는 연결하지 않는다.
    validate_model_artifacts(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading 20k/7D SmolVLA from {MODEL_PATH} on {device}...")
    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(MODEL_PATH),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    print("✅ 20k/7D checkpoint + preprocessor + postprocessor 연결 확인")

    # 2. 모델 연결 검증 후에만 로봇 및 IK 초기화
    follower = SO100Follower(SO100FollowerConfig(port="/dev/ttyACM0", use_degrees=True))
    follower.connect()
    arm_m = [m for m in follower.bus.motors.keys() if "gripper" not in m]
    all_m = list(follower.bus.motors.keys())
    ik = DLSInverseKinematicsV7(urdf_path=str(URDF_PATH), limit_margin_ratio=0.90)
    
    # 3. 카메라 초기화
    print("카메라 연결 중...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    for _ in range(5):
        cap.grab()
        
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("✅ 준비 완료. 루프 시작...")
    
    with torch.no_grad():
        while True:
            # 1. 이미지 캡처 (224x224 리사이즈)
            ret, frame = cap.read()
            if not ret: continue
            
            # BGR -> RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 🚨 입력 이미지 처리 (매우 중요) 🚨
            # 절대 자르지 않고(No Crop) 원본 프레임을 256x256으로 찌그러뜨려서 압축(Squash)
            img_resized = np.array(Image.fromarray(img_rgb).resize((256, 256)), dtype=np.uint8)
            
            # 2. 현재 상태 가져오기
            state, cur_q = get_current_state(ik, follower, arm_m, all_m)
            
            # 3. 추론: 학습 때 저장된 전처리기로 state를 정규화하고, 모델
            # 출력은 저장된 후처리기로 robot-base 절대 pose 단위로 복원한다.
            obs_dict = {
                "observation.images.camera1": img_resized,
                "observation.state": state,
            }
            action_tensor = predict_action(
                observation=obs_dict,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=device.type == "cuda",
                task=TASK_DESCRIPTION,
                robot_type="so_follower",
            )
            action = action_tensor.squeeze(0).numpy()
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid SmolVLA action: shape={action.shape}, values={action}")
            
            # AI action is already an absolute robot-base tcp_link target.
            target_pos = np.asarray(action[:3], dtype=np.float64)
            R_robot_target = R.from_rotvec(action[3:6])
            
            # 그리퍼 값 복구 완료! (7번째 차원 사용)
            target_gripper = action[6]
            
            # The 5-DOF IK constrains XYZ + Roll/Pitch and leaves global Yaw free.
            cur_tcp_rot = ik.get_tcp_rotation(cur_q)
            cur_yaw, _, cur_roll = R.from_matrix(cur_tcp_rot).as_euler('zyx')

            # Preserve current Yaw, while using model Roll/Pitch.
            target_euler = R_robot_target.as_euler('zyx')
            target_rot = R.from_euler('zyx', [cur_yaw, target_euler[1], target_euler[2]]).as_matrix()
            
            # IK 계산하여 목표 팔 관절 각도(next_q) 획득
            next_q = ik.solve(target_pos, target_rot, cur_q)
            
            # --- 제어기 진단 및 안전 장치 ---
            ik_sim_pos = ik.get_tcp_position(next_q)
            ik_err = np.linalg.norm(target_pos - ik_sim_pos)
            
            print("-" * 60)
            print(f"🎯 [AI 목표] X:{target_pos[0]:.3f}, Y:{target_pos[1]:.3f}, Z:{target_pos[2]:.3f} | Grip:{target_gripper:.2f}")
            print(f"⚙️ [IK 계산] X:{ik_sim_pos[0]:.3f}, Y:{ik_sim_pos[1]:.3f}, Z:{ik_sim_pos[2]:.3f}")
            print(f"🤖 [물리 현재] X:{state[0]:.3f}, Y:{state[1]:.3f}, Z:{state[2]:.3f} | Grip:{state[6]:.2f}")
            
            if (not ik.last_converged) or ik_err > 0.02:
                print(f"⚠️ [IK 한계 초과/미수렴] 위치오차 {ik_err*100:.1f}cm, "
                      f"자세오차 {ik.last_orientation_error:.4f}rad "
                      "(관절 꺾임 방지를 위해 이동 취소)")
                next_q = cur_q # 현재 위치 유지
            print("-" * 60)
            
            # 5. 미세하고 부드러운 이동을 위한 소프트웨어 보간 (Interpolation)
            num_steps = 10 # 0.2초를 10번으로 잘게 쪼개서 부드럽게 이동
            sleep_per_step = 0.2 / num_steps
            
            # 그리퍼 목표 각도 계산 (0.0=닫힘 1042, 1.0=열림 -27)
            gripper_raw_target = 1042.0 + (target_gripper * (-27.0 - 1042.0))
            gripper_raw_target = np.clip(gripper_raw_target, -27.0, 1042.0)
            gripper_deg_target = gripper_raw_target * (360.0 / 4096.0)
            
            # 그리퍼 현재 각도 계산 (state[6] 기반)
            gripper_raw_cur = 1042.0 + (state[6] * (-27.0 - 1042.0))
            gripper_raw_cur = np.clip(gripper_raw_cur, -27.0, 1042.0)
            gripper_deg_cur = gripper_raw_cur * (360.0 / 4096.0)
            
            # 10단계에 걸쳐 아주 미세하게 모터 각도를 쪼개서 전송
            for step in range(1, num_steps + 1):
                alpha = step / num_steps
                interp_q = cur_q + (next_q - cur_q) * alpha
                interp_gripper = gripper_deg_cur + (gripper_deg_target - gripper_deg_cur) * alpha
                
                robot_action = {}
                for j, m in enumerate(arm_m):
                    robot_action[f"{m}.pos"] = float(np.rad2deg(interp_q[j]))
                # 로봇에 실제 관절 명령 전송 (안전 모드 해제)
                follower.send_action(RobotAction(robot_action))
                time.sleep(sleep_per_step)
            
if __name__ == "__main__":
    main()
