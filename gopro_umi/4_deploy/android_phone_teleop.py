"""
SO-100 Follower Arm - Official HuggingFace Android WebXR Phone Teleoperation
==============================================================================
[기술 설명: 허깅페이스 공식 LeRobot 안드로이드 WebXR 텔레옵 모듈]
1. Android WebXR 3D Spatial Tracking:
   - 안드로이드 크롬 브라우저 WebXR AR 카메라를 이용하여 스마트폰의 3D 공간 포즈(X, Y, Z, Roll, Pitch, Yaw)를 실시간 추적.
2. Official LeRobot Processor Pipeline:
   - MapPhoneActionToRobotAction -> EEReferenceAndDelta -> EEBoundsAndSafety -> GripperVelocityToJoint -> InverseKinematicsEEToJoints
3. UMI 3D Wand Control:
   - 스마트폰 화면의 'Move' 버튼을 누르고 손을 허공에서 움직이면 UMI 고글 조작처럼 로봇 팔이 손의 3D 궤적을 1:1로 실시간 추종.
==============================================================================
"""

import time
import sys
from pathlib import Path

# Teleop/lerobot/src 경로 자동 추가
lerobot_src = Path(__file__).resolve().parent / "Teleop" / "lerobot" / "src"
if lerobot_src.exists() and str(lerobot_src) not in sys.path:
    sys.path.insert(0, str(lerobot_src))

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.types import RobotAction, RobotObservation

FPS = 30

def main():
    print("\n📱 [안드로이드 WebXR 3D 스마트폰 텔레옵 준비 중...]")
    
    # 1. 로봇 및 안드로이드 폰 텔레옵 설정
    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0", id="so100_follower_arm", use_degrees=True
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    robot = SO100Follower(robot_config)
    teleop_device = Phone(teleop_config)

    # 2. 공식 Kinematics Solver 초기화
    urdf_path = "URDF/so_arm_with_gopro_final.urdf"
    kinematics_solver = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="tcp_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # 3. 허깅페이스 공식 Phone-to-Robot IK 프로세서 파이프라인 수식 구축
    phone_to_robot_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[
            MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                motor_names=list(robot.bus.motors.keys()),
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
                initial_guess_current_joints=True,
                orientation_weight=0.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # 4. 로봇 및 안드로이드 스마트폰 서버 연결
    print("🤖 로봇 팔 연결 중...")
    robot.connect()
    
    print("\n🌐 안드로이드 WebXR 서버 가동 중...")
    print("------------------------------------------------------------")
    print("💡 아래에 출력되는 웹 URL 주소를 안드로이드 스마트폰 크롬 브라우저로 접속해 주세요!")
    print("------------------------------------------------------------")
    teleop_device.connect()

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("로봇 또는 스마트폰 연결에 실패했습니다!")

    print("\n============================================================")
    print("🚀 [안드로이드 폰 텔레옵 루프 시작!]")
    print("1. 스마트폰 화면에서 'Start' 버튼을 누릅니다.")
    print("2. 'Move' 버튼을 누르고 계신 동안 손을 허공에서 움직이면 UMI처럼 3D 추적됩니다.")
    print("3. 'A' 버튼: 그리퍼 열기 / 'B' 버튼: 그리퍼 닫기")
    print("4. Ctrl+C : 종료")
    print("============================================================\n")

    try:
        while True:
            t0 = time.perf_counter()

            # 관절 관측 및 폰 액션 수신
            robot_obs = robot.get_observation()
            phone_action = teleop_device.get_action()

            # Phone 3D Pose ➔ IK Pipeline ➔ 로봇 관절 명령 계산
            joint_action = phone_to_robot_joints_processor((phone_action, robot_obs))

            # 로봇으로 전송
            robot.send_action(joint_action)

            # 30Hz 루프 정밀 동기화
            dt_s = time.perf_counter() - t0
            time.sleep(max(1 / FPS - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\n텔레옵 종료 중...")
    finally:
        teleop_device.disconnect()
        robot.disconnect()

if __name__ == "__main__":
    main()
