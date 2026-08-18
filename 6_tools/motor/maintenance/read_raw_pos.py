import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    
    print("Connecting to robot on /dev/ttyACM0...")
    config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="None", # We use None because the calibration file was named None.json
        disable_torque_on_disconnect=True
    )
    robot = SO101Follower(config)
    
    try:
        robot.connect(calibrate=False)
        robot.bus.disable_torque() # 선배님이 쉽게 움직일 수 있도록 토크를 풉니다
        print("✅ Connection successful! Reading current raw values...")
        print("-" * 30)
        positions = robot.bus.sync_read("Present_Position")
        for name, pos in positions.items():
            print(f"{name.ljust(15)}: {pos}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
