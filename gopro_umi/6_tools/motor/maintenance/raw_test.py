import time
import sys
import os

# LeRobot이 설치한 feetech sdk를 가져옵니다.
try:
    import scservo_sdk as scs
except ImportError:
    print("scservo_sdk를 찾을 수 없습니다. lerobot_env2 환경인지 확인하세요.")
    sys.exit(1)

# 설정 (ID 6번: 그리퍼)
PORT = "/dev/ttyACM0"
BAUDRATE = 1000000
MOTOR_ID = 6

def main():
    print("Raw 테스트 시작...")
    portHandler = scs.PortHandler(PORT)
    packetHandler = scs.PacketHandler(1) # 프로토콜 1 (SCS)
    
    if not portHandler.openPort():
        print("포트를 열 수 없습니다.")
        return
        
    if not portHandler.setBaudRate(BAUDRATE):
        print("보드레이트 설정 실패.")
        return
        
    print(f"포트 연결 성공: {PORT}")
    
    # 패킷 읽기 (레지스터 56: Present Position, 2바이트)
    current_raw_pos, scs_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)
    
    if scs_comm_result != scs.COMM_SUCCESS:
        print(f"통신 에러: {packetHandler.getTxRxResult(scs_comm_result)}")
        return
    elif dxl_error != 0:
        print(f"모터 에러: {packetHandler.getRxPacketError(dxl_error)}")
        return
        
    print(f"현재 ID {MOTOR_ID}의 Raw 위치: {current_raw_pos}")
    
    # 500 정도 Raw 값을 더해봅니다.
    target_pos = current_raw_pos + 500
    
    # 토크 켜기 (레지스터 40: Torque Enable, 1바이트)
    print("모터의 토크(힘)를 강제로 켭니다!")
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)
    
    # 레지스터 42: Goal Position (2바이트)
    print(f"3초 뒤 Raw 위치 {target_pos}로 이동합니다.")
    time.sleep(3)
    
    # 목표 위치 전송
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target_pos)
    print("명령 전송 완료!")
    
    time.sleep(2)
    print("원래 위치로 복귀합니다.")
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, current_raw_pos)
    
    # 토크 끄기
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
    
    portHandler.closePort()
    print("종료.")

if __name__ == "__main__":
    main()
