import time
import sys

try:
    import scservo_sdk as scs
except ImportError:
    sys.exit("scservo_sdk를 찾을 수 없습니다.")

PORT = "/dev/ttyACM0"
BAUDRATE = 1000000
MOTOR_IDS = [1, 2, 3, 4, 5]

def main():
    portHandler = scs.PortHandler(PORT)
    packetHandler = scs.PacketHandler(1)
    
    if not portHandler.openPort():
        sys.exit("포트를 열 수 없습니다.")
    if not portHandler.setBaudRate(BAUDRATE):
        sys.exit("보드레이트 설정 실패.")
        
    print("=====================================================")
    print("관절 부호(Sign) 검증 스크립트 (상대 좌표 모드)")
    print("초기 시작 위치를 0으로 맞춥니다.")
    print("이제 손으로 움직여서 + (양수)가 되는지 - (음수)가 되는지 확인하세요.")
    print("Ctrl+C 를 누르면 종료됩니다.")
    print("=====================================================\n")
    
    # 토크 끄기
    for mid in MOTOR_IDS:
        packetHandler.write1ByteTxRx(portHandler, mid, 40, 0)
        
    # 초기 영점(0) 값 읽기
    initial_positions = {}
    time.sleep(0.5)
    for mid in MOTOR_IDS:
        val, comm, err = packetHandler.read2ByteTxRx(portHandler, mid, 56)
        if comm == scs.COMM_SUCCESS:
            initial_positions[mid] = val
        else:
            initial_positions[mid] = 0

    try:
        while True:
            positions = {}
            for mid in MOTOR_IDS:
                val, comm, err = packetHandler.read2ByteTxRx(portHandler, mid, 56)
                if comm == scs.COMM_SUCCESS and err == 0:
                    # 시작 위치 기준의 변화량(상대값)으로 계산
                    diff = val - initial_positions[mid]
                    # 양수면 앞에 +를 붙여서 직관적으로 보이게 함
                    positions[mid] = f"{diff:+d}" 
                else:
                    positions[mid] = "ERR"
            
            out_str = (
                f"Pan(1): {positions.get(1, 'E'):>6} | "
                f"Lift(2): {positions.get(2, 'E'):>6} | "
                f"Elbow(3): {positions.get(3, 'E'):>6} | "
                f"W_Flex(4): {positions.get(4, 'E'):>6} | "
                f"W_Roll(5): {positions.get(5, 'E'):>6}"
            )
            print(out_str, end="\r")
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n종료합니다.")
        portHandler.closePort()

if __name__ == "__main__":
    main()
