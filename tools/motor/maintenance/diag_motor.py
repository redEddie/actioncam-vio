def main():
    import time, sys
    try:
        import scservo_sdk as scs
    except ImportError:
        sys.exit("scservo_sdk 없음")
    
    portHandler = scs.PortHandler("/dev/ttyACM0")
    packetHandler = scs.PacketHandler(1)
    portHandler.openPort()
    portHandler.setBaudRate(1000000)
    MOTOR_ID = 6
    
    print("🔍 모터 정밀 진단 시스템 가동...")
    
    # 주요 상태 레지스터 읽기
    def read_1byte(addr):
        val, result, err = packetHandler.read1ByteTxRx(portHandler, MOTOR_ID, addr)
        return val
    
    def read_2byte(addr):
        val, result, err = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, addr)
        return val
    
    # 1. 하드웨어 에러 상태 (Address 69)
    hw_error = read_1byte(69)
    print(f"[상태 1] 하드웨어 에러 상태 (주소 69): {hw_error}")
    if hw_error > 0:
        print(" 🚨 경고: 하드웨어 에러가 감지되었습니다! (과부하, 과열, 전압 이상 등)")
        print(" 🛠 에러 초기화를 시도합니다...")
        # 에러 초기화를 위해 0을 씁니다 (일부 펌웨어 지원)
        packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 69, 0)
        time.sleep(0.1)
    
    # 2. 토크 한계값 (Address 48) - 2바이트
    torque_limit = read_2byte(48)
    print(f"[상태 2] 최대 토크 한계값 (주소 48): {torque_limit} / 1000")
    if torque_limit == 0:
        print(" 🚨 경고: 토크 한계값이 0으로 설정되어 있어 모터가 힘을 내지 못합니다!")
        packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 48, 1000)
        print(" 🛠 토크 한계값을 1000으로 복구했습니다.")
    
    # 3. 현재 동작 모드 (Address 33)
    op_mode = read_1byte(33)
    print(f"[상태 3] 현재 동작 모드 (주소 33): {op_mode} (0이어야 정상)")
    
    # 4. 현재 온도 (Address 62)
    temp = read_1byte(62)
    print(f"[상태 4] 현재 모터 온도: {temp}도")
    
    # 5. 토크 켜보기
    print("💡 토크를 켜고 실제 이동을 시도합니다.")
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)
    time.sleep(0.1)
    is_torque_on = read_1byte(40)
    print(f"[상태 5] 토크 켜짐 여부 (주소 40): {is_torque_on} (1이면 정상적으로 힘이 들어간 상태)")
    
    current_raw = read_2byte(56)
    target = current_raw + 300
    
    print(f"👉 {current_raw} 에서 {target} 으로 이동 명령을 내립니다.")
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)
    
    # 1초 동안 0.1초마다 현재 위치를 추적
    for i in range(10):
        pos = read_2byte(56)
        load = read_2byte(60)
        # Load 값은 10비트(0~1023), 최상위 비트는 방향
        load_mag = load & 0x3FF
        print(f"   [{i+1}/10] 현재 위치: {pos}, 현재 부하(Load): {load_mag}")
        time.sleep(0.1)
    
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
    portHandler.closePort()
    print("진단 완료.")


if __name__ == "__main__":
    main()
