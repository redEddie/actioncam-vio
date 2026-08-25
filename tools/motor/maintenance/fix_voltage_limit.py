def main():
    import time, sys
    try:
        import scservo_sdk as scs
    except ImportError:
        sys.exit("scservo_sdk 없음")
    
    portHandler = scs.PortHandler("/dev/ttyACM0")
    packetHandler = scs.PacketHandler(0)
    portHandler.openPort()
    portHandler.setBaudRate(1000000)
    
    MOTOR_ID = 6
    
    print("🔧 전압 제한(Min Voltage Limit)을 기존 모터와 동일하게 4.0V로 하향 조정합니다...")
    
    # EEPROM Lock 해제
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 55, 0)
    time.sleep(0.1)
    
    # Min Voltage Limit (주소 15)을 40 (4.0V)으로 설정
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 15, 40)
    time.sleep(0.1)
    
    # 상태 확인
    new_min_vol = packetHandler.read1ByteTxRx(portHandler, MOTOR_ID, 15)[0]
    present_vol = packetHandler.read1ByteTxRx(portHandler, MOTOR_ID, 62)[0]
    status = packetHandler.read1ByteTxRx(portHandler, MOTOR_ID, 65)[0]
    
    print(f"✅ 설정 완료! 새로운 최소 전압 제한: {new_min_vol} (4.0V)")
    print(f"현재 전압: {present_vol} (5.3V 예상)")
    print(f"현재 상태(Status): {status} (0이면 정상!)")
    
    print("\n🎯 전압 에러 우회 후 모터 이동 테스트!")
    # 토크 ON
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)  
    current = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    target = current + 500
    if target > 4095:
        target = current - 500
    
    print(f"  현재: {current} → 목표: {target}")
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)
    time.sleep(1)
    
    new_pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    print(f"  이동 후: {new_pos}")
    
    if abs(new_pos - target) < 50:
        print("  🎉 모터 정상 동작 확인! 전압 에러 우회 성공!")
    else:
        print(f"  ⚠️ 여전히 움직이지 않음. 목표와 차이: {abs(new_pos - target)}")
    
    # 원래 위치로
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, current)
    time.sleep(1)
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
    portHandler.closePort()


if __name__ == "__main__":
    main()
