def main():
    import time, sys
    try:
        import scservo_sdk as scs
    except ImportError:
        sys.exit("scservo_sdk 없음")
    
    portHandler = scs.PortHandler("/dev/ttyACM0")
    packetHandler = scs.PacketHandler(0)  # 프로토콜 0번
    portHandler.openPort()
    portHandler.setBaudRate(1000000)
    
    print("🔍 로봇 팔 전체 모터(ID 1~6) 모델명 스캔을 시작합니다...\n")
    
    model_dict = {
        777: "STS3215",
        2825: "STS3250"
    }
    
    for motor_id in range(1, 7):
        model, result, err = packetHandler.read2ByteTxRx(portHandler, motor_id, 3)
        
        if result == scs.COMM_SUCCESS:
            model_name = model_dict.get(model, f"알 수 없음 (모델코드: {model})")
            print(f"✅ ID {motor_id}번 모터 발견! ➔ 모델명: {model_name}")
        else:
            print(f"❌ ID {motor_id}번 모터 응답 없음 (연결 안 됨)")
    
    print("\n스캔이 완료되었습니다!")
    portHandler.closePort()


if __name__ == "__main__":
    main()
