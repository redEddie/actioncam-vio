import time, sys
try:
    import scservo_sdk as scs
except ImportError:
    sys.exit("scservo_sdk 없음")

portHandler = scs.PortHandler("/dev/ttyACM0")
packetHandler = scs.PacketHandler(0)  # 프로토콜 0번
portHandler.openPort()
portHandler.setBaudRate(1000000)

MOTOR_ID = 6

print("🔧 그리퍼(ID 6) 하드웨어 안전 한계 설정 중...")

# 1. EEPROM 락 해제
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 55, 0)
time.sleep(0.1)

# 2. 최소 위치 제한 (주소 9번, 2바이트) -> 0 설정 (최대로 벌린 상태)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 9, 0)
time.sleep(0.1)

# 3. 최대 위치 제한 (주소 11번, 2바이트) -> 3300 설정 (꽉 닫은 상태)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 11, 3300)
time.sleep(0.1)

# 4. 확인 차원에서 다시 읽어오기
min_lim = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 9)[0]
max_lim = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 11)[0]

print(f"\n✅ 설정 완료! 이제 이 모터는 어떤 명령을 받아도 절대 아래 범위를 넘어가지 않습니다:")
print(f"  ▶ 최소 열림 한계: {min_lim}")
print(f"  ▶ 최대 닫힘 한계: {max_lim}")

# 5. EEPROM 락 걸기 (안전)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 55, 1)

portHandler.closePort()
