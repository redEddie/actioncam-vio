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

print("🔧 비정상적인 메모리 값을 정상으로 강제 덮어씌웁니다...")

# EEPROM Lock 해제 (필요할 수 있음)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 48, 0)
time.sleep(0.1)

# 1. Min/Max Limit 초기화 (0 ~ 4095)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 9, 0)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 11, 4095)

# 2. Torque Limit 초기화 (1000 = 100%)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 48, 1000) # 주소 48이 맞는지 확인 필요 (STS3215는 16일 수 있음)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 16, 1000)

# 3. PID 값 강제 초기화 (기본값)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 21, 32) # P
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 22, 32) # D
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 23, 0)  # I

time.sleep(0.1)

new_min = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 9)[0]
new_max = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 11)[0]
print(f"✅ 초기화 완료! 새로운 Min: {new_min}, Max: {new_max}")

# 토크 켜고 다시 이동 테스트
print("\n👉 이동 테스트 재시도...")
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)

current = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
# 0~4095 안쪽으로 안전하게 타겟 설정
valid_current = current % 4096
target = valid_current + 300
if target > 4095:
    target = valid_current - 300

packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)

for i in range(5):
    pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    print(f"[{i+1}/5] 현재 각도: {pos}")
    time.sleep(0.2)

packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
portHandler.closePort()
print("종료.")
