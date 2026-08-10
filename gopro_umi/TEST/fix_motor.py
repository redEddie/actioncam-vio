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

print("🔥 완벽한 해결책 적용 시작 (영점 초기화 및 물리적 한계 돌파)...")

# 1. 토크 끄기
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
time.sleep(0.1)

# 2. Operating Mode(레지스터 33)를 0 (위치 제어 모드)으로 변경
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 33, 0)
time.sleep(0.1)

# 3. ★★★ 핵심 해결책: 현재 위치를 강제로 중앙값(2048)으로 영점 조절 (Calibration) ★★★
# Feetech 모터는 레지스터 40번에 128을 쓰면 현재 물리적 위치를 2048로 리셋합니다.
print("✅ 현재 꽉 끼어있는 이 위치를 강제로 '중앙(2048)'으로 재설정합니다!")
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 128)
time.sleep(0.5)

# 4. 토크 다시 켜기
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)

# 5. 현재 위치 읽어오기 (이제 무조건 2048 근처로 나와야 함)
current_raw = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
print(f"새로운 영점 조절 후 Raw 위치: {current_raw}")

# 6. 그리퍼를 살짝만 이동 (플라스틱이 걸리지 않도록 안전하게 +300만 이동)
target = current_raw + 300
print(f"👉 3초 뒤 {target}으로 살짝 벌어집니다.")
time.sleep(3)

# 속도 제한 걸기 (레지스터 46: Goal Speed) - 천천히 부드럽게
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 46, 500)
# 목표 위치 전송
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)

time.sleep(2)
print("원래 위치로 복귀합니다.")
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, current_raw)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
portHandler.closePort()
print("최종 복구 테스트 종료!")
