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

print("==================================================")
print("🚨 하드웨어 파손(엔코더 자석/기어) 확정 테스트 🚨")
print("==================================================\n")

# 토크 끄기 (수동 회전 테스트를 위함)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
time.sleep(0.1)

print("✅ [테스트 1] 수동 회전 센서 인식 테스트")
print("지금부터 10초 동안 모터의 톱니바퀴(축)를 손으로 잡고 양옆으로 마구 돌려보세요!")
print("만약 기어비 때문에 손으로 돌리기 너무 뻑뻑하다면, 최대한 힘을 줘서 조금이라도 꺾어보세요.")
print("시작합니다...\n")

start_time = time.time()
while time.time() - start_time < 10:
    pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    print(f"현재 센서가 읽고 있는 각도: {pos}")
    time.sleep(0.5)

print("\n--------------------------------------------------")
print("✅ [테스트 2] 내부 헛돔(Slipping) 감지 테스트")
print("이제 모터에 전기를 넣고 강제로 회전시켜 보겠습니다.")
print("외부 축은 가만히 있는데 내부 숫자가 미친듯이 튀는지 확인합니다.\n")

packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1) # 토크 ON
start_pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]

target = start_pos + 2000
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)

for i in range(10):
    pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    speed, _, _ = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 58)
    # Speed 값이 음수일 수 있으므로 15비트로 변환 (최상위 비트는 방향)
    real_speed = speed & 0x7FFF
    
    print(f"[{i+1}/10] 현재 각도: {pos} | 내부 회전 속도: {real_speed}")
    time.sleep(0.2)

packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
portHandler.closePort()

print("\n==================================================")
print("🛑 [결과 판독법]")
print("1. 손으로 축을 분명히 돌렸는데 [테스트 1]의 각도 숫자가 전혀 안 변했나요?")
print("2. [테스트 2]에서 눈으로 볼 땐 축이 안 돌았는데, 각도가 몇백~몇천 씩 튀고 내부 회전 속도가 잡히나요?")
print("=> 위 두 가지 중 하나라도 해당된다면 100% 하드웨어(자석 센서 또는 기어) 파손입니다!")
