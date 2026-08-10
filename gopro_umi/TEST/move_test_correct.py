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

print("⚠️ 주의: 기계적 한계를 강제로 넘어가서 톱니가 갈려나갈 수 있습니다!\n")

# 1. 토크 완전히 끄고 시작
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
time.sleep(0.5)

# 2. 현재 위치 읽기
current = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
print(f"현재 위치: {current}")

# 3. 토크 ON
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, current)
packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)
time.sleep(0.5)

# ==========================================
# 1단계: +2000 이동
# ==========================================
target1 = current + 2000
if target1 > 4095:
    target1 = 4095

print(f"\n👉 1단계: [ + 방향 ] 으로 2000만큼 크게 회전합니다! ({current} → {target1})")
time.sleep(1)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target1)

for i in range(15):
    pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    print(f"  [+] 현재 위치: {pos} / 목표: {target1}")
    time.sleep(0.15)

# ==========================================
# 2단계: -1000 되돌아오기 (target1 위치에서 -1000)
# ==========================================
target2 = target1 - 1000
if target2 < 0:
    target2 = 0

print(f"\n👉 2단계: [ - 방향 ] 으로 1000만큼 되돌아옵니다! ({target1} → {target2})")
time.sleep(1)
packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target2)

for i in range(15):
    pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    print(f"  [-] 현재 위치: {pos} / 목표: {target2}")
    time.sleep(0.15)

# 토크 유지 종료 (원래 위치로 안 돌아감)
print("\n✅ 이동 완료! (모터 토크는 켜둔 채로 유지합니다)")
portHandler.closePort()
