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

print("🔍 모터의 Min / Max 각도 제한(Limit) 설정을 확인합니다...\n")

def read_2byte(addr):
    val, result, err = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, addr)
    return val

# 레지스터 9: Min Angle Limit
# 레지스터 11: Max Angle Limit
min_limit = read_2byte(9)
max_limit = read_2byte(11)

print(f"▶ 현재 설정된 최소 각도 제한 (Min Limit, 주소 9): {min_limit}")
print(f"▶ 현재 설정된 최대 각도 제한 (Max Limit, 주소 11): {max_limit}")

print("\n--------------------------------------------------")
if min_limit == max_limit or min_limit > max_limit or max_limit == 0:
    print("🚨 경고: Min/Max 제한값이 비정상적으로 꼬여있습니다! 이것 때문에 안 움직였을 수 있습니다!")
    print("🛠 정상값(Min: 0, Max: 4095)으로 강제 초기화를 진행합니다...")
    
    # EEPROM Lock 해제 (필요할 수 있음)
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 48, 0)
    time.sleep(0.1)
    
    # 정상 범위 쓰기
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 9, 0)
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 11, 4095)
    time.sleep(0.1)
    
    new_min = read_2byte(9)
    new_max = read_2byte(11)
    print(f"✅ 초기화 완료! 새로운 Min: {new_min}, Max: {new_max}")
else:
    print("✅ Min/Max 제한 설정은 아주 정상적입니다. (범위 설정 문제 아님)")

portHandler.closePort()
print("==================================================")
