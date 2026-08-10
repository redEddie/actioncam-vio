import time, sys
try:
    import scservo_sdk as scs
except ImportError:
    sys.exit("scservo_sdk 없음")

portHandler = scs.PortHandler("/dev/ttyACM0")
packetHandler = scs.PacketHandler(0)  # STS 시리즈는 프로토콜 0
portHandler.openPort()
portHandler.setBaudRate(1000000)

print("🔍 새 STS3250 모터를 스캔합니다...\n")

# 0~10번까지 스캔해서 어떤 ID로 되어있는지 찾기
found_id = None
for scan_id in range(11):
    model, result, err = packetHandler.read2ByteTxRx(portHandler, scan_id, 3)
    if result == scs.COMM_SUCCESS:
        print(f"  ✅ ID {scan_id} 발견! (Model Number: {model})")
        found_id = scan_id

if found_id is None:
    print("❌ 모터를 찾을 수 없습니다. USB 연결과 chmod를 확인하세요.")
    portHandler.closePort()
    sys.exit(1)

if found_id == 6:
    print(f"\n이미 ID 6번으로 설정되어 있습니다! 변경할 필요 없습니다.")
else:
    print(f"\n현재 ID: {found_id} → 6번으로 변경합니다...")
    
    # EEPROM Lock 해제
    packetHandler.write1ByteTxRx(portHandler, found_id, 55, 0)
    time.sleep(0.1)
    
    # ID 변경 (주소 5, 1바이트)
    packetHandler.write1ByteTxRx(portHandler, found_id, 5, 6)
    time.sleep(0.3)
    
    # 변경 확인
    model, result, err = packetHandler.read2ByteTxRx(portHandler, 6, 3)
    if result == scs.COMM_SUCCESS:
        print(f"  ✅ ID 6번으로 변경 성공! (Model Number: {model})")
    else:
        print("  ❌ 변경 실패! 다시 시도해주세요.")

# ID 6번으로 이동 테스트
print("\n🎯 ID 6번 모터 이동 테스트!")
packetHandler.write1ByteTxRx(portHandler, 6, 40, 1)  # 토크 ON
current = packetHandler.read2ByteTxRx(portHandler, 6, 56)[0]
target = current + 500
if target > 4095:
    target = current - 500

print(f"  현재: {current} → 목표: {target}")
packetHandler.write2ByteTxRx(portHandler, 6, 42, target)
time.sleep(1)

new_pos = packetHandler.read2ByteTxRx(portHandler, 6, 56)[0]
print(f"  이동 후: {new_pos}")

if abs(new_pos - target) < 50:
    print("  🎉 모터 정상 동작 확인!")
else:
    print(f"  ⚠️ 목표와 차이: {abs(new_pos - target)}")

# 원래 위치로 복귀
packetHandler.write2ByteTxRx(portHandler, 6, 42, current)
time.sleep(1)
packetHandler.write1ByteTxRx(portHandler, 6, 40, 0)  # 토크 OFF
portHandler.closePort()
print("✅ 설정 완료!")
