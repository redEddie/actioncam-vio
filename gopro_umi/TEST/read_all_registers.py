import time, sys
try:
    import scservo_sdk as scs
except ImportError:
    sys.exit("scservo_sdk 없음")

portHandler = scs.PortHandler("/dev/ttyACM0")
packetHandler = scs.PacketHandler(0)  # ★ STS3215는 프로토콜 0번입니다!
portHandler.openPort()
portHandler.setBaudRate(1000000)
MOTOR_ID = 6

print("=" * 60)
print("🔍 ST3215 EEPROM 전체 레지스터 정밀 판독")
print("=" * 60)

def read_1byte(addr):
    val, result, err = packetHandler.read1ByteTxRx(portHandler, MOTOR_ID, addr)
    if result != scs.COMM_SUCCESS:
        return f"읽기실패({result})"
    return val

def read_2byte(addr):
    val, result, err = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, addr)
    if result != scs.COMM_SUCCESS:
        return f"읽기실패({result})"
    return val

# Sign-Magnitude 디코딩 함수 (15비트)
def decode_sign_magnitude_15(raw):
    if isinstance(raw, str):
        return raw
    sign = (raw >> 15) & 1      # 최상위 비트 = 부호 (0: +, 1: -)
    magnitude = raw & 0x7FFF    # 하위 15비트 = 크기
    if sign == 1:
        return -magnitude
    return magnitude

# Sign-Magnitude 디코딩 함수 (10비트) - Load용
def decode_sign_magnitude_10(raw):
    if isinstance(raw, str):
        return raw
    sign = (raw >> 10) & 1
    magnitude = raw & 0x3FF
    if sign == 1:
        return -magnitude
    return magnitude

print("\n[EPROM 영역 - 저장된 설정값]")
print(f"  Firmware Major Version (0):  {read_1byte(0)}")
print(f"  Firmware Minor Version (1):  {read_1byte(1)}")
print(f"  Model Number (3):            {read_2byte(3)}")
print(f"  ID (5):                      {read_1byte(5)}")
print(f"  Baud Rate (6):               {read_1byte(6)}")
print(f"  Return Delay Time (7):       {read_1byte(7)}")
print(f"  Response Status Level (8):   {read_1byte(8)}")

min_pos = read_2byte(9)
max_pos = read_2byte(11)
print(f"  ★ Min Position Limit (9):    {min_pos} (Raw) -> 디코딩: {decode_sign_magnitude_15(min_pos)}")
print(f"  ★ Max Position Limit (11):   {max_pos} (Raw) -> 디코딩: {decode_sign_magnitude_15(max_pos)}")

print(f"  Max Temperature Limit (13):  {read_1byte(13)} 도")
print(f"  Max Voltage Limit (14):      {read_1byte(14)}")
print(f"  Min Voltage Limit (15):      {read_1byte(15)}")
print(f"  Max Torque Limit (16):       {read_2byte(16)}")
print(f"  Operating Mode (33):         {read_1byte(33)} (0=위치제어, 1=속도, 2=PWM, 3=스텝)")

homing = read_2byte(31)
print(f"  Homing Offset (31):          {homing} (Raw) -> 디코딩: {decode_sign_magnitude_15(homing) if isinstance(homing, int) else homing}")

print(f"  Protective Torque (34):      {read_1byte(34)}")
print(f"  Protection Time (35):        {read_1byte(35)}")
print(f"  Overload Torque (36):        {read_1byte(36)}")
print(f"  Lock (55):                   {read_1byte(55)}")

print("\n[SRAM 영역 - 현재 실시간 상태]")
print(f"  Torque Enable (40):          {read_1byte(40)}")

goal_pos = read_2byte(42)
print(f"  Goal Position (42):          {goal_pos} (Raw) -> 디코딩: {decode_sign_magnitude_15(goal_pos)}")

present_pos = read_2byte(56)
print(f"  ★ Present Position (56):     {present_pos} (Raw) -> 디코딩: {decode_sign_magnitude_15(present_pos)}")

present_vel = read_2byte(58)
print(f"  Present Velocity (58):       {present_vel} (Raw) -> 디코딩: {decode_sign_magnitude_15(present_vel)}")

present_load = read_2byte(60)
print(f"  Present Load (60):           {present_load} (Raw) -> 디코딩: {decode_sign_magnitude_10(present_load)}")

print(f"  Present Voltage (62):        {read_1byte(62)}")
print(f"  Present Temperature (63):    {read_1byte(63)} 도")
print(f"  Status (65):                 {read_1byte(65)}")
print(f"  Moving (66):                 {read_1byte(66)}")

present_current = read_2byte(69)
print(f"  Present Current (69):        {present_current}")

portHandler.closePort()

print("\n" + "=" * 60)
print("📋 판독 완료! 위 결과를 캡처해서 판매자에게 보여주세요.")
print("=" * 60)
