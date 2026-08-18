def main():
    import time
    import sys
    import tty
    import termios
    try:
        import scservo_sdk as scs
    except ImportError:
        sys.exit("scservo_sdk 없음")
    
    def get_char():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    portHandler = scs.PortHandler("/dev/ttyACM0")
    packetHandler = scs.PacketHandler(0)  # 프로토콜 0번
    portHandler.openPort()
    portHandler.setBaudRate(1000000)
    MOTOR_ID = 6
    
    # 토크 ON 및 현재 위치 동기화
    current = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
    packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, current)
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 1)
    
    target = current
    
    print("=======================================")
    print("🎮 키보드 실시간 조종 모드 시작!")
    print(f"현재 시작 위치: {current}")
    print("  [q] : -50 이동")
    print("  [e] : +50 이동")
    print("  [ESC] : 프로그램 종료")
    print("=======================================\n")
    
    try:
        while True:
            ch = get_char()
            if ch == '\x1b':  # ESC
                break
            elif ch in ['q', 'Q']:
                target -= 50
            elif ch in ['e', 'E']:
                target += 50
            else:
                continue
    
            # 안전 범위 제한
            if target < 0: target = 0
            if target > 4095: target = 4095
    
            # 모터로 목표 위치 전송
            packetHandler.write2ByteTxRx(portHandler, MOTOR_ID, 42, target)
            
            # 아주 살짝 대기 후 실제 위치와 부하 읽기
            time.sleep(0.05)
            real_pos = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 56)[0]
            load = packetHandler.read2ByteTxRx(portHandler, MOTOR_ID, 60)[0]
            
            # 부하값(Sign-Magnitude) 디코딩
            if load >= 1024:
                load = -(load - 1024)
    
            # 화면에 한 줄로 계속 덮어쓰면서 출력
            sys.stdout.write(f"\r▶ 목표: {target:4d} | 실제 위치: {real_pos:4d} | 부하(Load): {load:5d}   ")
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        pass
    
    print("\n\n종료합니다. 모터 토크를 해제합니다.")
    packetHandler.write1ByteTxRx(portHandler, MOTOR_ID, 40, 0)
    portHandler.closePort()


if __name__ == "__main__":
    main()
