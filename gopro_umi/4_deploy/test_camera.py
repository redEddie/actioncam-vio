import cv2
import time

def test_camera(index=0):
    print(f"[{index}번 카메라 연결 시도 중...]")
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
        
    if not cap.isOpened():
        print(f"🚨 실패: {index}번 카메라를 열 수 없습니다.")
        return False
        
    # 캡처 보드의 고질적인 reshape 에러 해결을 위해 해상도/포맷 강제 지정
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 캡처 보드 워밍업 (초기 쓰레기 프레임 버리기)
    for _ in range(5):
        cap.grab()
        
    ret, frame = cap.read()
    if ret:
        print(f"✅ 성공! 카메라 연결 정상 작동 중.")
        print(f"📸 해상도: {frame.shape[1]}x{frame.shape[0]} (Channels: {frame.shape[2]})")
        cv2.imwrite(f"camera_test_{index}.jpg", frame)
        print(f"💾 현재 화면을 camera_test_{index}.jpg 로 저장했습니다.")
        cap.release()
        return True
    else:
        print(f"🚨 실패: 카메라는 인식되나 영상을 읽어올 수 없습니다.")
        cap.release()
        return False

if __name__ == "__main__":
    print("="*50)
    print("🎥 캡처보드(카메라) 연결 테스트")
    print("="*50)
    
    # 0번부터 2번까지 다 찔러보기
    success = False
    for i in range(3):
        if test_camera(i):
            success = True
            break
        time.sleep(1)
        
    if not success:
        print("\n❌ 모든 카메라 인덱스 연결 실패.")
        print("💡 확인 사항:")
        print("1. 권한 문제가 남았을 수 있으니 컴퓨터를 껐다 켜보세요 (또는 로그아웃 후 로그인).")
        print("2. 다른 프로그램(OBS 등)이 캡처보드를 사용 중이면 꺼주세요.")
