#!/usr/bin/env bash
# GoPro HDMI(캡처 동글) 녹화 파이프라인.
#
# 문제: MacroSilicon 동글은 USB 캡처를 열 때 HDMI 핫플러그를 발생시켜
# GoPro가 그때서야 HDMI 출력을 초기화한다. 캡처를 닫았다 다시 열면
# GoPro 상태(클린 출력/모니터 모드)가 리셋될 수 있다.
#
# 해결: 장치를 "한 번만" 열고 끝까지 유지한다 —
#   - 시작 즉시 파일 녹화 + 프리뷰 창 동시 출력 (ffmpeg tee)
#   - 프리뷰 창을 닫아도(q) 녹화는 계속됨 (onfail=ignore)
#   - GoPro 설정을 마친 뒤 이 터미널에서 Enter → "시작 마크"가 사이드카에 기록
#   - Ctrl+C 로 종료
#
# 사용:  ./scripts/hdmi_record.sh data/hero7black/hdmi_board.mkv
# 트림:  ffmpeg -ss <마크초> -i out.mkv -c copy trimmed.mkv
#        (마크초 = marks 파일의 mark - start 차이)
set -u
DEV=${HDMI_DEV:-/dev/v4l/by-id/usb-MACROSILICON_USB3._0_capture-video-index0}
OUT=${1:?사용법: hdmi_record.sh <출력.mkv>}
MARKS="${OUT%.*}.marks.txt"
mkdir -p "$(dirname "$OUT")"

if fuser "$DEV" >/dev/null 2>&1; then
    echo "오류: $DEV 를 다른 프로세스가 사용 중입니다:" >&2
    fuser -v "$DEV" >&2
    exit 1
fi

echo "start $(date +%s.%N)" > "$MARKS"
ffmpeg -hide_banner -loglevel error \
    -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 60 -i "$DEV" \
    -c copy -map 0 -f tee \
    "[f=matroska]${OUT}|[f=nut:onfail=ignore]pipe:1" \
    | ffplay -loglevel error -window_title "GoPro HDMI preview (q: 프리뷰만 종료)" - &
PIPE_PID=$!

cleanup() {
    # 파이프라인 전체 종료 (ffmpeg가 mkv를 정상 마감하도록 INT 전달)
    pkill -INT -P $$ ffmpeg 2>/dev/null
    kill -INT $PIPE_PID 2>/dev/null
    wait 2>/dev/null
    echo
    echo "저장: $OUT"
    echo "마크: $MARKS"
    exit 0
}
trap cleanup INT TERM

echo "===================================================================="
echo " 녹화가 이미 시작됐습니다 (프리뷰 창 확인)."
echo " 1. GoPro에서 HDMI 클린 출력 / 모니터 모드 설정을 완료하세요."
echo " 2. 준비되면 여기서 Enter → 시작 마크 기록 (여러 번 가능)."
echo " 3. 촬영이 끝나면 Ctrl+C."
echo "===================================================================="
while read -r _; do
    echo "mark $(date +%s.%N)" >> "$MARKS"
    echo "  ✓ 마크 기록 ($(date +%T))"
done
cleanup
