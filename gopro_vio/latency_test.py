"""End-to-end camera latency measurement (glass-to-host).

Shows a fullscreen pattern encoding the draw time (ArUco corner markers for
localization + a 24-bit binary strip encoding milliseconds), captures the
camera pointed at the monitor, decodes the visible timestamp from each
captured frame and compares it with the frame's host arrival time:

    latency = t_arrival(host clock) - t_drawn(same host clock)

Includes monitor response (~10-20 ms), camera exposure/readout, device
pipeline, USB transfer and decode — i.e. exactly the latency a policy sees.
Display and capture run in one process on one monotonic clock.

Usage (point the camera at the monitor so the whole pattern is visible):
  python -m gopro_vio.latency_test /dev/v4l/by-id/usb-Arashi_...-index0 \
      --size 1920x1080 --fps 30 --seconds 20
"""
from __future__ import annotations

import argparse
import queue
import threading
import time

import cv2
import numpy as np

BITS = 24
DICT = cv2.aruco.DICT_4X4_50


class Grabber(threading.Thread):
    """Drain the capture stream at full rate on a separate thread.

    The measurement loop is slower than the stream (detect + encode take tens
    of ms); reading in the same loop lets frames age in the driver queue and
    inflates the measured latency by up to one loop period. This thread keeps
    only the newest frame, timestamped at arrival, so the main loop always
    consumes a fresh (<= 1 frame period old) image.
    """

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame, self.t, self.seq, self.n = None, 0.0, 0, 0
        self.running = True

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            t = time.monotonic()
            if not ok:
                continue
            self.n += 1
            with self.lock:
                self.frame, self.t, self.seq = f, t, self.seq + 1

    def latest(self, last_seq):
        with self.lock:
            if self.seq == last_seq or self.frame is None:
                return None
            return self.t, self.frame, self.seq


def make_display(t_ms: int, w=1280, h=720):
    img = np.full((h, w, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    ms = 140  # marker size
    pos = {0: (40, 40), 1: (w - 40 - ms, 40),
           2: (w - 40 - ms, h - 40 - ms), 3: (40, h - 40 - ms)}
    for mid, (x, y) in pos.items():
        img[y:y+ms, x:x+ms] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(d, mid, ms), cv2.COLOR_GRAY2BGR)
    # local anchors flanking the strip: curved monitors + fisheye break a
    # global planar homography, so markers 4/5 sit right next to the strip
    yc = h // 2
    am = 150
    img[yc-am//2:yc+am//2, 40:40+am] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 4, am), cv2.COLOR_GRAY2BGR)
    img[yc-am//2:yc+am//2, w-40-am:w-40] = cv2.cvtColor(
        cv2.aruco.generateImageMarker(d, 5, am), cv2.COLOR_GRAY2BGR)
    # binary strip: guard cells (black, white) + BITS payload cells, MSB first
    ncell = BITS + 2
    x0, x1 = 240, w - 240
    cw = (x1 - x0) // ncell
    y0, y1 = yc - 80, yc + 80
    cells = [1] + [(t_ms >> (BITS - 1 - b)) & 1 for b in range(BITS)] + [0]
    for b, bit in enumerate(cells):
        img[y0:y1, x0 + b*cw: x0 + (b+1)*cw] = 0 if bit else 255
    cv2.rectangle(img, (x0, y0), (x0 + ncell*cw, y1), (128, 128, 128), 2)
    cv2.putText(img, f"{t_ms} ms", (x0, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    return img


def decode_strip(frame, det):
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None:
        return None
    got = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    if not (4 in got and 5 in got):
        return None
    # local homography from the two strip-flanking anchors (8 corners):
    # spans only the strip neighborhood, tolerant to curved monitors/fisheye
    w, h = 1280, 720
    yc = h // 2
    am = 150
    def sq(x, y):  # display-space marker corners TL,TR,BR,BL
        return [[x, y], [x+am, y], [x+am, y+am], [x, y+am]]
    src = np.float64(np.vstack([got[4], got[5]]))
    dst = np.float64(sq(40, yc-am//2) + sq(w-40-am, yc-am//2))
    Hm, _ = cv2.findHomography(src, dst)
    if Hm is None:
        return None
    inv = np.linalg.inv(Hm)
    ncell = BITS + 2
    x0, x1 = 240, w - 240
    cw = (x1 - x0) // ncell
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    samples = []
    for b in range(ncell):
        cx = x0 + b*cw + cw/2
        p = inv @ np.array([cx, yc, 1.0])
        u, v = int(p[0]/p[2]), int(p[1]/p[2])
        if not (2 <= u < frame.shape[1]-2 and 2 <= v < frame.shape[0]-2):
            return None
        samples.append(int(gray[v-2:v+3, u-2:u+3].mean()))
    black, white = samples[0], samples[-1]    # guard cells
    if white - black < 40:                    # strip not actually visible
        return None
    # strict per-cell classification: HDMI->USB converters can blend two
    # consecutive frames (double-exposure of two patterns); changed bits
    # then read mid-gray — reject such frames instead of mis-decoding
    span = white - black
    dark_thr = black + 0.3 * span
    bright_thr = white - 0.3 * span
    val = 0
    for s in samples[1:-1]:
        if s < dark_thr:
            bit = 1
        elif s > bright_thr:
            bit = 0
        else:
            return None                        # blended/ambiguous frame
        val = (val << 1) | bit
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("device")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--setup", action="store_true",
                    help="측정 전 셋업 단계: 라이브 프리뷰를 보여주고 (장치는 "
                         "계속 열린 채 — HDMI 핫플러그 리셋 방지) 카메라 설정/"
                         "조준을 마친 뒤 프리뷰 창에서 Space/Enter로 측정 시작")
    ap.add_argument("--setup-timeout", type=float, default=300)
    ap.add_argument("--hold-ms", type=float, default=0,
                    help="패턴을 N ms 동안 유지 (프레임 블렌딩이 심한 캡처 경로용; "
                         "측정값 = 지연 + 패턴 나이(0..N 균등) → min이 지연의 추정치)")
    ap.add_argument("--fourcc", default="MJPG",
                    help="캡처 픽셀 포맷 (예: MJPG, YUYV — 장치의 "
                         "v4l2-ctl --list-formats-ext 참조)")
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # don't queue stale frames
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.device}")

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT))

    if args.setup:
        # phase 1: keep the device streaming (GoPro HDMI stays initialized),
        # show what the capture card sees; advance with Space/Enter in-window
        print("[셋업] 카메라 설정(클린 HDMI/모니터 모드) 후 카메라를 모니터로 "
              "조준하고, 프리뷰 창에서 Space 또는 Enter를 누르세요.")
        cv2.namedWindow("setup_preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("setup_preview", 960, 540)
        t_setup = time.monotonic()
        confirmed = False
        while time.monotonic() - t_setup < args.setup_timeout:
            ok, frame = cap.read()
            if not ok:
                continue
            disp = frame.copy()
            cv2.putText(disp, "SETUP: Space/Enter = start measurement",
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.imshow("setup_preview", disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (13, 32):
                confirmed = True
                break
            if k == ord('q'):
                raise SystemExit("셋업 단계에서 종료됨")
        cv2.destroyWindow("setup_preview")
        if not confirmed:
            raise SystemExit("셋업 확인(Space/Enter) 없이 타임아웃 — 측정을 "
                             "시작하지 않고 종료합니다 (무인 측정 방지)")
        print("[셋업 완료] 측정 시작")

    cv2.namedWindow("latency_pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("latency_pattern", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    grab = Grabber(cap)
    grab.start()

    t0 = time.monotonic()
    lat = []
    stats = {"n_frames": 0, "n_markers": 0, "last_frame": None}
    raw_path = "/tmp/latency_raw.avi"
    raw_times = []
    vw = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"MJPG"),
                         args.fps, (W, H))

    # heavy per-frame work (encode + detect + decode, tens of ms) runs on a
    # worker thread so the display loop keeps repainting the pattern every few
    # ms — otherwise the on-screen timestamp itself goes stale by one loop
    # period and inflates the measurement.
    q_frames: queue.Queue = queue.Queue(maxsize=4)
    running = True

    def worker():
        while running or not q_frames.empty():
            try:
                t_arr, t_ms, frame = q_frames.get(timeout=0.2)
            except queue.Empty:
                continue
            stats["n_frames"] += 1
            stats["last_frame"] = frame
            vw.write(frame)
            raw_times.append((t_arr, t_ms))
            c_, i_, _ = det.detectMarkers(frame)
            if i_ is not None and len(i_) >= 4:
                stats["n_markers"] += 1
            seen = decode_strip(frame, det)
            if seen is None:
                continue
            d = t_arr - seen
            if 0 < d < 3000:
                lat.append(d)

    wk = threading.Thread(target=worker, daemon=True)
    wk.start()

    print("카메라가 화면 전체를 보도록 놓으세요. 측정 중... (q로 조기 종료)")
    held_val, held_until = None, -1.0
    seq = 0
    while time.monotonic() - t0 < args.seconds:
        now_ms = (time.monotonic() - t0) * 1000
        if held_val is None or now_ms >= held_until:
            held_val = int(now_ms) & ((1 << BITS) - 1)
            held_until = now_ms + args.hold_ms
        t_ms = held_val if args.hold_ms > 0 else int(now_ms) & ((1 << BITS) - 1)
        cv2.imshow("latency_pattern", make_display(t_ms))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        got = grab.latest(seq)
        if got is None:
            continue
        t_abs, frame, seq = got
        t_arr = (t_abs - t0) * 1000
        try:
            q_frames.put_nowait((t_arr, t_ms, frame))
        except queue.Full:
            pass  # worker is behind; drop — freshness beats coverage here
    grab.running = False
    running = False
    grab.join(timeout=2)
    wk.join(timeout=5)
    cap.release()
    vw.release()
    n_frames, n_markers = stats["n_frames"], stats["n_markers"]
    last_frame = stats["last_frame"]
    grab_fps = grab.n / max(time.monotonic() - t0, 1e-6)
    if grab_fps < 0.9 * args.fps:
        print(f"경고: 캡처 스레드 유효 {grab_fps:.1f} fps < 요청 {args.fps} — "
              "장치가 요청 포맷을 못 내거나 시스템이 과부하일 수 있음")
    np.savetxt("/tmp/latency_raw_times.csv",
               np.array(raw_times), delimiter=",",
               header="t_arrival_ms,t_displayed_ms", comments="")
    print(f"원시 캡처 저장: {raw_path} + /tmp/latency_raw_times.csv "
          "(실패해도 오프라인 재분석 가능)")
    cv2.destroyAllWindows()

    if len(lat) < 10:
        if last_frame is not None:
            cv2.imwrite("/tmp/latency_debug_frame.jpg", last_frame)
        raise SystemExit(
            f"디코딩된 샘플이 부족합니다({len(lat)}) — 캡처 {n_frames}프레임 중 "
            f"마커4개 검출 {n_markers}프레임. 마지막 프레임: /tmp/latency_debug_frame.jpg")
    lat = np.array(lat)
    print(f"\nsamples: {len(lat)} (캡처 {n_frames} 중)")
    print(f"latency(raw): median {np.median(lat):.0f} ms | "
          f"min {lat.min():.0f} | p10 {np.percentile(lat,10):.0f} | "
          f"p90 {np.percentile(lat,90):.0f}")
    if args.hold_ms > 0:
        print(f"hold {args.hold_ms:.0f} ms 보정: latency ≈ min {lat.min():.0f} ms "
              f"~ median-보정 {np.median(lat)-args.hold_ms/2:.0f} ms")
    print("(모니터 응답시간 ~10-20 ms 포함; 표시 루프 양자화 ±1프레임)")


if __name__ == "__main__":
    main()
