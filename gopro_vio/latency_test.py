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
import time

import cv2
import numpy as np

BITS = 24
DICT = cv2.aruco.DICT_4X4_50


def make_display(t_ms: int, w=1280, h=720):
    img = np.full((h, w, 3), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(DICT)
    ms = 140  # marker size
    pos = {0: (40, 40), 1: (w - 40 - ms, 40),
           2: (w - 40 - ms, h - 40 - ms), 3: (40, h - 40 - ms)}
    for mid, (x, y) in pos.items():
        img[y:y+ms, x:x+ms] = cv2.cvtColor(
            cv2.aruco.generateImageMarker(d, mid, ms), cv2.COLOR_GRAY2BGR)
    # binary strip: guard cells (black, white) + BITS payload cells, MSB first
    ncell = BITS + 2
    x0, x1 = 240, w - 240
    cw = (x1 - x0) // ncell
    y0, y1 = h // 2 - 80, h // 2 + 80
    cells = [1] + [(t_ms >> (BITS - 1 - b)) & 1 for b in range(BITS)] + [0]
    for b, bit in enumerate(cells):
        img[y0:y1, x0 + b*cw: x0 + (b+1)*cw] = 0 if bit else 255
    cv2.rectangle(img, (x0, y0), (x0 + ncell*cw, y1), (128, 128, 128), 2)
    cv2.putText(img, f"{t_ms} ms", (x0, y1 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    return img


def decode_strip(frame, det):
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None or len(ids) < 4:
        return None
    got = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    if not all(k in got for k in (0, 1, 2, 3)):
        return None
    # marker outer corners -> display-space homography
    w, h = 1280, 720
    ms = 140
    src = np.float64([got[0][0], got[1][1], got[2][2], got[3][3]])
    dst = np.float64([[40, 40], [w-40, 40], [w-40, h-40], [40, h-40]])
    Hm, _ = cv2.findHomography(src, dst)
    if Hm is None:
        return None
    inv = np.linalg.inv(Hm)
    ncell = BITS + 2
    x0, x1 = 240, w - 240
    cw = (x1 - x0) // ncell
    yc = h // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    samples = []
    for b in range(ncell):
        cx = x0 + b*cw + cw/2
        p = inv @ np.array([cx, yc, 1.0])
        u, v = p[0]/p[2], p[1]/p[2]
        if not (0 <= u < frame.shape[1]-1 and 0 <= v < frame.shape[0]-1):
            return None
        samples.append(int(gray[int(v), int(u)]))
    black, white = samples[0], samples[-1]    # guard cells
    if white - black < 40:                    # strip not actually visible
        return None
    thr = (black + white) / 2
    val = 0
    for s in samples[1:-1]:
        val = (val << 1) | (1 if s < thr else 0)
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("device")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    W, H = map(int, args.size.split("x"))

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # don't queue stale frames
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.device}")

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT))
    cv2.namedWindow("latency_pattern", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("latency_pattern", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    t0 = time.monotonic()
    lat = []
    print("카메라가 화면 전체를 보도록 놓으세요. 측정 중... (q로 조기 종료)")
    while time.monotonic() - t0 < args.seconds:
        t_ms = int((time.monotonic() - t0) * 1000) & ((1 << BITS) - 1)
        cv2.imshow("latency_pattern", make_display(t_ms))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ok, frame = cap.read()
        t_arr = (time.monotonic() - t0) * 1000
        if not ok:
            continue
        seen = decode_strip(frame, det)
        if seen is None:
            continue
        d = t_arr - seen
        if 0 < d < 3000:
            lat.append(d)
    cap.release()
    cv2.destroyAllWindows()

    if len(lat) < 10:
        raise SystemExit(f"디코딩된 샘플이 부족합니다({len(lat)}) — 패턴이 "
                         "화면에 꽉 차게, 초점/거리 조정 후 재시도")
    lat = np.array(lat)
    print(f"\nsamples: {len(lat)}")
    print(f"latency: median {np.median(lat):.0f} ms | "
          f"p10 {np.percentile(lat,10):.0f} | p90 {np.percentile(lat,90):.0f} | "
          f"mean {lat.mean():.0f} ms")
    print("(모니터 응답시간 ~10-20 ms 포함; 표시 루프 양자화 ±1프레임)")


if __name__ == "__main__":
    main()
