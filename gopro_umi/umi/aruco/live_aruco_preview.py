#!/usr/bin/env python3
"""
LIVE ArUco detection preview from a connected RealSense (D405 / D435i / D455).
Use it while designing the camera mount: point the camera at your markers and see
in real time which IDs are detected, at what distance, and whether they stay in FOV.

Draws detected marker outlines + ID + distance (m). Detection runs on the LEFT IR
image by default (what the SLAM pipeline uses); pass --color to use the RGB image.

    conda activate umi
    python aruco/live_aruco_preview.py                 # auto-detect camera, IR-left
    python aruco/live_aruco_preview.py --color         # use color stream
    python aruco/live_aruco_preview.py --no-window      # headless: print stats only

Keys (window mode): q = quit, s = save snapshot to /tmp.
Over SSH the window goes to the physical desktop (DISPLAY :1).
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import cv2
import yaml
import pyrealsense2 as rs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_size_map(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    sm = {int(k): float(v) for k, v in cfg["marker_size_map"].items() if k != "default"}
    grip = (cfg.get("gripper_left_finger_id"), cfg.get("gripper_right_finger_id"))
    offset = float(cfg.get("gripper_width_offset_m", 0.0))
    return cfg["aruco_dict"]["predefined"], sm, float(cfg["marker_size_map"].get("default", 0.16)), grip, offset


def marker_obj_points(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", action="store_true", help="detect on RGB instead of left IR")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "aruco_config.yaml"))
    ap.add_argument("--no-window", action="store_true", help="headless: print stats only")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":1"

    dict_name, size_map, default_size, (grip_l, grip_r), grip_offset = load_size_map(a.config)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        print("No RealSense device connected."); return 1
    dev = ctx.query_devices()[0]
    name = dev.get_info(rs.camera_info.name)

    cfg = rs.config()
    if a.color:
        cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, 30)
        stream = rs.stream.color
    else:
        cfg.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, 30)
        stream = rs.stream.infrared
    pipe = rs.pipeline()
    prof = pipe.start(cfg)

    # IR-left: turn emitter OFF so the projector dots don't corrupt the marker
    if not a.color:
        try:
            ds = prof.get_device().first_depth_sensor()
            if ds.supports(rs.option.emitter_enabled):
                ds.set_option(rs.option.emitter_enabled, 0)
        except Exception:
            pass

    vsp = prof.get_stream(stream if a.color else rs.stream.infrared, 1 if not a.color else 0)
    intr = vsp.as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
    dist = np.array(intr.coeffs, float).reshape(-1)

    print(f"Device: {name} | stream: {'color' if a.color else 'IR-left'} {a.width}x{a.height}")
    print("Watching for markers... (Ctrl-C to stop)")
    last = time.time(); seen_total = {}
    try:
        while True:
            fs = pipe.wait_for_frames()
            fr = fs.get_color_frame() if a.color else fs.get_infrared_frame(1)
            if not fr:
                continue
            img = np.asanyarray(fr.get_data())
            if a.color:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # detection needs grayscale
                vis = img.copy()                              # but display the RGB frame
            else:
                gray = img
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            corners, ids, _ = detector.detectMarkers(gray)
            info = []; tvec_by_id = {}
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                for c, i in zip(corners, ids.flatten()):
                    i = int(i); size = size_map.get(i, default_size)
                    ok, rvec, tvec = cv2.solvePnP(marker_obj_points(size),
                                                  c.reshape(-1, 2).astype(np.float64), K, dist,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    d = float(np.linalg.norm(tvec)) if ok else -1
                    info.append((i, d)); seen_total[i] = seen_total.get(i, 0) + 1
                    if ok:
                        tvec_by_id[i] = tvec.squeeze()
                    ctr = c.reshape(-1, 2).mean(0).astype(int)
                    cv2.putText(vis, f"id{i} {d:.2f}m", tuple(ctr), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
            # live gripper width = x-separation of left/right finger markers (camera frame)
            grip_w = None
            if grip_l in tvec_by_id and grip_r in tvec_by_id:
                raw = float(tvec_by_id[grip_r][0] - tvec_by_id[grip_l][0])
                grip_w = raw - grip_offset                      # calibrated width
                zl, zr = tvec_by_id[grip_l][2], tvec_by_id[grip_r][2]
                cv2.putText(vis, f"GRIP WIDTH: {grip_w*100:.1f} cm  (raw {raw*100:+.1f}, Lz {zl*100:.1f} Rz {zr*100:.1f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if time.time() - last > 0.5:
                last = time.time()
                txt = "  ".join(f"id{i}:{d:.2f}m" for i, d in sorted(info)) or "(none)"
                wtxt = f" | GRIP {grip_w*100:+.1f}cm" if grip_w is not None else ""
                print(f"\rdetected: {txt}{wtxt}        ", end="", flush=True)
            if not a.no_window:
                cv2.imshow("ArUco preview (q quit, s save)", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('s'):
                    p = f"/tmp/aruco_snap_{int(time.time())}.png"; cv2.imwrite(p, vis); print(f"\nsaved {p}")
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if not a.no_window:
            cv2.destroyAllWindows()
        print(f"\nper-id total detections: {dict(sorted(seen_total.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
