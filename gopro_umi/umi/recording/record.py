#!/usr/bin/env python3
"""
Device-agnostic RealSense recorder for the UMI pipeline.

Auto-detects the connected camera, loads the matching profile from
configs/cameras/*.yaml, enables the declared streams + sensor options, and
records everything to a lossless librealsense .bag (replayable later by the
same rs2 pipeline code, no ROS needed).

Usage:
    conda activate umi
    python recording/record.py                 # auto-detect + auto-name
    python recording/record.py --tag mapping    # filename tag (e.g. mapping vs demo)
    python recording/record.py --profile configs/cameras/d435i.yaml
    python recording/record.py --list           # just list connected devices

Stop recording with Ctrl-C.
"""
from __future__ import annotations
import argparse
import datetime as dt
import glob
import os
import signal
import sys
import time

import pyrealsense2 as rs
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERAS_DIR = os.path.join(REPO, "configs", "cameras")
DEFAULT_OUT = os.path.join(REPO, "data", "recordings")

# profile string -> rs enum
FORMATS = {
    "y8": rs.format.y8, "y16": rs.format.y16, "z16": rs.format.z16,
    "bgr8": rs.format.bgr8, "rgb8": rs.format.rgb8, "rgba8": rs.format.rgba8,
    "bgra8": rs.format.bgra8, "yuyv": rs.format.yuyv,
    "motion_xyz32f": rs.format.motion_xyz32f,
}
VIDEO_STREAMS = {
    "infrared": rs.stream.infrared, "color": rs.stream.color, "depth": rs.stream.depth,
}
MOTION_STREAMS = {"accel": rs.stream.accel, "gyro": rs.stream.gyro}
SENSOR_KEY_TO_NAME = {
    "stereo_module": "Stereo Module",
    "rgb_camera": "RGB Camera",
    "motion_module": "Motion Module",
}


def list_devices() -> list[rs.device]:
    return list(rs.context().query_devices())


def find_profile_for_device(dev: rs.device) -> str | None:
    """Return the camera profile yaml whose device.name matches this device."""
    name = dev.get_info(rs.camera_info.name)
    for path in sorted(glob.glob(os.path.join(CAMERAS_DIR, "*.yaml"))):
        with open(path) as f:
            prof = yaml.safe_load(f)
        key = str(prof.get("device", {}).get("name", "")).lower()
        if key and key in name.lower():
            return path
    return None


def build_config(prof: dict, serial: str, bag_path: str, preview: str = "none") -> rs.config:
    cfg = rs.config()
    cfg.enable_device(serial)
    streams = prof.get("streams", {})
    # --preview color needs a color stream; add one if the profile lacks it
    # (it will also be recorded into the .db3).
    if preview == "color" and "color" not in streams:
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    for sname, entries in streams.items():
        if entries is None:
            continue
        if sname in VIDEO_STREAMS:
            st = VIDEO_STREAMS[sname]
            for e in entries:
                cfg.enable_stream(st, int(e["index"]), int(e["width"]),
                                  int(e["height"]), FORMATS[e["format"]], int(e["fps"]))
        elif sname in MOTION_STREAMS:
            st = MOTION_STREAMS[sname]
            for e in entries:
                cfg.enable_stream(st, FORMATS.get("motion_xyz32f"), int(e["fps"]))
        else:
            print(f"  [warn] unknown stream group '{sname}' in profile — skipping")
    cfg.enable_record_to_file(bag_path)
    return cfg


def apply_options(dev: rs.device, prof: dict) -> None:
    options = prof.get("options", {}) or {}
    name_to_sensor = {s.get_info(rs.camera_info.name): s for s in dev.query_sensors()}
    for skey, opts in options.items():
        sensor_name = SENSOR_KEY_TO_NAME.get(skey, skey)
        sensor = name_to_sensor.get(sensor_name)
        if sensor is None:
            print(f"  [warn] sensor '{sensor_name}' not present — skipping its options")
            continue
        for okey, oval in (opts or {}).items():
            opt = getattr(rs.option, okey, None)
            if opt is None or not sensor.supports(opt):
                print(f"  [warn] option '{okey}' unsupported on {sensor_name} — skipping")
                continue
            sensor.set_option(opt, float(oval))
            print(f"  set {sensor_name}.{okey} = {oval}")


def human_size(path: str) -> str:
    try:
        b = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="RealSense .bag recorder (UMI pipeline)")
    ap.add_argument("--profile", help="camera profile yaml (default: auto-detect)")
    ap.add_argument("--serial", help="device serial (default: first connected)")
    ap.add_argument("--tag", default="rec", help="filename tag, e.g. mapping/demo")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--list", action="store_true", help="list devices and exit")
    ap.add_argument("--preview", choices=["none", "ir", "color"], default="none",
                    help="live window while recording (color also gets recorded)")
    args = ap.parse_args()

    devs = list_devices()
    if args.list or not devs:
        print(f"connected devices: {len(devs)}")
        for d in devs:
            print(f" - {d.get_info(rs.camera_info.name)} "
                  f"| SN {d.get_info(rs.camera_info.serial_number)} "
                  f"| {d.get_info(rs.camera_info.product_line)}")
        if not devs:
            print("No RealSense device detected. Plug one in (USB3) and retry.")
            return 1
        if args.list:
            return 0

    # pick device
    dev = devs[0]
    if args.serial:
        match = [d for d in devs if d.get_info(rs.camera_info.serial_number) == args.serial]
        if not match:
            print(f"serial {args.serial} not found")
            return 1
        dev = match[0]
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)

    # pick profile
    profile_path = args.profile or find_profile_for_device(dev)
    if not profile_path:
        print(f"No camera profile matches '{name}'. Create one in {CAMERAS_DIR}.")
        return 1
    with open(profile_path) as f:
        prof = yaml.safe_load(f)
    print(f"Device : {name} (SN {serial})")
    print(f"Profile: {os.path.relpath(profile_path, REPO)} "
          f"(slam_mode={prof.get('slam_mode')}, has_imu={prof.get('has_imu')})")

    os.makedirs(args.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pname = os.path.splitext(os.path.basename(profile_path))[0]
    # librealsense >=2.58 records to a SQLite .db3 (rosbag2-style), not the old .bag
    bag_path = os.path.join(args.out, f"{pname}_{args.tag}_{stamp}.db3")

    pipe = rs.pipeline()
    cfg = build_config(prof, serial, bag_path, args.preview)

    preview = args.preview
    if preview != "none":
        import numpy as np, cv2
        os.environ.setdefault("DISPLAY", ":1")   # show on physical desktop over SSH
        win = f"record preview ({preview}) - q to stop"

    print(f"\nRecording -> {bag_path}")
    pipe_profile = pipe.start(cfg)
    # options must be applied on the running device
    apply_options(pipe_profile.get_device(), prof)
    print(f"Recording... press Ctrl-C{' or q in window' if preview!='none' else ''} to stop.\n")

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    t0 = time.time()
    n = 0
    try:
        while not stop["flag"]:
            try:
                frames = pipe.wait_for_frames(2000)
                n += 1
            except RuntimeError:
                # motion-only intervals / timeout — keep going
                continue
            if preview != "none":
                fr = frames.get_color_frame() if preview == "color" else frames.get_infrared_frame(1)
                if fr:
                    img = np.asanyarray(fr.get_data())
                    cv2.imshow(win, img)
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        stop["flag"] = True
            if n % 30 == 0 and n:
                el = time.time() - t0
                print(f"\r  {el:6.1f}s  framesets={n}  size={human_size(bag_path)}",
                      end="", flush=True)
    finally:
        pipe.stop()
        if preview != "none":
            import cv2; cv2.destroyAllWindows()
        el = time.time() - t0
        print(f"\n\nStopped. duration={el:.1f}s  file={bag_path}  size={human_size(bag_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
