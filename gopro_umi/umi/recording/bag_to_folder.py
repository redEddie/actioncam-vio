#!/usr/bin/env python3
"""
Extract a RealSense .db3 recording into a calibration-friendly folder layout
(EuRoC / Kalibr-bagcreater style) for OpenICC / Kalibr and for debugging.

Output:
    <out>/
      cam0/<ts_ns>.png        left  infrared  (idx 1)
      cam1/<ts_ns>.png        right infrared  (idx 2)
      cam2/<ts_ns>.png        color           (if present)
      imu0.csv                ts_ns, gx,gy,gz [rad/s], ax,ay,az [m/s^2]
      intrinsics.json         factory intrinsics + stream extrinsics + IMU intrinsics

The accelerometer is linearly interpolated onto the gyro timestamps so imu0.csv
has one synchronized row per gyro sample (the convention Kalibr expects).

Usage:
    conda activate umi
    python recording/bag_to_folder.py <recording.db3> [--out DIR] [--no-color] [--max-frames N]
    python recording/bag_to_folder.py            # auto-pick newest .db3 in data/recordings
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(REPO, "data", "recordings")

# which infrared index maps to which cam folder
IR_TO_CAM = {1: "cam0", 2: "cam1"}
COLOR_CAM = "cam2"


def ts_ns(frame: rs.frame) -> int:
    # get_timestamp() is in milliseconds (global epoch when global_time enabled)
    return int(round(frame.get_timestamp() * 1e6))


def newest_recording() -> str | None:
    files = sorted(glob.glob(os.path.join(REC_DIR, "*.db3")), key=os.path.getmtime)
    return files[-1] if files else None


def index_profiles(profiles) -> dict:
    """Map recorded stream profiles -> canonical keys."""
    sp = {}
    for p in profiles:
        st = p.stream_type()
        if st == rs.stream.infrared:
            sp[f"infrared_{p.stream_index()}"] = p
        elif st == rs.stream.color:
            sp["color"] = p
        elif st == rs.stream.depth:
            sp["depth"] = p
        elif st == rs.stream.accel:
            sp["accel"] = p
        elif st == rs.stream.gyro:
            sp["gyro"] = p
    return sp


def dump_intrinsics(dev: rs.device, sp: dict, out_dir: str) -> None:
    streams = {}
    for key, p in sp.items():
        if key in ("accel", "gyro"):
            continue
        v = p.as_video_stream_profile()
        intr = v.get_intrinsics()
        streams[key] = {
            "width": intr.width, "height": intr.height, "fps": p.fps(),
            "model": str(intr.model),
            "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
            "coeffs": list(intr.coeffs),
        }

    extr = {}

    def add_extr(a, b):
        if a in sp and b in sp:
            e = sp[a].get_extrinsics_to(sp[b])
            extr[f"{a}_to_{b}"] = {"rotation": list(e.rotation),
                                   "translation": list(e.translation)}

    add_extr("infrared_1", "infrared_2")   # stereo baseline
    add_extr("infrared_1", "color")
    add_extr("infrared_1", "accel")        # camera<-IMU seed for calibration init
    add_extr("infrared_1", "gyro")

    imu = {}
    for key in ("accel", "gyro"):
        if key not in sp:
            continue
        mp = sp[key].as_motion_stream_profile()
        mi = mp.get_motion_intrinsics()
        imu[key] = {
            "fps": mp.fps(),
            "data": [list(r) for r in mi.data],
            "noise_variances": list(mi.noise_variances),
            "bias_variances": list(mi.bias_variances),
        }

    info = {
        "name": dev.get_info(rs.camera_info.name),
        "serial": dev.get_info(rs.camera_info.serial_number),
        "firmware": dev.get_info(rs.camera_info.firmware_version),
    }
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump({"device": info, "streams": streams,
                   "extrinsics": extr, "imu": imu}, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract .db3 -> Kalibr/OpenICC folder")
    ap.add_argument("recording", nargs="?", help=".db3 file (default: newest in data/recordings)")
    ap.add_argument("--out", help="output dir (default: <recording_dir>/<name>_extracted)")
    ap.add_argument("--no-color", action="store_true", help="skip color extraction")
    ap.add_argument("--max-frames", type=int, default=0, help="limit framesets (0=all)")
    args = ap.parse_args()

    path = args.recording or newest_recording()
    if not path or not os.path.exists(path):
        print("No .db3 recording found. Pass a path or record one first.")
        return 1
    out = args.out or os.path.join(os.path.dirname(path),
                                   os.path.splitext(os.path.basename(path))[0] + "_extracted")
    for c in ("cam0", "cam1", COLOR_CAM):
        os.makedirs(os.path.join(out, c), exist_ok=True)
    print(f"Input : {path}")
    print(f"Output: {out}")

    # Sensor-level playback (NOT pipeline): the pipeline syncer collapses the
    # 200 Hz IMU down to the video rate. Reading each sensor straight into a
    # frame_queue preserves every accel/gyro sample.
    ctx = rs.context()
    pb = ctx.load_device(path).as_playback()
    pb.set_real_time(False)
    sensors = pb.query_sensors()
    all_profiles = [p for s in sensors for p in s.get_stream_profiles()]
    sp = index_profiles(all_profiles)
    dump_intrinsics(pb, sp, out)

    q = rs.frame_queue(4096, keep_frames=True)
    for s in sensors:
        s.open(s.get_stream_profiles())
        s.start(q)

    accel: list[tuple[int, float, float, float]] = []
    gyro: list[tuple[int, float, float, float]] = []
    saved = {"cam0": 0, "cam1": 0, COLOR_CAM: 0}
    n = 0
    while True:
        try:
            fr = q.wait_for_frame(1500)   # EOF -> timeout -> break
        except RuntimeError:
            break
        fr = fr.as_frame()
        p = fr.get_profile()
        st = p.stream_type()
        t = ts_ns(fr)
        if st == rs.stream.infrared and p.stream_index() in IR_TO_CAM:
            cam = IR_TO_CAM[p.stream_index()]
            cv2.imwrite(os.path.join(out, cam, f"{t}.png"),
                        np.asanyarray(fr.as_video_frame().get_data()))
            saved[cam] += 1
        elif st == rs.stream.color and not args.no_color:
            cv2.imwrite(os.path.join(out, COLOR_CAM, f"{t}.png"),
                        np.asanyarray(fr.as_video_frame().get_data()))
            saved[COLOR_CAM] += 1
        elif st == rs.stream.accel:
            d = fr.as_motion_frame().get_motion_data()
            accel.append((t, d.x, d.y, d.z))
        elif st == rs.stream.gyro:
            d = fr.as_motion_frame().get_motion_data()
            gyro.append((t, d.x, d.y, d.z))
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    for s in sensors:
        try:
            s.stop(); s.close()
        except RuntimeError:
            pass
    n_fs = n

    # build imu0.csv: interpolate accel onto gyro timestamps
    n_imu = write_imu_csv(out, accel, gyro)

    print(f"\nframes read    : {n_fs}")
    print(f"images saved   : cam0={saved['cam0']} cam1={saved['cam1']} {COLOR_CAM}={saved[COLOR_CAM]}")
    print(f"imu samples    : accel={len(accel)} gyro={len(gyro)} -> imu0.csv rows={n_imu}")
    print(f"intrinsics     : {os.path.join(out, 'intrinsics.json')}")
    return 0


def write_imu_csv(out: str, accel, gyro) -> int:
    csv_path = os.path.join(out, "imu0.csv")
    if not gyro or not accel:
        # still write header so downstream tools don't choke
        with open(csv_path, "w") as f:
            f.write("#timestamp [ns],w_x,w_y,w_z [rad/s],a_x,a_y,a_z [m/s^2]\n")
        return 0
    accel.sort(); gyro.sort()
    at = np.array([a[0] for a in accel], dtype=np.float64)
    ax = np.array([[a[1], a[2], a[3]] for a in accel], dtype=np.float64)
    rows = []
    for t, gx, gy, gz in gyro:
        if t < at[0] or t > at[-1]:
            continue  # outside accel coverage -> skip
        a = [np.interp(t, at, ax[:, k]) for k in range(3)]
        rows.append((t, gx, gy, gz, a[0], a[1], a[2]))
    with open(csv_path, "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z [rad/s],a_x,a_y,a_z [m/s^2]\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.9f},{r[2]:.9f},{r[3]:.9f},{r[4]:.9f},{r[5]:.9f},{r[6]:.9f}\n")
    return len(rows)


if __name__ == "__main__":
    sys.exit(main())
