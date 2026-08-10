#!/usr/bin/env python3
"""
Convert RealSense .db3 recordings into an OpenICC (mynteye-style) dataset.

Uses the librealsense CLI `rs-convert` to extract frames + motion data (the
pyrealsense2 python playback API hangs mid-clip on long recordings; rs-convert
is the robust native path).

OpenICC's run_mynteye_calibration.py expects:
    <dataset>/cam/cam0/<ts_ns>.png          # camera-intrinsics clip (board, slow)
    <dataset>/cam_imu/cam0/<ts_ns>.png      # cam-IMU clip (board, excite 6 axes)
    <dataset>/cam_imu/imu0.csv              # EuRoC: ts_ns,gx,gy,gz,ax,ay,az (no header)

We use the LEFT IR image ("Infrared 1"). Image and IMU share the host-clock epoch
timestamp, so cam/IMU line up (OpenICC still solves the residual time offset).

    conda activate umi
    python calibration/db3_to_openicc.py \
        --cam     data/recordings/<cam_clip>.db3 \
        --cam_imu data/recordings/<cam_imu_clip>.db3 \
        --out     data/calib/d435i_openicc
"""
from __future__ import annotations
import argparse, csv, glob, os, re, shutil, subprocess, sys, tempfile
import numpy as np


def run_rs_convert(db3: str, tmp: str, want_imu: bool):
    cmd = ["rs-convert", "-i", db3, "-p", os.path.join(tmp, "f")]
    if want_imu:
        cmd += ["-v", os.path.join(tmp, "imu")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract(db3: str, img_dir: str, imu_csv: str | None):
    os.makedirs(img_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="rsconv_")
    try:
        run_rs_convert(db3, tmp, imu_csv is not None)
        # left IR frames:  "f_Infrared 1_<ms>.png"  ->  <ts_ns>.png
        n_img = 0
        for png in glob.glob(os.path.join(tmp, "f_Infrared 1_*.png")):
            m = re.search(r"_Infrared 1_([0-9.]+)\.png$", os.path.basename(png))
            if not m:
                continue
            ts_ns = int(round(float(m.group(1)) * 1e6))  # ms -> ns
            shutil.move(png, os.path.join(img_dir, f"{ts_ns}.png"))
            n_img += 1
        n_imu = 0
        if imu_csv is not None:
            csvs = glob.glob(os.path.join(tmp, "imu*"))
            n_imu = parse_motion_csv(csvs[0], imu_csv) if csvs else 0
        return n_img, n_imu
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_motion_csv(rs_csv: str, out_csv: str) -> int:
    """rs-convert motion csv -> EuRoC imu0.csv (header-less). Columns:
       Stream Type, F#, HW(ms), Backend(ms), Host(ms), x, y, z.  Use Host ms."""
    accel, gyro = [], []
    with open(rs_csv) as f:
        for row in csv.reader(f):
            if not row or row[0] not in ("Accel", "Gyro"):
                continue
            t_ns = int(round(float(row[4]) * 1e6))  # Host Timestamp(ms) -> ns
            x, y, z = float(row[5]), float(row[6]), float(row[7])
            (accel if row[0] == "Accel" else gyro).append((t_ns, x, y, z))
    return write_imu_csv(out_csv, accel, gyro)


def write_imu_csv(path, accel, gyro) -> int:
    """Header-less EuRoC rows: ts_ns,gx,gy,gz,ax,ay,az (accel interp -> gyro times)."""
    if not accel or not gyro:
        open(path, "w").close(); return 0
    accel.sort(); gyro.sort()
    at = np.array([a[0] for a in accel], float)
    av = np.array([[a[1], a[2], a[3]] for a in accel], float)
    rows = []
    for t, gx, gy, gz in gyro:
        if t < at[0] or t > at[-1]:
            continue
        a = [np.interp(t, at, av[:, k]) for k in range(3)]
        rows.append((t, gx, gy, gz, a[0], a[1], a[2]))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r[0], f"{r[1]:.9f}", f"{r[2]:.9f}", f"{r[3]:.9f}",
                        f"{r[4]:.9f}", f"{r[5]:.9f}", f"{r[6]:.9f}"])
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="RealSense .db3 -> OpenICC dataset (rs-convert)")
    ap.add_argument("--cam", help="camera-intrinsics clip .db3 (-> cam/cam0)")
    ap.add_argument("--cam_imu", help="cam-IMU clip .db3 (-> cam_imu/cam0 + imu0.csv)")
    ap.add_argument("--out", required=True, help="output dataset dir")
    args = ap.parse_args()
    if not args.cam and not args.cam_imu:
        ap.error("provide --cam and/or --cam_imu")

    out = os.path.abspath(args.out)
    if args.cam:
        print(f"[cam]     {args.cam}", flush=True)
        ni, _ = extract(args.cam, os.path.join(out, "cam", "cam0"), None)
        print(f"          -> cam/cam0: {ni} frames", flush=True)
    if args.cam_imu:
        print(f"[cam_imu] {args.cam_imu}", flush=True)
        ni, nimu = extract(args.cam_imu, os.path.join(out, "cam_imu", "cam0"),
                           os.path.join(out, "cam_imu", "imu0.csv"))
        print(f"          -> cam_imu/cam0: {ni} frames | imu0.csv: {nimu} rows", flush=True)
    print(f"\nDataset: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
