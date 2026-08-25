"""Extract IMU from Insta360 videos (Ace Pro 2 etc.) via telemetry-parser.

Insta360 stores telemetry in a proprietary trailer appended to the MP4
(invisible to ffprobe), parsed by AdrianEddy/telemetry-parser.  Output is
the same imu.csv format as gopro_vio.extract, so the rest of the pipeline
(imu_sync, slam, viz) is unchanged.

Timestamp caution (user-flagged): telemetry timestamps are taken relative
to the stream start; the residual constant offset against the video clock
is estimated downstream by gopro_vio.imu_sync (gyro vs ChArUco angular
velocity cross-correlation) — no assumptions needed here.

Usage:
  python -m gopro_vio.insta360 data/acepro2/VID_..._526.mp4 -o output/acepro2_526
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def extract(video_path: str, out_dir: str) -> dict:
    import telemetry_parser
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tp = telemetry_parser.Parser(str(video_path))
    imu = tp.normalized_imu()
    if not imu:
        raise RuntimeError(f"{video_path}: no IMU telemetry found")

    t = np.array([s["timestamp_ms"] for s in imu]) * 1e-3
    gyr = np.array([s["gyro"] for s in imu], dtype=np.float64)
    acc = np.array([s["accl"] for s in imu], dtype=np.float64)

    # unit auto-detection: gyro deg/s -> rad/s, accel g -> m/s^2
    gyro_unit, accl_unit = "rad/s", "m/s^2"
    if np.percentile(np.abs(gyr), 99) > 20:          # rad/s would be extreme
        gyr = np.radians(gyr)
        gyro_unit = "deg/s (converted)"
    acc_norm = float(np.median(np.linalg.norm(acc, axis=1)))
    if 0.5 < acc_norm < 2.0:                          # measured in g
        acc = acc * 9.80665
        accl_unit = "g (converted)"

    rate = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0
    info = {
        "path": str(video_path),
        "camera": f"{tp.camera} {tp.model}",
        "imu_samples": int(len(t)),
        "imu_hz": rate,
        "t_range_s": [float(t[0]), float(t[-1])],
        "gyro_unit_source": gyro_unit,
        "accl_unit_source": accl_unit,
        "acc_norm_med_ms2": float(np.median(np.linalg.norm(acc, axis=1))),
    }
    (out / "video_info.json").write_text(json.dumps(info, indent=2))

    tbl = np.column_stack([t, gyr, acc])
    np.savetxt(out / "imu.csv", tbl, delimiter=",",
               header="t,gx,gy,gz,ax,ay,az", comments="")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    print(json.dumps(extract(args.video, args.out), indent=2))


if __name__ == "__main__":
    main()
