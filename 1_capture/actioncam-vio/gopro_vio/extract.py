"""Extract IMU (ACCL/GYRO) telemetry and video info from a GoPro MP4.

Outputs, per video:
  imu_data.json   gopro-telemetry-style JSON (what UMI's ORB-SLAM3 fork reads)
  imu.csv         t[s], gx, gy, gz [rad/s], ax, ay, az [m/s^2]
  video_info.json resolution / fps / duration / creation time

Usage:
  python -m gopro_vio.extract data/GX014222.MP4 -o output/GX014222
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib

import numpy as np

from . import gpmf, mp4


def extract(video_path: str, out_dir: str) -> dict:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = mp4.parse(video_path)
    gpmd = info.find_track("gpmd")
    if gpmd is None:
        raise RuntimeError(f"{video_path}: no GPMF (gpmd) track found")
    video = None
    for t in info.tracks:
        if t.handler == "vide":
            video = t
            break

    payloads = mp4.read_track_samples(video_path, gpmd)
    accl = gpmf.extract_sensor(payloads, gpmd.sample_times, gpmd.sample_durations, "ACCL")
    gyro = gpmf.extract_sensor(payloads, gpmd.sample_times, gpmd.sample_durations, "GYRO")
    if accl is None or gyro is None:
        raise RuntimeError(f"{video_path}: ACCL/GYRO stream missing")

    def rate(s):
        return (len(s.t) - 1) / (s.t[-1] - s.t[0]) if len(s.t) > 1 else 0.0

    vinfo = {
        "path": str(video_path),
        "width": video.width if video else 0,
        "height": video.height if video else 0,
        "num_frames": len(video.sample_sizes) if video else 0,
        "fps": (len(video.sample_sizes) / video.duration) if video and video.duration else 0.0,
        "duration_s": video.duration if video else gpmd.duration,
        "creation_time": info.creation_time.isoformat() if info.creation_time else None,
        "accl_hz": rate(accl),
        "gyro_hz": rate(gyro),
        "accl_orientation": accl.orientation,
        "gyro_orientation": gyro.orientation,
        "accl_name": accl.name,
        "gyro_name": gyro.name,
    }
    (out / "video_info.json").write_text(json.dumps(vinfo, indent=2))

    # --- imu_data.json in gopro-telemetry layout (cts in milliseconds) ---
    t0 = info.creation_time or _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)

    def stream_json(s: gpmf.SensorStream):
        samples = []
        for ti, v in zip(s.t, s.data):
            cts = float(ti * 1000.0)
            date = (t0 + _dt.timedelta(seconds=float(ti))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            samples.append({"value": [float(x) for x in v], "cts": cts, "date": date})
        return {"samples": samples, "name": s.name, "units": s.units}

    imu_json = {"1": {"streams": {"ACCL": stream_json(accl), "GYRO": stream_json(gyro)},
                      "device name": "GoPro HERO7 Black"}}
    (out / "imu_data.json").write_text(json.dumps(imu_json))

    # --- compact CSV: gyro timeline, accel interpolated onto it ---
    acc_i = np.stack([np.interp(gyro.t, accl.t, accl.data[:, i]) for i in range(3)], axis=1)
    tbl = np.column_stack([gyro.t, gyro.data, acc_i])
    np.savetxt(out / "imu.csv", tbl, delimiter=",",
               header="t,gx,gy,gz,ax,ay,az", comments="")
    return vinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    vinfo = extract(args.video, args.out)
    print(json.dumps(vinfo, indent=2))


if __name__ == "__main__":
    main()
