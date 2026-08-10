#!/usr/bin/env python3
"""
Stage A1 - dump factory calibration for EVERY connected RealSense.

Reads straight from librealsense stream profiles (no streaming needed), so it
works for any model regardless of which streams it has:
  - D435i / D455 : IR stereo + RGB + depth + IMU
  - D405         : stereo only (no IMU)

For each device writes calibration/factory_calib_<serial>.json with intrinsics,
stereo + cross-stream extrinsics, IMU intrinsics (if present), depth scale, and
the ORB-SLAM3 stereo params (fx,fy,cx,cy, baseline, T_b_c1 seed).

    conda activate umi
    python calibration/dump_factory_calib.py
"""
import json, os, sys
import numpy as np
import pyrealsense2 as rs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ext_to_T(ext):
    R = np.array(ext.rotation, float).reshape(3, 3).T  # rs2 is column-major
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = ext.translation
    return T


def key_of(p):
    st = str(p.stream_type()).replace("stream.", "")
    return f"{st}_{p.stream_index()}" if p.stream_index() else st


def pick(profs):
    """Prefer 640x480@30 for video; otherwise a sensible default."""
    vids = [p for p in profs if p.is_video_stream_profile()]
    if vids:
        def score(p):
            v = p.as_video_stream_profile(); res = (v.width(), v.height())
            pref = {(640, 480): 0, (848, 480): 1, (1280, 720): 2}.get(res, 3)
            return (pref, 0 if p.fps() == 30 else 1)
        return sorted(vids, key=score)[0]
    return sorted(profs, key=lambda p: -p.fps())[0]  # motion: highest rate


def dump_device(dev):
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)

    groups = {}
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            groups.setdefault(key_of(p), []).append(p)
    chosen = {k: pick(v) for k, v in groups.items()}

    streams, imu = {}, {}
    for k, p in chosen.items():
        if p.is_video_stream_profile():
            i = p.as_video_stream_profile().get_intrinsics()
            streams[k] = {"width": i.width, "height": i.height, "fps": p.fps(),
                          "model": str(i.model), "fx": i.fx, "fy": i.fy,
                          "ppx": i.ppx, "ppy": i.ppy, "coeffs": list(i.coeffs)}
        elif p.is_motion_stream_profile():
            mi = p.as_motion_stream_profile().get_motion_intrinsics()
            imu[k] = {"fps": p.fps(), "data": [list(r) for r in mi.data],
                      "noise_variances": list(mi.noise_variances),
                      "bias_variances": list(mi.bias_variances)}

    ref = "infrared_1" if "infrared_1" in chosen else ("depth" if "depth" in chosen else next(iter(chosen)))
    extr = {}
    for k, p in chosen.items():
        if k == ref:
            continue
        try:
            extr[f"{ref}_to_{k}"] = ext_to_T(chosen[ref].get_extrinsics_to(p)).tolist()
        except Exception:
            pass

    baseline = None
    if "infrared_1" in chosen and "infrared_2" in chosen:
        baseline = float(abs(chosen["infrared_1"].get_extrinsics_to(chosen["infrared_2"]).translation[0]))

    try:
        depth_scale = dev.first_depth_sensor().get_depth_scale()
    except Exception:
        depth_scale = None

    T_b_c1 = None
    if "infrared_1" in chosen and "gyro" in chosen:
        T_b_c1 = ext_to_T(chosen["infrared_1"].get_extrinsics_to(chosen["gyro"])).tolist()

    out = {"device": {"name": name, "serial": serial,
                      "firmware": dev.get_info(rs.camera_info.firmware_version)},
           "streams": streams, "imu": imu, "extrinsics": extr,
           "stereo_baseline_m": baseline, "T_b_c1_seed": T_b_c1, "depth_scale": depth_scale}
    path = os.path.join(REPO, "calibration", f"factory_calib_{serial}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {name}  (SN {serial}) ===")
    print(f"  streams: {sorted(streams)}   imu: {sorted(imu) or 'none'}")
    if "infrared_1" in streams:
        i = streams["infrared_1"]
        print(f"  [SLAM cam1=left IR]  fx={i['fx']:.3f} fy={i['fy']:.3f} cx={i['ppx']:.3f} cy={i['ppy']:.3f}"
              f"  dist={i['coeffs']}")
    if baseline:
        print(f"  stereo baseline: {baseline:.7f} m")
    if depth_scale:
        print(f"  depth scale: {depth_scale} m/unit")
    print(f"  -> {os.path.relpath(path, REPO)}")


def main():
    devs = list(rs.context().query_devices())
    if not devs:
        print("No RealSense devices connected."); return 1
    print(f"Found {len(devs)} device(s).")
    for d in devs:
        dump_device(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
