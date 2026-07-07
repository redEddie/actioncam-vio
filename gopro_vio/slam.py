"""Run ORB-SLAM3 monocular-inertial (UMI's chicheng/orb_slam3 docker image)
on a GoPro video using our own calibration.

Does three things:
 1. writes an ORB-SLAM3 settings yaml at the target processing resolution
    (frames are resized by the gopro_slam binary to Camera.width/height)
 2. rewrites the IMU stream as the gopro-telemetry-style json the binary
    expects: ACCL/GYRO resampled onto one uniform grid starting at exactly
    t=0 of the video timeline (the binary re-zeros on the first sample and
    pairs ACCL/GYRO index-wise, so a shared grid is required), with the
    calibrated time offset applied
 3. runs the container and collects trajectory csv/tum + map atlas

Usage:
  python -m gopro_vio.slam data/GX014217.MP4 \
      --imu output/GX014217/imu.csv -o output/GX014217/slam
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np

# UMI's chicheng/orb_slam3 with local patches (see docker/README):
#  - pre-IMU-init tracking gates relaxed (50->28, 30->20 inliers) so that
#    low-parallax forward-walking maps survive until IMU initialization
#  - IMU array out-of-bounds fix at end of video
#  - periodic trajectory checkpoint + crash-rescue signal handler
DOCKER_IMAGE = "orb_slam3:gate28-rescue"


def scaled_intrinsics(calib: dict, target_w: int):
    w, h = calib["image_size"]
    K = np.asarray(calib["K"])
    s = target_w / w
    target_h = round(h * s)
    # pixel-center convention: x' = (x + 0.5) * s - 0.5
    fx, fy = K[0, 0] * s, K[1, 1] * s
    cx = (K[0, 2] + 0.5) * s - 0.5
    cy = (K[1, 2] + 0.5) * s - 0.5
    return fx, fy, cx, cy, target_w, target_h


def write_settings_yaml(calib: dict, extr: dict, path: pathlib.Path,
                        target_w=960, n_features=2000,
                        ini_fast=20, min_fast=7, far_points=100.0,
                        fps_div=2):
    fx, fy, cx, cy, w, h = scaled_intrinsics(calib, target_w)
    D = calib["D"]
    R_ic = np.asarray(extr["R_imu_cam"])
    t_ic = np.asarray(extr.get("t_imu_cam_m", [0, 0, 0]))
    Tbc = np.eye(4)
    Tbc[:3, :3] = R_ic
    Tbc[:3, 3] = t_ic
    data = ", ".join(f"{v:.8f}" for v in Tbc.ravel())

    yaml = f"""%YAML:1.0
File.version: "1.0"
Camera.type: "KannalaBrandt8"
# fisheye KB4 intrinsics calibrated from ChArUco (gopro_vio.charuco),
# scaled from {calib['image_size'][0]}x{calib['image_size'][1]} to {w}x{h}
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: {D[0]:.10f}
Camera1.k2: {D[1]:.10f}
Camera1.k3: {D[2]:.10f}
Camera1.k4: {D[3]:.10f}
Camera.width: {w}
Camera.height: {h}
Camera.fps: {round(calib['fps'] / fps_div)}
Camera.RGB: 1
# camera->IMU(body) transform calibrated from gyro/vision alignment
# (gopro_vio.imu_sync); translation not observable, set to zero
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [{data}]
# GoPro-class IMU noise (values from UMI's GoPro settings, inflated)
IMU.NoiseGyro: 0.0015
IMU.NoiseAcc: 0.017
IMU.GyroWalk: 5.0e-5
IMU.AccWalk: 0.0055
IMU.Frequency: 200.0
System.thFarPoints: {far_points}
ORBextractor.nFeatures: {n_features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: {ini_fast}
ORBextractor.minThFAST: {min_fast}
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 0.5
"""
    path.write_text(yaml)


def write_slam_imu_json(imu_csv: str, time_offset_s: float,
                        path: pathlib.Path, rate=200.0, start_s: float = 0.0):
    """Resample gyro+accel onto a shared uniform grid starting at video t=0.

    The grid is padded 0.5 s past the last IMU sample (edge-held values) so
    that no video frame ever arrives without IMU measurements — ORB-SLAM3
    crashes on empty inter-frame IMU intervals at the end of the video.
    """
    d = np.loadtxt(imu_csv, delimiter=",", skiprows=1)
    # move IMU samples onto the (possibly trimmed) video clock
    t = d[:, 0] - time_offset_s - start_s
    gyr, acc = d[:, 1:4], d[:, 4:7]
    t_end = t[-1] + 0.5
    grid = np.arange(0.0, t_end, 1.0 / rate)
    gi = np.stack([np.interp(grid, t, gyr[:, i]) for i in range(3)], 1)
    ai = np.stack([np.interp(grid, t, acc[:, i]) for i in range(3)], 1)

    def stream(vals):
        return {"samples": [{"value": [float(x) for x in v],
                             "cts": float(ti * 1000.0)}
                            for ti, v in zip(grid, vals)]}

    path.write_text(json.dumps(
        {"1": {"streams": {"ACCL": stream(ai), "GYRO": stream(gi)},
               "device name": "GoPro HERO7 Black"}}))


def transcode(video: str, out_path: pathlib.Path, w: int, h: int,
              fps_div: int = 2, start_s: float = 0.0):
    """Re-encode to the SLAM processing resolution and frame rate, dropping
    audio/telemetry tracks.

    - the docker image's old FFmpeg chokes on raw 2.7K GoPro files
      (cap.read fails a few seconds in)
    - 59.94 fps gives almost no parallax between consecutive frames, so the
      monocular map after two-view init is too weak to survive the
      pre-IMU-init 50-inlier gate; ORB-SLAM3 is tuned for 20-30 fps
      (EuRoC/TUM-VI).  fps_div=2 -> 29.97 fps, fps_div=3 -> 19.98 fps.
    """
    if out_path.exists():
        return
    vf = f"fps=60000/{1001 * fps_div},scale={w}:{h}"
    seek = ["-ss", f"{start_s}"] if start_s > 0 else []
    cmd = ["ffmpeg", "-y", *seek, "-i", str(video), "-map", "0:v:0",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", "-an", "-dn",
           str(out_path)]
    print("[transcode]", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


def run_slam(video: str, imu_csv: str, out_dir: str,
             calib_json="cameras/hero7black/calibration/intrinsics.json",
             extr_json="cameras/hero7black/calibration/imu_extrinsics.json",
             target_w=960, n_features=2000, timeout_s=7200,
             ini_fast=20, min_fast=7, far_points=100.0, fps_div=2,
             start_s=0.0, mask_png=None, load_map=None, extra_args=()):
    out = pathlib.Path(out_dir).absolute()
    out.mkdir(parents=True, exist_ok=True)
    calib = json.loads(pathlib.Path(calib_json).read_text())
    extr = json.loads(pathlib.Path(extr_json).read_text())

    write_settings_yaml(calib, extr, out / "orbslam3_settings.yaml",
                        target_w, n_features, ini_fast, min_fast, far_points,
                        fps_div)
    write_slam_imu_json(imu_csv, extr["time_offset_s"], out / "imu_slam.json",
                        start_s=start_s)

    fx, fy, cx, cy, w, h = scaled_intrinsics(calib, target_w)
    fps_tag = round(calib["fps"] / fps_div)
    tag = f"_{w}x{h}_{fps_tag}fps" + (f"_s{start_s:g}" if start_s else "")
    slam_video = out / f"video_slam{tag}.mp4"
    transcode(video, slam_video, w, h, fps_div, start_s)
    video = slam_video

    mounts, opt_args = [], []
    if mask_png:
        # nonzero mask pixels are blacked out before feature extraction
        # (binary resizes the mask to the processing resolution itself)
        shutil.copy(mask_png, out / "slam_mask.png")
        opt_args += ["--mask_img", "/work/slam_mask.png"]
    if load_map:
        m = pathlib.Path(load_map).absolute()
        mounts += ["-v", f"{m.parent}:/map:ro"]
        opt_args += ["--load_map", f"/map/{m.name}"]
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{out}:/work",
        "-v", f"{video}:/work/video.mp4:ro",
        *mounts,
        DOCKER_IMAGE,
        "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
        "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting", "/work/orbslam3_settings.yaml",
        "--input_video", "/work/video.mp4",
        "--input_imu_json", "/work/imu_slam.json",
        "--output_trajectory_csv", "/work/camera_trajectory.csv",
        "--output_trajectory_tum", "/work/camera_trajectory.tum",
        "--save_map", "/work/map_atlas.osa",
        *opt_args,
        *extra_args,
    ]
    print("[slam]", " ".join(cmd))
    log = (out / "slam_log.txt").open("w")
    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                          timeout=timeout_s)
    log.close()
    print(f"[slam] exit code {proc.returncode}, log: {out/'slam_log.txt'}")

    # if the run crashed, promote the crash-rescue / checkpoint trajectory
    final = out / "camera_trajectory.csv"
    if not final.exists():
        for cand in (out / "camera_trajectory.csv.rescue",
                     out / "camera_trajectory.csv.ckpt"):
            if cand.exists():
                final.write_bytes(cand.read_bytes())
                print(f"[slam] promoted {cand.name} -> camera_trajectory.csv")
                break
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--imu", required=True, help="imu.csv from gopro_vio.extract")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--calib", default="cameras/hero7black/calibration/intrinsics.json")
    ap.add_argument("--extr", default="cameras/hero7black/calibration/imu_extrinsics.json")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--features", type=int, default=2000)
    ap.add_argument("--ini-fast", type=int, default=20)
    ap.add_argument("--min-fast", type=int, default=7)
    ap.add_argument("--far-points", type=float, default=100.0)
    ap.add_argument("--fps-div", type=int, default=2,
                    help="temporal subsampling divisor (2 -> 29.97 fps)")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="skip the first N seconds of video+IMU")
    ap.add_argument("--mask", help="png mask; nonzero pixels excluded from "
                    "tracking (UMI gripper mask)")
    ap.add_argument("--load-map", help="map_atlas.osa to localize against "
                    "(UMI map+relocalize workflow)")
    args, extra = ap.parse_known_args()
    rc = run_slam(args.video, args.imu, args.out, args.calib, args.extr,
                  args.width, args.features,
                  ini_fast=args.ini_fast, min_fast=args.min_fast,
                  far_points=args.far_points, fps_div=args.fps_div,
                  start_s=args.start_s, mask_png=args.mask,
                  load_map=args.load_map, extra_args=extra)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
