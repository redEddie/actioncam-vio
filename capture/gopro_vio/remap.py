"""Remap recorded-mode video to webcam-mode geometry (VLA domain alignment).

Both calibrations describe the SAME physical lens, so converting a recorded
frame to the webcam view is an exact per-pixel reprojection: for every target
(webcam) pixel, unproject through the webcam KB4 model to a ray, project the
ray through the recorded KB4 model, and sample.  Requires the target FOV to
be a subset of the source FOV (measured: 113x70 deg ⊂ 130x104 deg).

Usage:
  python -m gopro_vio.remap data/acepro2/VID_..._528.mp4 \
      --src-calib cameras/acepro2/calibration/v527/intrinsics.json \
      --dst-calib cameras/acepro2/calibration/webcam/intrinsics.json \
      -o output/acepro2_528/webcam_view.mp4 --fps-div 2
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
from tqdm import tqdm


def kb4_theta_d(theta, D):
    t2 = theta * theta
    return theta * (1 + D[0]*t2 + D[1]*t2**2 + D[2]*t2**3 + D[3]*t2**4)


def build_maps(src_calib: dict, dst_calib: dict):
    """cv2.remap maps: for each dst pixel -> src pixel coordinates."""
    Ks = np.array(src_calib["K"]); Ds = np.array(src_calib["D"])
    Kd = np.array(dst_calib["K"]); Dd = np.array(dst_calib["D"])
    Wd, Hd = dst_calib["image_size"]

    # dst pixel grid -> rays (undistortPoints inverts the dst KB4 model)
    u, v = np.meshgrid(np.arange(Wd, dtype=np.float64),
                       np.arange(Hd, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    n = cv2.fisheye.undistortPoints(pts, Kd, Dd.reshape(4, 1)).reshape(-1, 2)

    # rays -> src pixels through the src KB4 model
    r = np.linalg.norm(n, axis=1)
    theta = np.arctan(r)
    rd = kb4_theta_d(theta, Ds)
    scale = np.where(r > 1e-9, rd / np.maximum(r, 1e-12), 1.0)
    xs = Ks[0, 0] * n[:, 0] * scale + Ks[0, 2]
    ys = Ks[1, 1] * n[:, 1] * scale + Ks[1, 2]
    map_x = xs.reshape(Hd, Wd).astype(np.float32)
    map_y = ys.reshape(Hd, Wd).astype(np.float32)
    return map_x, map_y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--src-calib", required=True,
                    help="recorded-mode intrinsics.json")
    ap.add_argument("--dst-calib", required=True,
                    help="webcam-mode intrinsics.json (target geometry)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--fps-div", type=int, default=2,
                    help="temporal subsample (59.94 -> 29.97)")
    ap.add_argument("--t-start", type=float, default=0.0)
    ap.add_argument("--t-end", type=float, default=None)
    args = ap.parse_args()

    src = json.loads(pathlib.Path(args.src_calib).read_text())
    dst = json.loads(pathlib.Path(args.dst_calib).read_text())
    map_x, map_y = build_maps(src, dst)
    Wd, Hd = dst["image_size"]

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(args.t_start * fps)
    f1 = int(args.t_end * fps) if args.t_end else n
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / args.fps_div, (Wd, Hd))
    written = 0
    for idx in tqdm(range(f0, f1), desc="remap", unit="f"):
        ok, img = cap.read()
        if not ok:
            break
        if (idx - f0) % args.fps_div:
            continue
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        vw.write(warped)
        written += 1
    vw.release()
    print(f"[done] {out} ({written} frames @ {fps/args.fps_div:.2f} fps, "
          f"{Wd}x{Hd})")


if __name__ == "__main__":
    main()
