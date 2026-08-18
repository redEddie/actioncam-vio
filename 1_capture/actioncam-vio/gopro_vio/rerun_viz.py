"""Interactive SLAM map + trajectory visualization with rerun (rerun.io).

Writes a .rrd recording containing:
  - the map point cloud (colored by height, or by submap id if fragmented)
  - keyframe positions
  - the full trajectory line
  - a time-animated camera pose with pinhole frustum (+ optional video
    thumbnails on the timeline)

View with:  rerun output/GX014220/slam/map.rrd

Usage:
  python -m gopro_vio.rerun_viz output/GX014220/slam \
      --video output/GX014220/slam/video_slam_960x720_30fps.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import re

import numpy as np
import pandas as pd
import rerun as rr

from .map_viz import load_ply


def parse_settings(yaml_path: pathlib.Path):
    txt = yaml_path.read_text()
    def num(key, default=None):
        m = re.search(rf"^{key}:\s*([-\d.eE+]+)", txt, re.M)
        return float(m.group(1)) if m else default
    return {
        "fx": num("Camera1.fx"), "fy": num("Camera1.fy"),
        "cx": num("Camera1.cx"), "cy": num("Camera1.cy"),
        "w": int(num("Camera.width", 960)), "h": int(num("Camera.height", 720)),
    }


def height_colors(vals: np.ndarray) -> np.ndarray:
    import matplotlib
    lo, hi = np.percentile(vals, [2, 98])
    t = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
    return (matplotlib.colormaps["turbo"](t)[:, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slam_dir", help="output/<VIDEO>/slam directory")
    ap.add_argument("--video", default=None,
                    help="transcoded SLAM video for timeline thumbnails")
    ap.add_argument("--video-fps", type=float, default=2.0)
    ap.add_argument("--thumb-width", type=int, default=480,
                    help="thumbnail width; pinhole intrinsics are scaled to "
                         "match so images fill the frustum image plane")
    ap.add_argument("--out", default=None, help=".rrd output path")
    ap.add_argument("--traj-radius-cm", type=float, default=0.2,
                    help="trajectory line radius in cm (scene units)")
    ap.add_argument("--kf-radius-cm", type=float, default=0.5,
                    help="keyframe sphere radius in cm (scene units)")
    args = ap.parse_args()

    d = pathlib.Path(args.slam_dir)
    out = pathlib.Path(args.out) if args.out else d / "map.rrd"
    name = d.parent.name

    rr.init(f"gopro_slam/{name}")
    rr.save(str(out))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # --- map point cloud (static) ---
    ply = d / "map_points.ply"
    if ply.exists():
        data = load_ply(str(ply))
        pts = data[:, :3]
        map_ids = data[:, 3].astype(int) if data.shape[1] > 3 else None
        if map_ids is not None and len(np.unique(map_ids)) > 1:
            palette = np.array([[230, 60, 60], [60, 160, 230], [90, 200, 90],
                                [230, 180, 60], [180, 90, 220], [90, 220, 210],
                                [240, 130, 180], [160, 160, 160]], np.uint8)
            colors = palette[map_ids % len(palette)]
        else:
            # vertical axis = smallest extent
            up = int(np.argmin(pts.max(0) - pts.min(0)))
            colors = height_colors(-pts[:, up])
        # scene-size-adaptive point radius: ~2 mm on a tabletop map, capped
        # at the old 2 cm for outdoor-scale maps
        lo, hi = np.percentile(pts, [2, 98], axis=0)
        pt_radius = float(np.clip(np.linalg.norm(hi - lo) * 0.003, 0.002, 0.02))
        rr.log("world/map", rr.Points3D(pts, colors=colors, radii=pt_radius),
               static=True)
        print(f"[rerun] map points: {len(pts)} (radius {pt_radius*1000:.1f} mm)")

    # --- keyframes (static) ---
    kf_csv = d / "keyframes.csv"
    if kf_csv.exists():
        kf = pd.read_csv(kf_csv)
        rr.log("world/keyframes",
               rr.Points3D(kf[["x", "y", "z"]].to_numpy(),
                           colors=[255, 255, 255],
                           radii=args.kf_radius_cm / 100), static=True)
        print(f"[rerun] keyframes: {len(kf)}")

    # --- trajectory: static line + animated camera ---
    traj_csv = d / "camera_trajectory.csv"
    cam = None
    if traj_csv.exists():
        df = pd.read_csv(traj_csv)
        lost = df["is_lost"].astype(str).str.strip().str.lower().eq("true")
        cam = df[~lost]
        xyz = cam[["x", "y", "z"]].to_numpy()
        rr.log("world/trajectory",
               rr.LineStrips3D([xyz], colors=[255, 40, 40],
                               radii=args.traj_radius_cm / 100),
               static=True)
        print(f"[rerun] trajectory: {len(cam)} poses")

        st = parse_settings(d / "orbslam3_settings.yaml")
        for row in cam.itertuples():
            rr.set_time("video", duration=float(row.timestamp))
            rr.log("world/camera", rr.Transform3D(
                translation=[row.x, row.y, row.z],
                rotation=rr.Quaternion(xyzw=[row.q_x, row.q_y, row.q_z, row.q_w])))
        # Pinhole scaled to the thumbnail resolution: rerun paints images in
        # the pinhole's pixel coordinates, so a mismatch leaves the image in
        # a corner of the frustum's image plane.  Scaling fx with the width
        # keeps the frustum FOV identical.
        s = args.thumb_width / st["w"]
        tw, th = args.thumb_width, round(st["h"] * s)
        rr.set_time("video", duration=float(cam["timestamp"].iloc[0]))
        rr.log("world/camera/image", rr.Pinhole(
            focal_length=[st["fx"] * s, st["fy"] * s],
            principal_point=[(st["cx"] + 0.5) * s - 0.5,
                             (st["cy"] + 0.5) * s - 0.5],
            resolution=[tw, th],
            image_plane_distance=0.5), static=True)

    # --- optional video thumbnails on the timeline ---
    if args.video:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tw = args.thumb_width
        th = round(h0 * tw / w0)
        step = max(1, round(fps / args.video_fps))
        logged = 0
        for idx in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                break
            small = cv2.resize(img, (tw, th))
            ok2, jpg = cv2.imencode(".jpg", small,
                                    [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok2:
                continue
            rr.set_time("video", duration=idx / fps)
            rr.log("world/camera/image",
                   rr.EncodedImage(contents=jpg.tobytes(),
                                   media_type="image/jpeg"))
            logged += 1
        cap.release()
        print(f"[rerun] video thumbnails: {logged}")

    print(f"[done] {out}  →  열기: rerun {out}")


if __name__ == "__main__":
    main()
