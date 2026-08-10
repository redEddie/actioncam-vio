#!/usr/bin/env python3
"""
Log a SLAM trajectory to Rerun: camera poses (frustums via Pinhole), the map
point cloud, and the path -- with a frame timeline you can scrub. Saves a .rrd
file (headless-friendly); open it with `rerun file.rrd` on the desktop, or
`rerun --web-viewer file.rrd` for a browser.

    python dataset/log_rerun.py \
        --traj data/maps/mapping_..._traj_frames.txt \
        --mappoints data/maps/mapping_..._traj_mappoints.txt \
        --t_world_map calibration/tx_slam_tag_d435i_envmap2.json \
        --out data/maps/slam.rrd

traj: EuRoC "ts tx ty tz qx qy qz qw" = camera pose Twc (OpenCV RDF optical frame).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import rerun as rr


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--mappoints", default=None)
    ap.add_argument("--t_world_map", default=None)
    ap.add_argument("--out", required=True, help="output .rrd")
    ap.add_argument("--fx", type=float, default=390.08)
    ap.add_argument("--cx", type=float, default=320.45)
    ap.add_argument("--cy", type=float, default=240.71)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    T = np.loadtxt(a.traj)
    if T.ndim == 1:
        T = T[None, :]
    pos, quat = T[:, 1:4], T[:, 4:8]

    Rwm = np.eye(3); twm = np.zeros(3)
    if a.t_world_map:
        Twm = np.array(json.load(open(a.t_world_map))["T_world_map"], float)
        Rwm, twm = Twm[:3, :3], Twm[:3, 3]
    posw = (Rwm @ pos.T).T + twm

    rr.init("umi_slam")
    rr.save(a.out)

    # world axes (id13 origin if transformed): RGB triad at (0,0,0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/id13_origin", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)

    # map point cloud (static)
    if a.mappoints:
        M = np.loadtxt(a.mappoints)
        if M.ndim == 1:
            M = M[None, :]
        Mw = (Rwm @ M.T).T + twm
        rr.log("world/map_points", rr.Points3D(Mw, colors=[160, 160, 160], radii=0.003),
               static=True)

    # full camera path (static)
    rr.log("world/path", rr.LineStrips3D([posw], colors=[80, 130, 255]), static=True)

    # per-frame camera pose + frustum on a scrubable timeline
    for i in range(len(posw)):
        rr.set_time("frame", sequence=i)
        Rwc = Rwm @ quat_to_R(quat[i])
        rr.log("world/camera", rr.Transform3D(translation=posw[i], mat3x3=Rwc))
        rr.log("world/camera", rr.Pinhole(
            resolution=[a.width, a.height],
            focal_length=[a.fx, a.fx],
            principal_point=[a.cx, a.cy],
            image_plane_distance=0.04,
            camera_xyz=rr.ViewCoordinates.RDF))   # OpenCV optical: X right, Y down, Z fwd

    print(f"logged {len(posw)} camera poses -> {a.out}")
    print(f"open: rerun {a.out}   (or: rerun --web-viewer {a.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
