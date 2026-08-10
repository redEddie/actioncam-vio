#!/usr/bin/env python3
"""
Transform a SLAM camera trajectory (map frame) into the WORLD (origin-tag) frame
using T_world_map from calibrate_slam_tag, and visualize.

    T_world_cam(t) = T_world_map @ T_map_cam(t)

World frame = the ID-13 marker frame, so the marker sits at (0,0,0). For a marker
flat on a table, the camera should be on the +Z side (marker normal) ~0.3-2 m up.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def quat_to_mat(qx, qy, qz, qw, t):
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx,qy,qz,qw = qx/n,qy/n,qz/n,qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]], float)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx_slam_tag", required=True)
    ap.add_argument("--trajectory", required=True, help="EuRoC traj in map frame")
    ap.add_argument("--out", required=True, help="output world trajectory txt")
    ap.add_argument("--plot", required=True)
    a = ap.parse_args()

    cal = json.load(open(a.tx_slam_tag))
    T_world_map = np.array(cal["T_world_map"], float)
    tag_in_map = np.array(cal["tx_slam_tag"], float)[:3, 3]

    traj = np.loadtxt(a.trajectory)              # ts tx ty tz qx qy qz qw
    ts = traj[:, 0]
    Pm = traj[:, 1:4]                            # camera positions in MAP frame
    Pw = np.zeros_like(Pm)
    out_rows = []
    for i, r in enumerate(traj):
        T_map_cam = quat_to_mat(r[4], r[5], r[6], r[7], r[1:4])
        T_world_cam = T_world_map @ T_map_cam
        Pw[i] = T_world_cam[:3, 3]
        q = mat_to_quat(T_world_cam[:3, :3])
        out_rows.append([r[0], *T_world_cam[:3, 3], q[1], q[2], q[3], q[0]])
    np.savetxt(a.out, np.array(out_rows),
               header="ts tx ty tz qx qy qz qw (WORLD frame)", comments="# ")

    # sanity: marker position transformed to world should be ~(0,0,0)
    marker_world = (T_world_map @ np.array([*tag_in_map, 1.0]))[:3]

    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(131)
    ax1.plot(Pm[:, 0]*100, Pm[:, 2]*100, '-', lw=1); ax1.set_title("MAP frame (X-Z)")
    ax1.set_xlabel("X cm"); ax1.set_ylabel("Z cm"); ax1.axis("equal"); ax1.grid(alpha=.3)
    ax2 = fig.add_subplot(132)
    ax2.plot(Pw[:, 0]*100, Pw[:, 1]*100, '-', lw=1, label="camera path")
    ax2.scatter([0], [0], c='r', s=80, marker='*', label="marker origin")
    ax2.set_title("WORLD frame top-down (X-Y on marker)")
    ax2.set_xlabel("X cm"); ax2.set_ylabel("Y cm"); ax2.axis("equal"); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(Pw[:, 0]*100, Pw[:, 1]*100, Pw[:, 2]*100, lw=1)
    ax3.scatter([0], [0], [0], c='r', s=80, marker='*')
    ax3.set_title("WORLD 3D (cm)"); ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z(up)")
    plt.tight_layout(); plt.savefig(a.plot, dpi=90)

    print(f"frames: {len(traj)}")
    print(f"marker->world check (should be ~0,0,0): {np.round(marker_world*100,2)} cm")
    print(f"camera height above marker plane (world Z): "
          f"min {Pw[:,2].min()*100:.1f}  max {Pw[:,2].max()*100:.1f} cm  (expect positive ~30-200)")
    print(f"world XY extent: {(Pw[:,0].max()-Pw[:,0].min())*100:.1f} x {(Pw[:,1].max()-Pw[:,1].min())*100:.1f} cm")
    print(f"-> {a.out}\n-> {a.plot}")


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i==0: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
        elif i==1: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
        else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    return np.array([w,x,y,z])


if __name__ == "__main__":
    sys.exit(main())
