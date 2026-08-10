#!/usr/bin/env python3
"""
Visualize a UMI episode (or a raw SLAM trajectory) as a 3D path.

Episode CSV (from extract_episode.py): pose in WORLD frame, colored by gripper
width, with the ArUco origin drawn at (0,0,0) and TCP orientation axes sampled
along the path. Headless-friendly: writes a PNG (no display needed over SSH).

    python dataset/plot_episode.py --episode data/episodes/demo.csv --out demo.png
    python dataset/plot_episode.py --traj data/maps/..._frames.txt --out raw.png

--episode expects: ts_ns tx ty tz qx qy qz qw gripper_width_m   (world frame)
--traj    expects: ts tx ty tz qx qy qz qw                      (any frame)
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)


def set_equal_aspect(ax, P):
    c = P.mean(0); r = max((P.max(0) - P.min(0)).max() * 0.5, 0.05)
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode", help="episode CSV in WORLD frame (with gripper width)")
    g.add_argument("--traj", help="raw EuRoC trajectory (ts tx ty tz qx qy qz qw)")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--axes", type=int, default=12, help="number of orientation triads to draw")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    src = a.episode or a.traj
    D = np.loadtxt(src)
    if D.ndim == 1:
        D = D[None, :]
    P = D[:, 1:4]
    Q = D[:, 4:8]
    is_ep = a.episode is not None
    width = D[:, 8] if (is_ep and D.shape[1] >= 9) else None

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # path, colored by gripper width if available else by time
    if width is not None and np.isfinite(width).any():
        c = np.where(np.isfinite(width), width, np.nan) * 100.0  # cm
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, cmap="viridis", s=8)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1); cb.set_label("gripper width (cm)")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="0.7", lw=0.6, alpha=0.6)
    else:
        t = np.linspace(0, 1, len(P))
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=t, cmap="plasma", s=8)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1); cb.set_label("time (normalized)")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="0.7", lw=0.6, alpha=0.6)

    # start / end
    ax.scatter(*P[0], c="green", s=80, marker="o", label="start")
    ax.scatter(*P[-1], c="red", s=80, marker="X", label="end")

    # world origin (ArUco anchor) for episode (world frame)
    if is_ep:
        L = max((P.max(0) - P.min(0)).max() * 0.15, 0.03)
        for vec, col in zip(np.eye(3), ["r", "g", "b"]):
            ax.quiver(0, 0, 0, *(vec * L), color=col, lw=2)
        ax.scatter(0, 0, 0, c="k", s=60, marker="*", label="world origin (ArUco)")

    # TCP orientation triads sampled along the path
    if a.axes > 0 and len(P) > 1:
        L = max((P.max(0) - P.min(0)).max() * 0.06, 0.015)
        idx = np.linspace(0, len(P)-1, min(a.axes, len(P))).astype(int)
        for i in idx:
            R = quat_to_R(Q[i])
            for k, col in enumerate(["r", "g", "b"]):
                d = R[:, k] * L
                ax.plot([P[i,0], P[i,0]+d[0]], [P[i,1], P[i,1]+d[1]],
                        [P[i,2], P[i,2]+d[2]], color=col, lw=1.2, alpha=0.9)

    set_equal_aspect(ax, np.vstack([P, [[0,0,0]]]) if is_ep else P)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ttl = a.title or (("episode (world frame): " if is_ep else "trajectory: ") + os.path.basename(src))
    ax.set_title(ttl)
    ax.legend(loc="upper right", fontsize=8)

    span = (P.max(0) - P.min(0)) * 100
    print(f"frames: {len(P)} | extent (cm): X{span[0]:.1f} Y{span[1]:.1f} Z{span[2]:.1f}")
    if width is not None:
        wv = width[np.isfinite(width)]
        if len(wv): print(f"gripper width (cm): {wv.min()*100:.1f}..{wv.max()*100:.1f} | valid {len(wv)}/{len(P)}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
