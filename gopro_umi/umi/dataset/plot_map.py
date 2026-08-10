#!/usr/bin/env python3
"""
Visualize SLAM map points (and optionally the camera trajectory) to inspect map
density. Headless-friendly: writes a PNG.

    python dataset/plot_map.py --mappoints <traj>_mappoints.txt \
        [--traj <traj>_frames.txt] [--t_world_map tx_slam_tag.json] --out map.png

mappoints file: "x y z" per line (map frame), from the offline driver.
traj file:      EuRoC "ts tx ty tz qx qy qz qw" (camera path, map frame).
--t_world_map:  tx_slam_tag.json -> transform points+path into the ArUco id13
                WORLD frame and draw the marker origin at (0,0,0).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mappoints", required=True)
    ap.add_argument("--traj", default=None, help="optional camera trajectory overlay")
    ap.add_argument("--t_world_map", default=None,
                    help="tx_slam_tag.json -> draw in ArUco id13 world frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    M = np.loadtxt(a.mappoints)
    if M.ndim == 1:
        M = M[None, :]

    # optional: transform map frame -> ArUco id13 world frame
    Twm = None
    if a.t_world_map:
        Twm = np.array(json.load(open(a.t_world_map))["T_world_map"], float)
        M = (Twm[:3, :3] @ M.T).T + Twm[:3, 3]
    # robust extent: clip extreme outliers for the view (keep all for the count)
    n_total = len(M)
    lo, hi = np.percentile(M, 1, axis=0), np.percentile(M, 99, axis=0)
    keep = np.all((M >= lo - 0.5) & (M <= hi + 0.5), axis=1)
    Mv = M[keep] if keep.sum() > 10 else M

    # camera path (transform to world frame too, if requested)
    P = None
    if a.traj:
        T = np.loadtxt(a.traj)
        if T.ndim == 1:
            T = T[None, :]
        P = T[:, 1:4]
        if Twm is not None:
            P = (Twm[:3, :3] @ P.T).T + Twm[:3, 3]

    world = Twm is not None
    olen = max((Mv.max(0) - Mv.min(0)).max() * 0.12, 0.05)  # origin axis length

    fig = plt.figure(figsize=(13, 6))

    # 3D view
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(Mv[:, 0], Mv[:, 1], Mv[:, 2], s=2, c="steelblue", alpha=0.5)
    if P is not None:
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="crimson", lw=1.2, label="camera path")
    if world:
        for vec, col in zip(np.eye(3), ["r", "g", "b"]):
            ax.quiver(0, 0, 0, *(vec * olen), color=col, lw=2)
        ax.scatter(0, 0, 0, c="k", s=70, marker="*", label="id13 origin")
    ax.legend(fontsize=8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3D map points" + (" (id13 world frame)" if world else " (SLAM map frame)"))

    # top-down (X-Z, the typical ground plane for a forward-looking cam)
    ax2 = fig.add_subplot(122)
    ax2.scatter(Mv[:, 0], Mv[:, 2], s=2, c="steelblue", alpha=0.5)
    if P is not None:
        ax2.plot(P[:, 0], P[:, 2], color="crimson", lw=1.2)
    if world:
        ax2.plot([0, olen], [0, 0], "r", lw=2); ax2.plot([0, 0], [0, olen], "b", lw=2)
        ax2.scatter(0, 0, c="k", s=70, marker="*", label="id13 origin"); ax2.legend(fontsize=8)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); ax2.set_aspect("equal")
    ax2.set_title("top-down (X-Z)")
    ax2.grid(alpha=0.3)

    span = Mv.max(0) - Mv.min(0)
    fig.suptitle(a.title or f"{os.path.basename(a.mappoints)} | {n_total} map points | "
                 f"extent {span[0]:.2f}x{span[1]:.2f}x{span[2]:.2f} m")
    print(f"map points: {n_total} | extent (m): X{span[0]:.2f} Y{span[1]:.2f} Z{span[2]:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=130)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
