"""Visualize an exported ORB-SLAM3 map point cloud (+ trajectory overlay).

The PLY comes from the docker `export_map` tool (see docker/README.md):
  docker run --rm -v $PWD/output/GX014220/slam:/work orb_slam3:gate28-rescue \
      /ORB_SLAM3/Examples/Monocular-Inertial/export_map \
      -s /work/orbslam3_settings.yaml -l /work/map_atlas.osa \
      -o /work/map_points.ply -k /work/keyframes.csv

Usage:
  python -m gopro_vio.map_viz output/GX014220/slam/map_points.ply \
      --traj output/GX014220/slam/camera_trajectory.csv
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ply(path: str) -> np.ndarray:
    """Minimal ASCII PLY reader for x y z [map_id] vertices."""
    with open(path) as f:
        n = 0
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        data = np.loadtxt(f, max_rows=n)
    return data


def robust_limits(p: np.ndarray, lo=1.0, hi=99.0, pad=0.05):
    a, b = np.percentile(p, lo, axis=0), np.percentile(p, hi, axis=0)
    c = (a + b) / 2
    half = (b - a) / 2 * (1 + pad)
    return c - half, c + half


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply")
    ap.add_argument("--traj", default=None, help="camera_trajectory.csv to overlay")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--max-points", type=int, default=120000)
    args = ap.parse_args()

    ply = pathlib.Path(args.ply)
    out = pathlib.Path(args.out) if args.out else ply.parent
    title = args.title or ply.parent.parent.name

    data = load_ply(str(ply))
    pts = data[:, :3]
    if len(pts) > args.max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), args.max_points,
                                                  replace=False)]
    print(f"[map] {len(data)} points loaded")

    traj = None
    if args.traj:
        df = pd.read_csv(args.traj)
        lost = df["is_lost"].astype(str).str.strip().str.lower().eq("true")
        traj = df[~lost][["x", "y", "z"]].to_numpy()

    # vertical axis = smallest extent of the trajectory (fallback: points)
    ref = traj if traj is not None and len(traj) else pts
    ext = np.percentile(ref, 99, axis=0) - np.percentile(ref, 1, axis=0)
    up = int(np.argmin(ext))
    hor = [i for i in range(3) if i != up]
    names = "xyz"

    lo, hi = robust_limits(pts)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    c = pts[:, up]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, c=c, cmap="terrain",
               alpha=0.35, linewidths=0)
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="red", lw=1.5,
                label="trajectory")
        ax.legend()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"map point cloud — {title}")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(pts[:, hor[0]], pts[:, hor[1]], s=0.3, c=pts[:, up],
                cmap="terrain", alpha=0.35, linewidths=0)
    if traj is not None:
        ax2.plot(traj[:, hor[0]], traj[:, hor[1]], color="red", lw=1.5)
    ax2.set_xlim(lo[hor[0]], hi[hor[0]]); ax2.set_ylim(lo[hor[1]], hi[hor[1]])
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.set_xlabel(f"{names[hor[0]]} [m]"); ax2.set_ylabel(f"{names[hor[1]]} [m]")
    ax2.set_title("top view (map + trajectory)")

    fig.tight_layout()
    png = out / "map.png"
    fig.savefig(png, dpi=130)
    print(f"[done] {png}")


if __name__ == "__main__":
    main()
