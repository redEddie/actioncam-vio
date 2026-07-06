"""Trajectory visualization + statistics for ORB-SLAM3 outputs.

Usage:
  python -m gopro_vio.viz output/GX014217/slam/camera_trajectory.csv -o output/GX014217/slam
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ORB-SLAM3 Tracking::eTrackingState
STATES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED",
          2: "OK", 3: "RECENTLY_LOST", 4: "LOST", 5: "OK_KLT"}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # column contains "true"/"false" strings; pandas may or may not have
    # coerced them to bool depending on version
    df["is_lost"] = (df["is_lost"].astype(str).str.strip().str.lower()
                     .eq("true"))
    return df


def stats(df: pd.DataFrame) -> dict:
    ok = df[~df["is_lost"]]
    xyz = ok[["x", "y", "z"]].to_numpy()
    dist = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0
    per_state = {STATES.get(s, str(s)): int(n)
                 for s, n in df["state"].value_counts().items()}
    out = {
        "n_frames": int(len(df)),
        "n_tracked": int(len(ok)),
        "tracked_ratio": float(len(ok) / max(len(df), 1)),
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
        "path_length_m": dist,
        "extent_m": (xyz.max(0) - xyz.min(0)).tolist() if len(xyz) else None,
        "states": per_state,
    }
    # RECENTLY_LOST frames carry IMU dead-reckoning guesses that can drift
    # far; report visually-confirmed (state OK) path separately
    ok_vis = ok[ok["state"] == 2]
    p = ok_vis[["x", "y", "z"]].to_numpy()
    out["path_length_visual_ok_m"] = (
        float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0)
    out["n_visual_ok"] = int(len(ok_vis))

    if "map_id" in df.columns:
        per_map = {}
        for mid, g in ok.groupby("map_id"):
            gv = g[g["state"] == 2]
            p = gv[["x", "y", "z"]].to_numpy()
            per_map[int(mid)] = {
                "n_frames": int(len(g)),
                "n_visual_ok": int(len(gv)),
                "t_start": float(g["timestamp"].iloc[0]),
                "t_end": float(g["timestamp"].iloc[-1]),
                "path_length_visual_ok_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0,
            }
        out["maps"] = per_map
    return out


def plot(df: pd.DataFrame, out_png: str, title=""):
    ok = df[~df["is_lost"]]
    xyz = ok[["x", "y", "z"]].to_numpy()
    t = ok["timestamp"].to_numpy()
    # ORB-SLAM3's gravity-aligned world puts "up" on one axis; auto-detect it
    # as the axis with the smallest extent and use the other two for top view
    ext = xyz.max(0) - xyz.min(0) if len(xyz) else np.ones(3)
    up = int(np.argmin(ext))
    hor = [i for i in range(3) if i != up]
    names = "xyz"

    # per-submap coloring when a map_id column is present (allmaps export)
    seg_color = None
    if "map_id" in ok.columns and ok["map_id"].nunique() > 1:
        seg_color = ok["map_id"].to_numpy()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    if len(xyz):
        if seg_color is not None:
            p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=seg_color,
                           s=1, cmap="tab10")
            fig.colorbar(p, ax=ax, label="map_id", shrink=0.6)
        else:
            p = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, s=1,
                           cmap="viridis")
            fig.colorbar(p, ax=ax, label="t [s]", shrink=0.6)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.3, alpha=0.4, color="k")
    ax.set_title(f"3D trajectory {title}")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    try:
        ax.set_box_aspect([1, 1, 1])
        lim = np.array([xyz.min(0), xyz.max(0)])
        c, r = lim.mean(0), (lim[1] - lim[0]).max() / 2 + 1e-6
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
    except Exception:
        pass

    ax2 = fig.add_subplot(1, 3, 2)
    if len(xyz):
        ax2.scatter(xyz[:, hor[0]], xyz[:, hor[1]],
                    c=seg_color if seg_color is not None else t, s=1,
                    cmap="tab10" if seg_color is not None else "viridis")
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_title(f"top view ({names[hor[0]]}-{names[hor[1]]})")
    ax2.set_xlabel(f"{names[hor[0]]} [m]")
    ax2.set_ylabel(f"{names[hor[1]]} [m]")

    ax3 = fig.add_subplot(1, 3, 3)
    lost = df[df["is_lost"]]
    for i in range(3):
        ax3.plot(t, xyz[:, i], lw=0.8, label="xyz"[i])
    if len(lost):
        ax3.scatter(lost["timestamp"], np.zeros(len(lost)), marker="|",
                    color="r", s=40, label="lost")
    ax3.grid(alpha=0.3); ax3.legend(); ax3.set_xlabel("t [s]"); ax3.set_ylabel("[m]")
    ax3.set_title("position vs time")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    csv_path = pathlib.Path(args.csv)
    out = pathlib.Path(args.out) if args.out else csv_path.parent
    out.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    s = stats(df)
    print(json.dumps(s, indent=2))
    (out / "trajectory_stats.json").write_text(json.dumps(s, indent=2))
    title = args.title or csv_path.parent.name
    plot(df, str(out / "trajectory.png"), title)
    print(f"[done] {out/'trajectory.png'}")


if __name__ == "__main__":
    main()
