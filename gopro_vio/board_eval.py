"""Evaluate a SLAM trajectory against the ChArUco board reference.

The calibration video gives a mm-accurate metric camera trajectory for free
(solvePnP against the 0.023 m board, saved in board_poses.npz).  This tool
aligns the SLAM trajectory to that reference (rigid Umeyama on the shared
video clock) and reports ATE in cm — i.e. how well close-range VIO tracks.

Usage:
  python -m gopro_vio.board_eval calibration/board_poses.npz \
      output/GX014222/slam/camera_trajectory.csv -o output/GX014222/eval
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
from scipy.spatial.transform import Rotation as R


def board_trajectory(npz_path: str):
    """Camera position in the (metric, static) board frame per frame."""
    z = np.load(npz_path)
    rot_cb = R.from_rotvec(z["rvec"])                # board -> camera
    p = -np.einsum("nij,nj->ni", rot_cb.inv().as_matrix(), z["tvec"])
    return z["t"], p, z["n_corners"]


def umeyama(A: np.ndarray, B: np.ndarray, with_scale=False):
    """Least-squares similarity/rigid transform mapping A -> B."""
    mA, mB = A.mean(0), B.mean(0)
    Ac, Bc = A - mA, B - mB
    H = Ac.T @ Bc / len(A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (S * [1, 1, d]).sum() / (Ac ** 2).sum() * len(A) if with_scale else 1.0
    t = mB - s * Rm @ mA
    return s, Rm, t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("board_poses")
    ap.add_argument("slam_csv")
    ap.add_argument("-o", "--out", default="output/GX014222/eval")
    ap.add_argument("--min-corners", type=int, default=20,
                    help="only reference poses with this many ChArUco corners")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--ok-only", action="store_true",
                    help="use only state==OK poses (exclude RECENTLY_LOST "
                         "IMU dead-reckoning guesses)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_ref, p_ref, ncorn = board_trajectory(args.board_poses)
    good = ncorn >= args.min_corners
    t_ref, p_ref = t_ref[good], p_ref[good]
    print(f"[ref] {len(t_ref)} board poses (>= {args.min_corners} corners)")

    df = pd.read_csv(args.slam_csv)
    lost = df["is_lost"].astype(str).str.strip().str.lower().eq("true")
    df = df[~lost]
    if args.ok_only:
        df = df[df["state"] == 2]
    if args.t_min is not None:
        df = df[df["timestamp"] >= args.t_min]
    if args.t_max is not None:
        df = df[df["timestamp"] <= args.t_max]
    t_slam = df["timestamp"].to_numpy()
    p_slam = df[["x", "y", "z"]].to_numpy()
    print(f"[slam] {len(df)} tracked poses, t {t_slam[0]:.1f}-{t_slam[-1]:.1f}s")

    # associate on the shared video clock (both are frame_idx/fps of the
    # same video); nearest-neighbor within half a reference frame period
    order = np.argsort(t_ref)
    t_ref, p_ref = t_ref[order], p_ref[order]
    idx = np.searchsorted(t_ref, t_slam)
    idx = np.clip(idx, 1, len(t_ref) - 1)
    left_closer = (t_slam - t_ref[idx - 1]) < (t_ref[idx] - t_slam)
    idx = idx - left_closer.astype(int)
    dt = np.abs(t_ref[idx] - t_slam)
    m = dt < 0.010  # 10 ms
    A, B, T = p_slam[m], p_ref[idx[m]], t_slam[m]
    print(f"[assoc] {m.sum()} pose pairs (|dt|<10 ms)")

    results = {}
    for label, with_scale in [("rigid (scale=1, metric test)", False),
                              ("similarity (scale free)", True)]:
        s, Rm, tv = umeyama(A, B, with_scale)
        err = np.linalg.norm((s * A @ Rm.T + tv) - B, axis=1)
        results[label] = {
            "scale": float(s),
            "ate_rmse_cm": float(np.sqrt((err ** 2).mean()) * 100),
            "ate_mean_cm": float(err.mean() * 100),
            "ate_median_cm": float(np.median(err) * 100),
            "ate_p95_cm": float(np.percentile(err, 95) * 100),
            "ate_max_cm": float(err.max() * 100),
        }
        print(f"[{label}] scale={s:.4f}  RMSE={results[label]['ate_rmse_cm']:.2f}cm  "
              f"median={results[label]['ate_median_cm']:.2f}cm  "
              f"p95={results[label]['ate_p95_cm']:.2f}cm")

    # plot with the rigid alignment
    s, Rm, tv = umeyama(A, B, False)
    A_al = A @ Rm.T + tv
    err = np.linalg.norm(A_al - B, axis=1)

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(*B.T, lw=1.2, color="k", label="ChArUco reference (mm-grade)")
    ax.plot(*A_al.T, lw=1.0, color="tab:red", alpha=0.8, label="VIO (ORB-SLAM3)")
    ax.legend(); ax.set_title("close-range trajectory [m]")
    ax.set_box_aspect([1, 1, 1])
    lim = np.array([B.min(0), B.max(0)])
    c, r = lim.mean(0), (lim[1] - lim[0]).max() / 2 * 1.1
    ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(B[:, 0], B[:, 1], lw=1.2, color="k")
    ax2.plot(A_al[:, 0], A_al[:, 1], lw=1.0, color="tab:red", alpha=0.8)
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_title("top view (board x-y) [m]")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(T, err * 100, lw=0.8)
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("position error [cm]")
    ax3.grid(alpha=0.3)
    ax3.set_title(f"ATE: RMSE {results['rigid (scale=1, metric test)']['ate_rmse_cm']:.2f} cm, "
                  f"median {results['rigid (scale=1, metric test)']['ate_median_cm']:.2f} cm")
    fig.tight_layout()
    fig.savefig(out / "board_eval.png", dpi=130)

    (out / "board_eval.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "aligned.npz", t=T, vio=A_al, ref=B, err=err)
    print(f"[done] {out}/board_eval.png, board_eval.json")


if __name__ == "__main__":
    main()
