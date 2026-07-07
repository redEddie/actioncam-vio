"""Calibration quality report: coverage, radial reprojection error, FOV.

Usage:
  python -m gopro_vio.calib_report calibration_acepro2 --square-size 0.023
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from .charuco import make_board


def kb4_theta_d(theta, D):
    t2 = theta * theta
    return theta * (1 + D[0]*t2 + D[1]*t2**2 + D[2]*t2**3 + D[3]*t2**4)


def report(calib_dir: str, square_size=0.023):
    d = pathlib.Path(calib_dir)
    calib = json.loads((d / "intrinsics.json").read_text())
    K = np.array(calib["K"]); D = np.array(calib["D"])
    W, H = calib["image_size"]
    z = np.load(d / "detections_cache.npz", allow_pickle=True)
    dets = list(z["dets"])
    bp = np.load(d / "board_poses.npz")
    board = make_board(tuple(calib["board"]["layout"]), square_size,
                       calib["board"]["dict"], calib["board"]["legacy"])
    obj_all = board.getChessboardCorners()

    # --- coverage ---
    gw, gh = 10, 8
    grid = np.zeros((gh, gw), int)
    allc = np.concatenate([dd["corners"] for dd in dets])
    gx = np.clip((allc[:, 0]/W*gw).astype(int), 0, gw-1)
    gy = np.clip((allc[:, 1]/H*gh).astype(int), 0, gh-1)
    np.add.at(grid, (gy, gx), 1)
    cov_cells = int((grid > 100).sum())
    cx, cy = K[0, 2], K[1, 2]
    r = np.hypot(allc[:, 0]-cx, allc[:, 1]-cy)
    rmax = np.hypot(max(cx, W-cx), max(cy, H-cy))
    radial_p99 = float(np.percentile(r, 99) / rmax)

    # --- radial reprojection error (all frames, stored poses) ---
    f2i = {int(f): i for i, f in enumerate(bp["frame"])}
    errs, rads = [], []
    for dd in dets[::3]:
        i = f2i.get(int(dd["frame"]))
        if i is None:
            continue
        proj, _ = cv2.fisheye.projectPoints(
            obj_all[dd["ids"]].astype(np.float64).reshape(1, -1, 3),
            bp["rvec"][i], bp["tvec"][i], K, D.reshape(4, 1))
        e = np.linalg.norm(proj.reshape(-1, 2) - dd["corners"], axis=1)
        errs.append(e)
        rads.append(np.hypot(*(dd["corners"] - [cx, cy]).T))
    errs = np.concatenate(errs); rads = np.concatenate(rads)
    bins = np.linspace(0, rmax, 6)
    radial_err = []
    for a, b in zip(bins, bins[1:]):
        m = (rads >= a) & (rads < b)
        if m.sum() > 50:
            radial_err.append({"r_px": [float(a), float(b)],
                               "mean_px": float(errs[m].mean()),
                               "n": int(m.sum())})

    # --- FOV from the KB4 model ---
    # dense-grid crossing search instead of brentq: the fitted polynomial can
    # roll over beyond the sampled radius, leaving no bracketable root
    fx = K[0, 0]
    th_grid = np.linspace(1e-3, 2.2, 4000)
    r_grid = fx * kb4_theta_d(th_grid, D)
    mono_end = int(np.argmax(np.diff(r_grid) <= 0)) or len(r_grid) - 1
    def solve_fov(target_px):
        rg = r_grid[:mono_end + 1]
        if target_px > rg[-1]:
            return None  # model rolls over before reaching this radius
        return float(2 * np.degrees(np.interp(target_px, rg,
                                              th_grid[:mono_end + 1])))
    fov = {"H": solve_fov(max(cx, W-1-cx)), "V": solve_fov(max(cy, H-1-cy)),
           "D": solve_fov(np.hypot(max(cx, W-1-cx), max(cy, H-1-cy)))}
    fov_note = None
    if any(v is None for v in fov.values()):
        fov_note = ("model not valid out to the farthest corner "
                    f"(monotonic to r={r_grid[mono_end]:.0f}px, "
                    f"theta={np.degrees(th_grid[mono_end]):.1f}deg) — "
                    "null entries exceed the calibrated/extrapolable range")

    out = {
        "image_size": [W, H],
        "rms_px_calib_views": calib["rms_px"],
        "reproj_all_frames": {"mean_px": float(errs.mean()),
                              "median_px": float(np.median(errs)),
                              "p95_px": float(np.percentile(errs, 95))},
        "radial_error": radial_err,
        "coverage_cells_gt100": f"{cov_cells}/{gw*gh}",
        "radial_coverage_p99": radial_p99,
        "fov_deg": {k: (round(v, 1) if v is not None else None) for k, v in fov.items()},
        "fov_note": fov_note,
        "fx_fy_ratio": float(K[0, 0]/K[1, 1]),
        "principal_offset_px": [float(cx - W/2), float(cy - H/2)],
    }
    (d / "calib_report.json").write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im = axes[0].imshow(grid, cmap="viridis")
    axes[0].set_title("corner coverage (10x8 cells)")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    rr = [(e["r_px"][0]+e["r_px"][1])/2 for e in radial_err]
    axes[1].plot(rr, [e["mean_px"] for e in radial_err], "o-")
    axes[1].set_xlabel("radius from principal point [px]")
    axes[1].set_ylabel("mean reproj error [px]"); axes[1].grid(alpha=0.3)
    axes[1].set_title("radial error profile")
    th = np.linspace(0, np.radians(fov["D"] if fov["D"] else 120)/2, 100)
    axes[2].plot(np.degrees(th), fx*kb4_theta_d(th, D))
    axes[2].axhline(max(cx, W-1-cx), ls="--", c="gray", label="img half-width")
    axes[2].axhline(np.hypot(max(cx, W-1-cx), max(cy, H-1-cy)), ls=":",
                    c="gray", label="img corner")
    axes[2].set_xlabel("theta [deg]"); axes[2].set_ylabel("radius [px]")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title('KB4 curve — FOV H %s / D %s' % tuple((f'{v:.0f}°' if v else 'n/a') for v in (fov['H'], fov['D'])))
    fig.tight_layout()
    fig.savefig(d / "calib_report.png", dpi=130)
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calib_dir")
    ap.add_argument("--square-size", type=float, default=0.023)
    args = ap.parse_args()
    report(args.calib_dir, args.square_size)


if __name__ == "__main__":
    main()
