#!/usr/bin/env python3
"""
Compare two ORB-SLAM3 keyframe trajectories (EuRoC fmt: ts tx ty tz qx qy qz qw),
e.g. STEREO vs IMU_STEREO. Matches keyframes by nearest timestamp, aligns rigidly
(Umeyama, no scale -- both are already metric), and reports path length + ATE.

    python slam/compare_trajectories.py traj_stereo.txt traj_imu.txt [--plot out.png]
"""
import argparse, sys
import numpy as np


def load(f):
    d = np.loadtxt(f)
    return d[:, 0], d[:, 1:4]


def umeyama_no_scale(A, B):
    mA, mB = A.mean(0), B.mean(0)
    H = (A - mA).T @ (B - mB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (R @ A.T).T + (mB - R @ mA)


def plen(p):
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_a"); ap.add_argument("traj_b")
    ap.add_argument("--plot")
    a = ap.parse_args()
    ta, pa = load(a.traj_a); tb, pb = load(a.traj_b)
    idx = [int(np.argmin(np.abs(ta - t))) for t in tb]          # match b->nearest a
    pa_m = pa[idx]
    pa_al = umeyama_no_scale(pa_m, pb)
    err = np.linalg.norm(pa_al - pb, axis=1) * 100
    print(f"matched KFs           : {len(pb)}")
    print(f"path length  A / B    : {plen(pa)*100:.1f} / {plen(pb)*100:.1f} cm")
    print(f"ATE (rigid-aligned)   : mean={err.mean():.2f}  rmse={np.sqrt((err**2).mean()):.2f}  max={err.max():.2f} cm")
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for axi, (i, j, lab) in zip(ax, [(0, 2, "X-Z top"), (0, 1, "X-Y"), (2, 1, "Z-Y")]):
            axi.plot(pb[:, i], pb[:, j], "-o", ms=3, label="B")
            axi.plot(pa_al[:, i], pa_al[:, j], "-s", ms=3, alpha=.7, label="A(aligned)")
            axi.set_title(lab); axi.axis("equal"); axi.grid(alpha=.3); axi.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(a.plot, dpi=90); print(f"plot -> {a.plot}")


if __name__ == "__main__":
    sys.exit(main())
