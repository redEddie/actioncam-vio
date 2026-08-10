#!/usr/bin/env python3
"""
Clean a raw UMI episode CSV: reject transient SLAM glitches, trim, optionally
smooth. Order matters: outlier rejection FIRST (smoothing a glitch smears it),
then trim, then optional light smoothing for VLA action labels.

    python dataset/clean_episode.py data/episodes/demoA01.csv \
        --out data/episodes/demoA01_clean.csv --max_step_cm 2 --smooth 5

Glitch rejection uses a PHYSICAL per-frame step cap (a hand cannot jump several
cm between 30 Hz frames), which is robust regardless of how many glitches a demo
has -- unlike a fixed q99 that assumes a constant ~1% outlier rate. (q01/q99 is
the right tool later, for action-value normalization in the LeRobot export.)

CSV columns: ts_ns tx ty tz qx qy qz qw gripper_width_m  (world frame).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def reject_glitches(P, max_step_m):
    """Keep frame i only if it is within max_step of the last KEPT frame.
    Removes isolated jump-out/jump-back spikes; real (sub-cap) motion survives."""
    keep = [0]
    last = 0
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[last]) <= max_step_m:
            keep.append(i); last = i
    return np.array(keep, int)


def smooth_traj(P, Q, win):
    """Centered moving average on translation; sign-aligned mean on quaternions."""
    if win < 3:
        return P, Q
    k = win // 2
    Ps = P.copy()
    Qs = Q.copy()
    # sign-align quaternions to a running reference (avoid double-cover flips)
    Qa = Q.copy()
    for i in range(1, len(Qa)):
        if np.dot(Qa[i], Qa[i-1]) < 0:
            Qa[i] = -Qa[i]
    for i in range(len(P)):
        lo, hi = max(0, i-k), min(len(P), i+k+1)
        Ps[i] = P[lo:hi].mean(0)
        q = Qa[lo:hi].mean(0)
        Qs[i] = q / np.linalg.norm(q)
    return Ps, Qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="raw episode CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_step_cm", type=float, default=2.0,
                    help="reject frames jumping more than this from the last kept frame")
    ap.add_argument("--trim_start", type=int, default=0, help="drop N leading frames")
    ap.add_argument("--trim_end", type=int, default=0, help="drop N trailing frames")
    ap.add_argument("--smooth", type=int, default=0, help="moving-average window (0=off)")
    a = ap.parse_args()

    D = np.loadtxt(a.episode)
    if D.ndim == 1:
        D = D[None, :]
    n0 = len(D)
    P0 = D[:, 1:4]
    path0 = np.linalg.norm(np.diff(P0, axis=0), axis=1).sum() * 100

    # 1) reject transient glitches
    keep = reject_glitches(D[:, 1:4], a.max_step_cm / 100.0)
    D = D[keep]
    n_rej = n0 - len(D)

    # 2) trim
    s, e = a.trim_start, len(D) - a.trim_end
    D = D[s:e]

    # 3) optional smoothing
    P, Q = D[:, 1:4], D[:, 4:8]
    if a.smooth >= 3:
        P, Q = smooth_traj(P, Q, a.smooth)
        D[:, 1:4], D[:, 4:8] = P, Q

    path1 = np.linalg.norm(np.diff(D[:, 1:4], axis=0), axis=1).sum() * 100
    ext = (D[:, 1:4].max(0) - D[:, 1:4].min(0)) * 100
    disp = np.linalg.norm(D[-1, 1:4] - D[0, 1:4]) * 100
    w = D[:, 8] * 100; wv = w[np.isfinite(w)]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    hdr = "ts_ns tx ty tz qx qy qz qw gripper_width_m  (world frame, cleaned)"
    np.savetxt(a.out, D, header=hdr, comments="# ")

    print(f"{os.path.basename(a.episode)}: {n0} -> {len(D)} frames "
          f"(rejected {n_rej} glitch, trimmed {a.trim_start}+{a.trim_end})")
    print(f"  path length {path0:.1f} -> {path1:.1f} cm | extent X{ext[0]:.1f} Y{ext[1]:.1f} Z{ext[2]:.1f} cm "
          f"| start->end {disp:.1f} cm")
    if len(wv):
        print(f"  grip width: {wv.min():.1f}..{wv.max():.1f} cm (start {w[0]:.1f}, end {w[-1]:.1f})")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
