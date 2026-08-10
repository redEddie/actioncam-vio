#!/usr/bin/env python3
"""
Compute T_world<-map (the SLAM-map -> world/origin-tag transform) from ArUco
detections + the per-frame SLAM trajectory, following UMI's calibrate_slam_tag.

For every frame where the origin tag (id 13) is seen:
    tx_slam_tag = tx_slam_cam @ tx_cam_tag
then take a robust average (geometric-median translation + mean-quaternion rotation).
World frame = origin tag frame, so  T_world_map = inv(tx_slam_tag).

    conda activate umi
    python aruco/calibrate_slam_tag.py \
        --detections data/maps/tag_detections.pkl \
        --trajectory data/maps/localize_<...>_full.txt \
        --out data/maps/tx_slam_tag.json [--tag_id 13]
"""
from __future__ import annotations
import argparse, json, pickle, sys
import numpy as np
import cv2


def pose_to_mat(rvec, tvec):
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, float))
    T[:3, 3] = np.asarray(tvec, float).reshape(3)
    return T


def quat_to_mat(q_xyzw, t):
    x, y, z, w = q_xyzw
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], float)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def geometric_median(P, iters=200, eps=1e-8):
    m = P.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(P - m, axis=1)
        d = np.where(d < eps, eps, d)
        w = 1.0 / d
        m_new = (P * w[:, None]).sum(0) / w.sum()
        if np.linalg.norm(m_new - m) < eps:
            break
        m = m_new
    return m


def mean_rotation(Rs):
    # chordal L2 mean via quaternion eigenvector (Markley)
    Q = []
    for R in Rs:
        q = cv2.Rodrigues(R)[0]  # not a quat; build quat from R directly
        Q.append(rotmat_to_quat(R))
    Q = np.array(Q)
    Q *= np.sign(Q[:, 0:1] + 1e-12)  # hemisphere
    A = Q.T @ Q
    w, v = np.linalg.eigh(A)
    q = v[:, -1]
    return quat_wxyz_to_mat(q)


def rotmat_to_quat(R):  # returns wxyz
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1]-R[1, 2])/s; y = (R[0, 2]-R[2, 0])/s; z = (R[1, 0]-R[0, 1])/s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1+R[0, 0]-R[1, 1]-R[2, 2])*2
            w = (R[2, 1]-R[1, 2])/s; x = 0.25*s; y = (R[0, 1]+R[1, 0])/s; z = (R[0, 2]+R[2, 0])/s
        elif i == 1:
            s = np.sqrt(1+R[1, 1]-R[0, 0]-R[2, 2])*2
            w = (R[0, 2]-R[2, 0])/s; x = (R[0, 1]+R[1, 0])/s; y = 0.25*s; z = (R[1, 2]+R[2, 1])/s
        else:
            s = np.sqrt(1+R[2, 2]-R[0, 0]-R[1, 1])*2
            w = (R[1, 0]-R[0, 1])/s; x = (R[0, 2]+R[2, 0])/s; y = (R[1, 2]+R[2, 1])/s; z = 0.25*s
    return np.array([w, x, y, z])


def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--trajectory", required=True, help="per-frame EuRoC traj (_full.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag_id", type=int, default=13)
    ap.add_argument("--min_dist", type=float, default=0.3)
    ap.add_argument("--max_dist", type=float, default=2.0)
    args = ap.parse_args()

    dets = pickle.load(open(args.detections, "rb"))
    traj = np.loadtxt(args.trajectory)  # ts tx ty tz qx qy qz qw
    t_traj = traj[:, 0]
    # ORB-SLAM3 SaveTrajectoryEuRoC writes timestamps in nanoseconds (~1e18);
    # detect_aruco stores seconds (~1e9). Align to seconds.
    if np.median(t_traj) > 1e15:
        t_traj = t_traj * 1e-9

    samples = []
    for d in dets:
        td = d["tag_dict"]
        if args.tag_id not in td:
            continue
        tag = td[args.tag_id]
        tx_cam_tag = pose_to_mat(tag["rvec"], tag["tvec"])
        dist = np.linalg.norm(tx_cam_tag[:3, 3])
        if dist < args.min_dist or dist > args.max_dist:
            continue
        j = int(np.argmin(np.abs(t_traj - d["time"])))
        if abs(t_traj[j] - d["time"]) > 0.05:     # no SLAM pose within 50 ms
            continue
        r = traj[j]
        tx_slam_cam = quat_to_mat(r[4:8], r[1:4])
        samples.append(tx_slam_cam @ tx_cam_tag)

    if len(samples) < 3:
        print(f"ERROR: only {len(samples)} usable tag-{args.tag_id} samples. "
              "Record the origin marker more (0.3-2 m, centered, multiple angles).")
        return 1

    S = np.array(samples)
    t_med = geometric_median(S[:, :3, 3])
    R_mean = mean_rotation([s[:3, :3] for s in S])
    tx_slam_tag = np.eye(4); tx_slam_tag[:3, :3] = R_mean; tx_slam_tag[:3, 3] = t_med
    T_world_map = np.linalg.inv(tx_slam_tag)

    # spread diagnostics
    spread = np.linalg.norm(S[:, :3, 3] - t_med, axis=1)
    out = {"tag_id": args.tag_id, "n_samples": len(samples),
           "tx_slam_tag": tx_slam_tag.tolist(), "T_world_map": T_world_map.tolist(),
           "translation_spread_cm": {"median": float(np.median(spread)*100),
                                     "max": float(spread.max()*100)}}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"samples: {len(samples)} | translation spread: median "
          f"{np.median(spread)*100:.2f}cm max {spread.max()*100:.2f}cm")
    print(f"tag origin in SLAM map (m): {np.round(t_med,4).tolist()}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
