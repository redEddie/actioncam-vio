#!/usr/bin/env python3
"""
Compute T_cam_tcp (camera -> gripper TCP, a fixed rigid transform) from the two
finger markers + the known marker-center -> fingertip offset.

Per frame (both finger markers visible):
    R   = right-marker rotation  (gripper orientation in camera; marker axes == gripper axes)
    tip_right = t_right + R @ offset_right
    tip_left  = t_left  + R @ offset_left      (offset_left = mirror of right, x flips)
    TCP_pos   = (tip_right + tip_left) / 2      (width-invariant midpoint of a symmetric gripper)
    T_cam_tcp = [R | TCP_pos]
Averaged over frames (geometric-median translation + mean-quaternion rotation).
Only the LEFT marker's center is used, so its mounting orientation is irrelevant.

    conda activate umi
    # 1) detect markers on a clip where BOTH finger markers (6,7) are visible:
    python aruco/detect_aruco.py --db3 <clip>.db3 \
        --intrinsics calibration/factory_calib_<sn>.json --out data/maps/tcp_dets.pkl
    # 2) compute T_cam_tcp:
    python aruco/calibrate_cam_tcp.py --detections data/maps/tcp_dets.pkl \
        --out data/maps/T_cam_tcp.json
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
import numpy as np
import cv2
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rotmat_to_quat(R):  # wxyz
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
    return np.array([w, x, y, z])


def quat_to_mat(q):
    w, x, y, z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)


def mean_quat(Q):
    Q = np.array(Q); Q *= np.sign(Q[:, 0:1] + 1e-12)
    w, v = np.linalg.eigh(Q.T @ Q)
    return v[:, -1]


def geometric_median(P, iters=200, eps=1e-9):
    m = P.mean(0)
    for _ in range(iters):
        d = np.linalg.norm(P - m, axis=1); d = np.where(d < eps, eps, d)
        m2 = (P / d[:, None]).sum(0) / (1/d).sum()
        if np.linalg.norm(m2 - m) < eps:
            break
        m = m2
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True, help="pkl from detect_aruco (both 6,7 visible)")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "aruco_config.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    lid = int(cfg["gripper_left_finger_id"]); rid = int(cfg["gripper_right_finger_id"])
    off_r = np.array(cfg["gripper_marker_to_tip_right_m"], float)
    off_l = off_r * np.array([-1, 1, 1], float)      # mirror across gripper center plane

    dets = pickle.load(open(a.detections, "rb"))
    tcps, quats = [], []
    for d in dets:
        td = d["tag_dict"]
        if lid not in td or rid not in td:
            continue
        R = cv2.Rodrigues(np.asarray(td[rid]["rvec"], float))[0]   # gripper orient. in cam
        t_r = np.asarray(td[rid]["tvec"], float).reshape(3)
        t_l = np.asarray(td[lid]["tvec"], float).reshape(3)
        tip_r = t_r + R @ off_r
        tip_l = t_l + R @ off_l
        tcps.append((tip_r + tip_l) / 2.0)
        quats.append(rotmat_to_quat(R))

    if len(tcps) < 3:
        print(f"ERROR: only {len(tcps)} frames with BOTH markers {lid},{rid} visible. "
              "Record a clip where both finger markers are clearly in view.")
        return 1

    tcps = np.array(tcps)
    tcp_med = geometric_median(tcps)
    R_mean = quat_to_mat(mean_quat(quats))
    T = np.eye(4); T[:3, :3] = R_mean; T[:3, 3] = tcp_med

    spread = np.linalg.norm(tcps - tcp_med, axis=1)
    json.dump({"frames": len(tcps), "T_cam_tcp": T.tolist(),
               "tcp_pos_m": tcp_med.tolist(),
               "tcp_translation_spread_cm": {"median": float(np.median(spread)*100),
                                             "max": float(spread.max()*100)}},
              open(a.out, "w"), indent=2)
    print(f"frames (both markers): {len(tcps)}")
    print(f"TCP in camera (m): {np.round(tcp_med, 4).tolist()}  (dist {np.linalg.norm(tcp_med)*100:.1f}cm)")
    print(f"TCP translation spread: median {np.median(spread)*100:.2f}cm  max {spread.max()*100:.2f}cm")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
