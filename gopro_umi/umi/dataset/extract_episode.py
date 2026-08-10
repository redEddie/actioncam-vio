#!/usr/bin/env python3
"""
Extract a UMI episode: gripper 6DoF pose in WORLD + gripper width, per frame.

    T_world_tcp(t) = T_world_map @ T_map_cam(t) @ T_cam_tcp
    gripper_width(t) = (x_right - x_left) - offset   [from finger markers 6/7]

Inputs:
  --trajectory   per-frame SLAM trajectory (EuRoC: ts tx ty tz qx qy qz qw) in MAP frame
                 (from `run_slam.sh localize <demo> ... ` -> *_full.txt)
  --t_world_map  tx_slam_tag.json (has "T_world_map")        [from calibrate_slam_tag]
  --t_cam_tcp    T_cam_tcp.json (has "T_cam_tcp")            [from calibrate_cam_tcp]
  --detections   tag_detections.pkl for this demo           [from detect_aruco]
Output: episode CSV  ts_ns,tx,ty,tz,qx,qy,qz,qw,gripper_width_m  (pose in WORLD frame)
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quat_to_mat(qx, qy, qz, qw, t):
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx,qy,qz,qw = qx/n,qy/n,qz/n,qw/n
    R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                  [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                  [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]], float)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=t; return T


def mat_to_quat(R):  # xyzw
    t = np.trace(R)
    if t > 0:
        s=np.sqrt(t+1)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i=int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i==0: s=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s;x=.25*s;y=(R[0,1]+R[1,0])/s;z=(R[0,2]+R[2,0])/s
        elif i==1: s=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s;x=(R[0,1]+R[1,0])/s;y=.25*s;z=(R[1,2]+R[2,1])/s
        else: s=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s;x=(R[0,2]+R[2,0])/s;y=(R[1,2]+R[2,1])/s;z=.25*s
    return np.array([x,y,z,w])


def build_width_lookup(dets, cfg):
    lid = int(cfg["gripper_left_finger_id"]); rid = int(cfg["gripper_right_finger_id"])
    off = float(cfg.get("gripper_width_offset_m", 0.0))
    # auto nominal_z = median finger-marker depth; generous tolerance
    zs = [d["tag_dict"][i]["tvec"][2] for d in dets for i in (lid, rid) if i in d["tag_dict"]]
    zc = float(np.median(zs)) if zs else None
    ztol = 0.03
    times, widths = [], []
    for d in dets:
        td = d["tag_dict"]
        w = None
        if lid in td and rid in td:
            zl, zr = td[lid]["tvec"][2], td[rid]["tvec"][2]
            if zc is None or (abs(zl-zc) < ztol and abs(zr-zc) < ztol):
                w = float(td[rid]["tvec"][0] - td[lid]["tvec"][0]) - off
        times.append(d["time"]); widths.append(w)
    return np.array(times), widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--t_world_map", required=True)
    ap.add_argument("--t_cam_tcp", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "aruco_config.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    T_world_map = np.array(json.load(open(a.t_world_map))["T_world_map"], float)
    T_cam_tcp = np.array(json.load(open(a.t_cam_tcp))["T_cam_tcp"], float)
    dets = pickle.load(open(a.detections, "rb"))
    det_t, det_w = build_width_lookup(dets, cfg)

    traj = np.loadtxt(a.trajectory)
    ts = traj[:, 0]
    ts_s = ts * 1e-9 if np.median(ts) > 1e15 else ts   # EuRoC ns -> s

    rows, n_w = [], 0
    for i, r in enumerate(traj):
        T_map_cam = quat_to_mat(r[4], r[5], r[6], r[7], r[1:4])
        T_world_tcp = T_world_map @ T_map_cam @ T_cam_tcp
        q = mat_to_quat(T_world_tcp[:3, :3]); p = T_world_tcp[:3, 3]
        # nearest gripper width within 50 ms
        j = int(np.argmin(np.abs(det_t - ts_s[i])))
        w = det_w[j] if abs(det_t[j] - ts_s[i]) < 0.05 else None
        if w is not None: n_w += 1
        # store timestamp as nanoseconds (int seconds truncated sub-second info,
        # collapsing dt to 0 within the same second -> breaks velocity/resampling).
        ts_ns = int(round(ts_s[i] * 1e9))
        rows.append([ts_ns, *p, *q, w if w is not None else np.nan])

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    hdr = "ts_ns tx ty tz qx qy qz qw gripper_width_m  (pose in WORLD frame)"
    np.savetxt(a.out, np.array(rows, float), header=hdr, comments="# ")
    P = np.array([r[1:4] for r in rows])
    ws = np.array([r[8] for r in rows]); wv = ws[~np.isnan(ws)]
    print(f"frames: {len(rows)} | width valid: {n_w}")
    print(f"TCP world extent (cm): X{(P[:,0].max()-P[:,0].min())*100:.1f} "
          f"Y{(P[:,1].max()-P[:,1].min())*100:.1f} Z{(P[:,2].max()-P[:,2].min())*100:.1f}")
    if len(wv): print(f"gripper width (cm): min {wv.min()*100:.1f}  max {wv.max()*100:.1f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
