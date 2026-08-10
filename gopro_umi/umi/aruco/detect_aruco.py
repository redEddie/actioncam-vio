#!/usr/bin/env python3
"""
Detect ArUco markers (DICT_4X4_50) in the LEFT-IR frames of a RealSense .db3 and
estimate each marker's pose in the camera (IR-left) frame.

LEFT IR is used because ORB-SLAM3 tracks on it -> tag poses (tx_cam_tag) and SLAM
poses (tx_slam_cam) share the same camera frame, which calibrate_slam_tag needs.

Pose via solvePnP (estimatePoseSingleMarkers is removed in OpenCV >= 4.8).

    conda activate umi
    python aruco/detect_aruco.py \
        --db3 data/recordings/<mapmarker>.db3 \
        --intrinsics calibration/factory_calib_<sn>.json \
        --config configs/aruco_config.yaml \
        --out data/maps/tag_detections.pkl
"""
from __future__ import annotations
import argparse, glob, json, os, pickle, re, shutil, subprocess, sys, tempfile
import numpy as np
import cv2
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_intrinsics(path):
    d = json.load(open(path))
    i = d["streams"]["infrared_1"]          # left IR = SLAM camera 1
    K = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]], float)
    dist = np.array(i.get("coeffs", [0, 0, 0, 0, 0]), float).reshape(-1)  # ~0 (rectified)
    return K, dist


def marker_obj_points(size):
    h = size / 2.0
    # order matches cv2.aruco corner order: TL, TR, BR, BL
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def main():
    ap = argparse.ArgumentParser(description="Detect ArUco in .db3 left-IR frames")
    ap.add_argument("--db3", required=True)
    ap.add_argument("--intrinsics", required=True, help="factory_calib_<sn>.json")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "aruco_config.yaml"))
    ap.add_argument("--out", required=True, help="output .pkl of detections")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dict_name = cfg["aruco_dict"]["predefined"]
    size_map = {int(k): float(v) for k, v in cfg["marker_size_map"].items() if k != "default"}
    default_size = float(cfg["marker_size_map"].get("default", 0.16))
    K, dist = load_intrinsics(args.intrinsics)

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    tmp = tempfile.mkdtemp(prefix="rsdet_")
    detections = []
    try:
        subprocess.run(["rs-convert", "-i", args.db3, "-p", os.path.join(tmp, "f")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pngs = sorted(glob.glob(os.path.join(tmp, "f_Infrared 1_*.png")))
        n_tag = 0
        for png in pngs:
            m = re.search(r"_Infrared 1_([0-9.]+)\.png$", os.path.basename(png))
            if not m:
                continue
            t_s = float(m.group(1)) * 1e-3                     # ms -> s (epoch), matches SLAM traj
            img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
            corners, ids, _ = detector.detectMarkers(img)
            tag_dict = {}
            if ids is not None:
                for c, i in zip(corners, ids.flatten()):
                    i = int(i)
                    size = size_map.get(i, default_size)
                    cc = c.reshape(-1, 2).astype(np.float64)   # 4x2
                    ok, rvec, tvec = cv2.solvePnP(marker_obj_points(size), cc, K, dist,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if not ok:
                        continue
                    tag_dict[i] = {"rvec": rvec.squeeze(), "tvec": tvec.squeeze(), "corners": cc}
                    n_tag += 1
            detections.append({"time": t_s, "tag_dict": tag_dict})
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        pickle.dump(detections, open(args.out, "wb"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    seen = {}
    for d in detections:
        for i in d["tag_dict"]:
            seen[i] = seen.get(i, 0) + 1
    print(f"frames: {len(detections)} | total tag detections: {n_tag}")
    print(f"per-id frame counts: {dict(sorted(seen.items()))}")
    oid = cfg.get("origin_tag_id", 13)
    print(f"origin tag {oid}: seen in {seen.get(oid, 0)} frames")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
