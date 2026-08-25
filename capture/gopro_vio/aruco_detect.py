"""Detect ArUco markers in a (fisheye) video and estimate metric poses in the
camera frame.

rs_slam의 detect_aruco.py와 같은 출력 스키마이지만, 정류된 스테레오 IR 대신
KB4 어안 원본에서 동작한다: 검출된 코너를 cv2.fisheye.undistortPoints로
정규화 좌표로 편 뒤 K=I로 solvePnP(IPPE_SQUARE) — 왜곡 모델과 PnP를 분리.

타임스탬프는 video t=0 기준 초 (frame_idx / fps) — gopro_vio.slam의
camera_trajectory.csv 타임라인과 동일하므로 별도 정렬이 필요 없다.

    python -m gopro_vio.aruco_detect data/hero7black/umi/GX014229.MP4 \
        --calib cameras/hero7black/calibration/intrinsics.json \
        -o output/umi_GX014229/tags.pkl [--step 2] [--ids 13]
"""
from __future__ import annotations
import argparse
import pathlib
import pickle

import cv2
import numpy as np
import yaml

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "umi_aruco.yaml"


def marker_object_points(size: float) -> np.ndarray:
    h = size / 2.0
    # cv2.aruco corner order: TL, TR, BR, BL
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def load_calib(path: str):
    import json
    d = json.loads(pathlib.Path(path).read_text())
    return np.array(d["K"], float), np.array(d["D"], float).reshape(-1, 1)


def build_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(adict, params)


def pnp_fisheye(corners_px: np.ndarray, size: float, K: np.ndarray, D: np.ndarray):
    """Marker pose in the camera frame from distorted pixel corners (KB4)."""
    und = cv2.fisheye.undistortPoints(
        corners_px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        marker_object_points(size), und, np.eye(3), None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


def detect_video(video: str, K, D, cfg: dict, step: int = 1,
                 only_ids: set | None = None):
    size_map = {int(k): float(v) for k, v in cfg["marker_size_map"].items()
                if str(k) != "default"}
    default_size = float(cfg["marker_size_map"].get("default", 0.10))
    det = build_detector(cfg["aruco_dict"])

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    detections, frame_idx = [], -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        tag_dict = {}
        if ids is not None:
            for c, mid in zip(corners, ids.flatten()):
                mid = int(mid)
                if only_ids and mid not in only_ids:
                    continue
                cc = c.reshape(-1, 2).astype(np.float64)
                pose = pnp_fisheye(cc, size_map.get(mid, default_size), K, D)
                if pose is None:
                    continue
                tag_dict[mid] = {"rvec": pose[0], "tvec": pose[1], "corners": cc}
        detections.append({"time": frame_idx / fps, "frame_idx": frame_idx,
                           "tag_dict": tag_dict})
    cap.release()
    return detections


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--calib", required=True, help="intrinsics.json (KB4)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("-o", "--out", required=True, help="output detections .pkl")
    ap.add_argument("--step", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--ids", type=int, nargs="*", help="only keep these marker ids")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    K, D = load_calib(args.calib)
    dets = detect_video(args.video, K, D, cfg, args.step,
                        set(args.ids) if args.ids else None)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(dets, open(out, "wb"))

    seen = {}
    for d in dets:
        for i, tag in d["tag_dict"].items():
            seen.setdefault(i, []).append(np.linalg.norm(tag["tvec"]))
    print(f"frames processed: {len(dets)} (step {args.step})")
    for i in sorted(seen):
        dist = np.array(seen[i])
        print(f"  id {i}: {len(dist)} frames | cam-dist median "
              f"{np.median(dist)*100:.1f} cm (min {dist.min()*100:.1f}, "
              f"max {dist.max()*100:.1f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
