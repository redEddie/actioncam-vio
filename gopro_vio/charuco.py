"""ChArUco fisheye (Kannala-Brandt) intrinsic calibration for GoPro videos.

Auto-detects the ArUco dictionary, board orientation (WxH vs HxW) and
legacy/new ChArUco pattern, then calibrates cv2.fisheye (KB4) intrinsics and
estimates a per-frame board pose timeline (used later for IMU-camera sync).

Usage:
  python -m gopro_vio.charuco data/GX014222.MP4 -o calibration \
      --squares 10 8 --square-size 0.023
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np
from tqdm import tqdm

CAND_DICTS = [
    "DICT_ARUCO_ORIGINAL",
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_100", "DICT_5X5_250", "DICT_6X6_250", "DICT_APRILTAG_36h11",
]


def make_board(squares_xy, square_size, dict_name, legacy):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(tuple(squares_xy), square_size,
                                   square_size * 0.75, d)
    board.setLegacyPattern(legacy)
    return board


def autodetect_board(video, squares_xy, square_size, n_probe=8):
    """Try dictionary x orientation x legacy on a few frames; pick the config
    with the most interpolated ChArUco corners."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(n * 0.1, n * 0.9, n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, img = cap.read()
        if ok:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    cap.release()

    best, best_score = None, -1
    for dict_name in CAND_DICTS:
        # marker detection score is orientation/legacy independent
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        det = cv2.aruco.ArucoDetector(d)
        n_markers = sum(len(det.detectMarkers(f)[0] or ()) for f in frames)
        if n_markers < len(frames) * 4:
            continue
        for layout in (tuple(squares_xy), tuple(squares_xy[::-1])):
            for legacy in (False, True):
                board = make_board(layout, square_size, dict_name, legacy)
                cdet = cv2.aruco.CharucoDetector(board)
                score = 0
                for f in frames:
                    corners, ids, _, _ = cdet.detectBoard(f)
                    if corners is not None and ids is not None:
                        score += len(ids)
                if score > best_score:
                    best_score = score
                    best = (dict_name, layout, legacy)
    if best is None:
        raise RuntimeError("Could not auto-detect board configuration")
    max_possible = (squares_xy[0] - 1) * (squares_xy[1] - 1) * len(frames)
    print(f"[autodetect] dict={best[0]} layout={best[1]} legacy={best[2]} "
          f"corners={best_score}/{max_possible}")
    return best


def detect_all(video, board, stride=1):
    """Detect ChArUco corners on every `stride`-th frame.

    Returns list of dicts: frame idx, time (s), corners (N,2), ids (N,).
    """
    cdet = cv2.aruco.CharucoDetector(board)
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dets = []
    for idx in tqdm(range(n), desc="charuco detect", unit="f"):
        ok, img = cap.read()
        if not ok:
            break
        if idx % stride:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = cdet.detectBoard(gray)
        if corners is not None and ids is not None and len(ids) >= 6:
            dets.append({"frame": idx, "t": idx / fps,
                         "corners": corners.reshape(-1, 2).astype(np.float64),
                         "ids": ids.ravel().astype(int)})
    cap.release()
    size = (int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return dets, size, fps


def select_views(dets, img_size, n_views=60, min_corners=20):
    """Greedy selection of diverse calibration views: spread over time and
    image coverage, preferring frames with many corners."""
    cand = [d for d in dets if len(d["ids"]) >= min_corners]
    if len(cand) <= n_views:
        return cand
    # score coverage on a coarse grid; greedily add the view that covers the
    # most not-yet-covered cells (ties broken by corner count)
    gw, gh = 8, 6
    def cells(d):
        c = d["corners"]
        gx = np.clip((c[:, 0] / img_size[0] * gw).astype(int), 0, gw - 1)
        gy = np.clip((c[:, 1] / img_size[1] * gh).astype(int), 0, gh - 1)
        return set(zip(gx.tolist(), gy.tolist()))
    cand_cells = [cells(d) for d in cand]
    covered, chosen = set(), []
    order = np.argsort([-len(d["ids"]) for d in cand])
    while len(chosen) < n_views:
        best_i, best_gain = None, -1
        for i in order:
            if i in chosen:
                continue
            gain = len(cand_cells[i] - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i is None:
            break
        chosen.append(best_i)
        covered |= cand_cells[best_i]
        if best_gain == 0:
            # coverage saturated: fill remaining slots uniformly in time
            remaining = [i for i in range(len(cand)) if i not in chosen]
            step = max(1, len(remaining) // (n_views - len(chosen) + 1))
            chosen += remaining[::step][: n_views - len(chosen)]
            break
    return [cand[i] for i in sorted(set(chosen))]


def calibrate_fisheye(views, board, img_size):
    obj_all = board.getChessboardCorners()  # (N,3)
    objpoints, imgpoints = [], []
    for d in views:
        # cv2.fisheye.calibrate (OpenCV 5) requires (1, N, C) per view
        objpoints.append(obj_all[d["ids"]].reshape(1, -1, 3).astype(np.float64))
        imgpoints.append(d["corners"].reshape(1, -1, 2).astype(np.float64))

    # OpenCV 5 moved the fisheye CALIB_* flags to the top-level cv2 namespace
    _f = cv2 if hasattr(cv2, "CALIB_RECOMPUTE_EXTRINSIC") else cv2.fisheye
    flags = (_f.CALIB_RECOMPUTE_EXTRINSIC
             | _f.CALIB_FIX_SKEW
             | _f.CALIB_CHECK_COND)
    K = np.eye(3)
    D = np.zeros((4, 1))
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    # CALIB_CHECK_COND aborts on degenerate views; drop offenders and retry
    views = list(views)
    for _ in range(15):
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, imgpoints, img_size, K, D,
                flags=flags, criteria=crit)
            return rms, K, D, views
        except cv2.error as err:
            msg = str(err)
            if "CALIB_CHECK_COND" in msg and "input array" in msg:
                bad = int(msg.split("input array")[1].split()[0])
                for lst in (objpoints, imgpoints, views):
                    lst.pop(bad)
                continue
            # fall back without cond check
            flags &= ~_f.CALIB_CHECK_COND
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_size, K, D, flags=flags, criteria=crit)
    return rms, K, D, views


def board_poses(dets, board, K, D):
    """Per-frame board pose via undistort + solvePnP (camera_T_board)."""
    obj_all = board.getChessboardCorners()
    poses = []
    for d in dets:
        if len(d["ids"]) < 8:
            continue
        obj = obj_all[d["ids"]].astype(np.float64)
        pts = d["corners"].reshape(-1, 1, 2).astype(np.float64)
        und = cv2.fisheye.undistortPoints(pts, K, D).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        poses.append((d["frame"], d["t"], rvec.ravel(), tvec.ravel(),
                      len(d["ids"])))
    return poses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="calibration")
    ap.add_argument("--squares", nargs=2, type=int, default=[10, 8])
    ap.add_argument("--square-size", type=float, default=0.023)
    ap.add_argument("--n-views", type=int, default=60)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = out / "detections_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        dets = list(z["dets"])
        img_size = tuple(int(x) for x in z["img_size"])
        fps = float(z["fps"])
        dict_name, layout, legacy = (str(z["dict_name"]),
                                     tuple(int(x) for x in z["layout"]),
                                     bool(z["legacy"]))
        board = make_board(layout, args.square_size, dict_name, legacy)
        print(f"[detect] loaded {len(dets)} cached detections from {cache}")
    else:
        dict_name, layout, legacy = autodetect_board(
            args.video, args.squares, args.square_size)
        board = make_board(layout, args.square_size, dict_name, legacy)
        dets, img_size, fps = detect_all(args.video, board, args.stride)
        np.savez(cache, dets=np.array(dets, dtype=object),
                 img_size=img_size, fps=fps, dict_name=dict_name,
                 layout=layout, legacy=legacy)
    print(f"[detect] {len(dets)} frames with >=6 corners "
          f"(video {img_size} @ {fps:.3f} fps)")

    views = select_views(dets, img_size, n_views=args.n_views)
    print(f"[calib] using {len(views)} views")
    rms, K, D, used = calibrate_fisheye(views, board, img_size)
    print(f"[calib] RMS reprojection error: {rms:.4f} px  ({len(used)} views)")
    print("K=\n", K, "\nD=", D.ravel())

    poses = board_poses(dets, board, K, D)
    np.savez(out / "board_poses.npz",
             frame=np.array([p[0] for p in poses]),
             t=np.array([p[1] for p in poses]),
             rvec=np.array([p[2] for p in poses]),
             tvec=np.array([p[3] for p in poses]),
             n_corners=np.array([p[4] for p in poses]))

    result = {
        "model": "kannala_brandt (cv2.fisheye, KB4)",
        "image_size": list(img_size),
        "fps": fps,
        "K": K.tolist(),
        "D": D.ravel().tolist(),
        "rms_px": float(rms),
        "n_views": len(used),
        "board": {"dict": dict_name, "layout": list(layout),
                  "square_size_m": args.square_size, "legacy": legacy},
        "video": str(args.video),
    }
    (out / "intrinsics.json").write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out/'intrinsics.json'} and board_poses.npz "
          f"({len(poses)} poses)")


if __name__ == "__main__":
    main()
