"""Anchor a SLAM trajectory to the world frame defined by the origin ArUco tag
(id 13), following UMI's calibrate_slam_tag + transform_to_world.

원점 태그가 보이는 모든 프레임에서
    tx_map_tag_i = T_map_cam(t_i) @ T_cam_tag_i
를 모아 강건 평균 → T_world_map = inv(tx_map_tag).  월드 원점 = 태그 중심,
월드 z축 = 태그 법선.

mono-inertial 스케일 오차 보정이 추가돼 있다 (rs_slam은 스테레오라 불필요했음):
태그는 크기를 알아 PnP가 미터 단위 p_i를 주므로, 태그 고정점 조건
    c_i + s·R_i·p_i = X   (c_i, R_i: SLAM 포즈 / s: 맵 단위 per meter)
를 [X, s]에 대한 선형 최소제곱으로 풀어 s를 추정하고, 맵 좌표를 1/s로
리스케일한 뒤 T_world_map을 계산한다.

    # 원점 태그가 보이는 영상에서 T_world_map 추정 + 궤적 변환:
    python -m gopro_vio.world_align output/umi_GX014229/tags.pkl \
        output/umi_GX014229/slam/camera_trajectory.csv -o output/umi_GX014229/world

    # 태그가 안 보이는 데모(같은 맵에 --load-map으로 localize한 경우):
    # 맵 영상에서 얻은 변환을 그대로 적용
    python -m gopro_vio.world_align --apply output/umi_GX014229/world/tx_slam_tag.json \
        output/umi_GX014230/slam/camera_trajectory.csv -o output/umi_GX014230/world
"""
from __future__ import annotations
import argparse
import json
import pathlib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def load_trajectory(path: str):
    """Return t (s), pos (N,3), quat xyzw (N,4), ok-mask; supports our CSV and TUM."""
    p = pathlib.Path(path)
    if p.suffix == ".csv":
        import pandas as pd
        d = pd.read_csv(p)
        t = d["timestamp"].to_numpy(float)
        pos = d[["x", "y", "z"]].to_numpy(float)
        quat = d[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
        ok = (d["state"].to_numpy(int) == 2) & (np.linalg.norm(quat, axis=1) > 0.5)
    else:  # TUM: t x y z qx qy qz qw
        a = np.loadtxt(p)
        t, pos, quat = a[:, 0], a[:, 1:4], a[:, 4:8]
        ok = np.linalg.norm(quat, axis=1) > 0.5
    return t, pos, quat, ok


def fit_scale(cs: np.ndarray, Rps: np.ndarray):
    """Solve c_i + s*(R_i p_i) = X for [X, s]; returns X, s, residuals (m)."""
    n = len(cs)
    A = np.zeros((3 * n, 4))
    A[:, :3] = np.tile(np.eye(3), (n, 1))
    A[:, 3] = -Rps.reshape(-1)
    x, *_ = np.linalg.lstsq(A, cs.reshape(-1), rcond=None)
    res = np.linalg.norm(cs + x[3] * Rps - x[:3], axis=1)
    return x[:3], x[3], res


def robust_fit_scale(cs, Rps, rounds=2):
    keep = np.ones(len(cs), bool)
    for _ in range(rounds):
        X, s, res = fit_scale(cs[keep], Rps[keep])
        mad = np.median(np.abs(res - np.median(res))) * 1.4826
        thr = max(3 * mad + np.median(res), 0.01)
        new = np.ones(len(cs), bool)
        new[np.where(keep)[0][res > thr]] = False
        if new.sum() == keep.sum() or new.sum() < 3:
            break
        keep = new
    X, s, _ = fit_scale(cs[keep], Rps[keep])
    res_all = np.linalg.norm(cs + s * Rps - X, axis=1)
    return X, s, res_all, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("detections", nargs="?",
                    help="tags.pkl from gopro_vio.aruco_detect")
    ap.add_argument("trajectory", help="camera_trajectory.csv (or .tum) in map frame")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("--apply", metavar="TX_JSON",
                    help="tx_slam_tag.json from the mapping video: apply its "
                         "T_world_map + scale instead of estimating (for demos "
                         "localized into the same map without the origin tag)")
    ap.add_argument("--tag-id", type=int, default=13)
    ap.add_argument("--min-dist", type=float, default=0.15)
    ap.add_argument("--max-dist", type=float, default=2.5)
    ap.add_argument("--time-tol", type=float, default=0.05)
    ap.add_argument("--no-scale", action="store_true",
                    help="skip scale estimation (s=1), UMI/rs_slam behaviour")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_traj, pos, quat, ok = load_trajectory(args.trajectory)
    t_ok, pos_ok, quat_ok = t_traj[ok], pos[ok], quat[ok]
    if len(t_ok) < 3:
        raise SystemExit("trajectory has <3 tracked poses")

    if args.apply:
        cal = json.loads(pathlib.Path(args.apply).read_text())
        s = float(cal["scale_map_per_m"])
        T_world_map = np.array(cal["T_world_map_scaled"], float)
        transform_and_save(out, t_traj, pos, quat, ok, T_world_map, s)
        return
    if not args.detections:
        raise SystemExit("either detections.pkl or --apply is required")
    dets = pickle.load(open(args.detections, "rb"))

    # gather per-frame samples where the origin tag is seen and SLAM is tracking
    cs, Rms, Rps, Rtags = [], [], [], []
    for d in dets:
        tag = d["tag_dict"].get(args.tag_id)
        if tag is None:
            continue
        p_cam = np.asarray(tag["tvec"], float)
        dist = np.linalg.norm(p_cam)
        if not (args.min_dist < dist < args.max_dist):
            continue
        j = int(np.argmin(np.abs(t_ok - d["time"])))
        if abs(t_ok[j] - d["time"]) > args.time_tol:
            continue
        R_mc = Rotation.from_quat(quat_ok[j]).as_matrix()   # map <- cam
        cs.append(pos_ok[j])
        Rms.append(R_mc)
        Rps.append(R_mc @ p_cam)
        Rtags.append(R_mc @ cv2.Rodrigues(np.asarray(tag["rvec"], float))[0])
    if len(cs) < 3:
        raise SystemExit(f"only {len(cs)} usable tag-{args.tag_id} samples "
                         "(0.15-2.5 m, while tracking). Record the tag more.")
    cs, Rps = np.array(cs), np.array(Rps)

    if args.no_scale:
        s = 1.0
        X = cs + Rps  # per-sample tag positions
        keep = np.ones(len(cs), bool)
        tag_map = np.median(X, axis=0)
        res = np.linalg.norm(X - tag_map, axis=1)
    else:
        X, s, res, keep = robust_fit_scale(cs, Rps)
        tag_map = X / s  # in scale-corrected (metric) map coords

    # rotation: mean of R_map_tag over inliers
    R_mean = Rotation.from_matrix(np.array(Rtags)[keep]).mean().as_matrix()

    tx_map_tag = np.eye(4)
    tx_map_tag[:3, :3] = R_mean
    tx_map_tag[:3, 3] = tag_map
    T_world_map = np.linalg.inv(tx_map_tag)

    marker_world = (T_world_map @ np.array([*tag_map, 1.0]))[:3]
    info = {"tag_id": args.tag_id, "n_samples": int(keep.sum()),
            "n_rejected": int((~keep).sum()),
            "scale_map_per_m": float(s),
            "tx_map_tag_scaled": tx_map_tag.tolist(),
            "T_world_map_scaled": T_world_map.tolist(),
            "residual_cm": {"median": float(np.median(res[keep]) * 100),
                            "max": float(res[keep].max() * 100)}}
    json_out = out / "tx_slam_tag.json"
    json_out.write_text(json.dumps(info, indent=2))

    print(f"samples: {keep.sum()} used / {len(keep)} gathered "
          f"({(~keep).sum()} rejected)")
    print(f"scale (map units per meter): {s:.4f}  "
          f"({(s - 1) * 100:+.1f}% vs metric)")
    print(f"tag-fix residual: median {np.median(res[keep])*100:.2f} cm, "
          f"max {res[keep].max()*100:.2f} cm")
    print(f"marker->world check (~0):"
          f" {np.round(marker_world * 100, 2).tolist()} cm")
    print(f"-> {json_out}")
    transform_and_save(out, t_traj, pos, quat, ok, T_world_map, s)


def transform_and_save(out, t_traj, pos, quat, ok, T_world_map, s):
    """Rescale the map-frame trajectory by 1/s, move it to the world frame,
    write trajectory_world.csv + a 3-panel plot."""
    pos_m = pos / s
    Pw = (T_world_map[:3, :3] @ pos_m.T).T + T_world_map[:3, 3]
    Rw = Rotation.from_matrix(
        T_world_map[:3, :3] @ Rotation.from_quat(
            np.where(np.linalg.norm(quat, axis=1, keepdims=True) > 0.5, quat,
                     [0, 0, 0, 1])).as_matrix())
    qw = Rw.as_quat()

    import pandas as pd
    dfw = pd.DataFrame({"timestamp": t_traj,
                        "x": Pw[:, 0], "y": Pw[:, 1], "z": Pw[:, 2],
                        "q_x": qw[:, 0], "q_y": qw[:, 1], "q_z": qw[:, 2],
                        "q_w": qw[:, 3], "ok": ok.astype(int)})
    traj_out = out / "trajectory_world.csv"
    dfw.to_csv(traj_out, index=False)

    # plot: map frame / world top-down / world 3D.  Split the polyline where
    # tracking gaps exceed 0.2 s so lost stretches aren't drawn as straight jumps.
    t_okv, Pm_ok, Pw_ok = t_traj[ok], pos[ok], Pw[ok]
    cuts = np.where(np.diff(t_okv) > 0.2)[0] + 1
    chunks = np.split(np.arange(len(t_okv)), cuts)
    fig = plt.figure(figsize=(14, 4.2))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")
    for k, c in enumerate(chunks):
        lbl = "camera" if k == 0 else None
        ax1.plot(Pm_ok[c, 0], Pm_ok[c, 2], lw=1, color="C0")
        ax2.plot(Pw_ok[c, 0] * 100, Pw_ok[c, 1] * 100, lw=1, color="C0", label=lbl)
        ax3.plot(Pw_ok[c, 0] * 100, Pw_ok[c, 1] * 100, Pw_ok[c, 2] * 100,
                 lw=1, color="C0")
    ax1.set_title("MAP frame (X-Z, m)"); ax1.axis("equal"); ax1.grid(alpha=.3)
    ax2.scatter([0], [0], c="r", marker="*", s=120, label="tag origin")
    ax2.set_title(f"WORLD top-down (X-Y on tag, cm)  [{len(chunks)} segment(s)]")
    ax2.axis("equal"); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    ax3.scatter([0], [0], [0], c="r", marker="*", s=120)
    ax3.set_title("WORLD 3D (cm)")
    fig.tight_layout()
    plot_out = out / "world_align.png"
    fig.savefig(plot_out, dpi=100)

    zw = Pw_ok[:, 2]
    print(f"tracked poses: {ok.sum()} / {len(ok)}")
    print(f"camera height above tag plane: {zw.min()*100:.1f} .. "
          f"{zw.max()*100:.1f} cm (positive = tag-normal side)")
    print(f"-> {traj_out}\n-> {plot_out}")


if __name__ == "__main__":
    main()
