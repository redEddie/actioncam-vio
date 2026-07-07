"""Extract a continuous gripper-width signal from the two finger ArUco markers.

rs_slam의 extract_episode.py 너비 계산부와 동일한 원리:
    width_raw = (오른쪽 마커 카메라 x) - (왼쪽 마커 카메라 x) - offset
좌/우는 카메라 x좌표 중앙값으로 자동 판별한다 (마커 id 배치와 무관).
보조로 두 마커 중심 3D 거리(|t_r - t_l|)도 기록 — 방향 불변이지만 마커의
전후/상하 오프셋이 섞인다. offset(완전-닫힘 보정)은 configs/umi_aruco.yaml.

    python -m gopro_vio.gripper_width output/umi_GX014232/tags.pkl \
        -o output/umi_GX014232/gripper
"""
from __future__ import annotations
import argparse
import pathlib
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "umi_aruco.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("detections", help="tags.pkl from gopro_vio.aruco_detect")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("-o", "--out", required=True, help="output directory")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ids = [int(i) for i in cfg["gripper_finger_ids"]]
    offset = float(cfg.get("gripper_width_offset_m", 0.0))
    dets = pickle.load(open(args.detections, "rb"))

    # auto left/right: median camera-frame x per id
    xs = {i: [] for i in ids}
    for d in dets:
        for i in ids:
            if i in d["tag_dict"]:
                xs[i].append(d["tag_dict"][i]["tvec"][0])
    if any(len(v) == 0 for v in xs.values()):
        raise SystemExit(f"finger markers {ids} not both present in detections")
    lid, rid = sorted(ids, key=lambda i: np.median(xs[i]))
    print(f"left = id {lid}, right = id {rid} "
          f"(median cam-x {np.median(xs[lid])*100:.1f} / "
          f"{np.median(xs[rid])*100:.1f} cm)")

    rows = []
    for d in dets:
        td = d["tag_dict"]
        if lid in td and rid in td:
            tl = np.asarray(td[lid]["tvec"], float)
            tr = np.asarray(td[rid]["tvec"], float)
            rows.append([d["time"], tr[0] - tl[0] - offset,
                         float(np.linalg.norm(tr - tl))])
        else:
            rows.append([d["time"], np.nan, np.nan])
    a = np.array(rows)
    t, wx, w3 = a[:, 0], a[:, 1], a[:, 2]
    valid = ~np.isnan(wx)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "gripper_width.csv",
               a, delimiter=",", comments="",
               header="t_s,width_x_m,width_3d_m")

    # noise estimate: high-frequency component via frame-to-frame diff
    wv = wx[valid]
    diff_mm = np.diff(wv) * 1000
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[valid], wv * 100, ".-", ms=2, lw=0.8, label="width_x (cam-x diff)")
    ax.plot(t[valid], w3[valid] * 100, alpha=0.4, lw=0.8, label="|t_r - t_l| 3D")
    ax.set_xlabel("t (s)"); ax.set_ylabel("width (cm)")
    ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"gripper width  ({valid.sum()}/{len(t)} frames, "
                 f"frame-to-frame noise {np.median(np.abs(diff_mm)):.2f} mm)")
    fig.tight_layout()
    fig.savefig(out / "gripper_width.png", dpi=100)

    print(f"frames with both markers: {valid.sum()} / {len(t)} "
          f"({100*valid.sum()/len(t):.1f}%)")
    print(f"width_x range: {np.nanmin(wx)*100:.2f} .. {np.nanmax(wx)*100:.2f} cm")
    print(f"frame-to-frame noise (median |diff|): {np.median(np.abs(diff_mm)):.2f} mm")
    print(f"-> {out/'gripper_width.csv'}\n-> {out/'gripper_width.png'}")


if __name__ == "__main__":
    main()
