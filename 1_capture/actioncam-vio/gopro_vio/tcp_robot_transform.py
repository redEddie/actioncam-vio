"""Convert an ArUco-tag-frame camera trajectory into TCP and robot-base poses.

Input ``trajectory_world.csv`` is produced by :mod:`gopro_vio.world_align`.
Despite its historic name, it is a camera-optical-centre trajectory expressed
in the ID-13 ArUco tag frame.  This command preserves it and writes:

* ``trajectory_tcp_tag.csv``: TCP pose in the tag frame;
* ``trajectory_tcp_robot.csv``: the same TCP pose in the robot-base frame.

The fixed transforms are explicit in ``configs/umi_coordinate_frames.yaml``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation


REQUIRED_COLUMNS = {
    "timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "ok"
}


def _load_transform(data: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    entry = data.get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"missing {name} transform")
    R = np.asarray(entry.get("rotation_matrix"), dtype=float)
    t = np.asarray(entry.get("translation_m"), dtype=float)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"{name} must contain a 3x3 rotation and 3-vector translation")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        raise ValueError(f"{name} contains NaN/inf")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(R), 1.0, atol=1e-9):
        raise ValueError(f"{name} rotation is not a proper SO(3) rotation")
    return R, t


def _pose_csv(t: np.ndarray, p: np.ndarray, q: np.ndarray, ok: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": t,
        "x": p[:, 0], "y": p[:, 1], "z": p[:, 2],
        "q_x": q[:, 0], "q_y": q[:, 1], "q_z": q[:, 2], "q_w": q[:, 3],
        "ok": ok.astype(int),
    })


def transform(world_csv: pathlib.Path, config_path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    cfg_text = config_path.read_bytes()
    cfg = yaml.safe_load(cfg_text)
    R_ct, t_ct = _load_transform(cfg, "camera_to_tcp")
    R_rt, t_rt = _load_transform(cfg, "tag_to_robot")

    source = pd.read_csv(world_csv)
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"{world_csv} missing columns: {sorted(missing)}")
    if len(source) == 0:
        raise ValueError(f"{world_csv} is empty")
    t = source["timestamp"].to_numpy(float)
    p_tc = source[["x", "y", "z"]].to_numpy(float)
    q_tc = source[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
    ok = source["ok"].to_numpy(int).astype(bool)
    if not np.all(np.isfinite(p_tc[ok])) or not np.all(np.isfinite(q_tc[ok])):
        raise ValueError("tracked camera poses contain NaN/inf")
    if np.any(np.diff(t) < 0):
        raise ValueError("timestamps are not monotonic")

    R_tc = Rotation.from_quat(q_tc).as_matrix()  # tag <- camera
    p_tt = p_tc + np.einsum("nij,j->ni", R_tc, t_ct)
    R_tt = R_tc @ R_ct                              # tag <- tcp
    q_tt = Rotation.from_matrix(R_tt).as_quat()

    p_rt = (R_rt @ p_tt.T).T + t_rt
    R_rt_tcp = R_rt @ R_tt                          # robot <- tcp
    q_rt = Rotation.from_matrix(R_rt_tcp).as_quat()

    out_dir.mkdir(parents=True, exist_ok=True)
    tag_csv = out_dir / "trajectory_tcp_tag.csv"
    robot_csv = out_dir / "trajectory_tcp_robot.csv"
    _pose_csv(t, p_tt, q_tt, ok).to_csv(tag_csv, index=False)
    _pose_csv(t, p_rt, q_rt, ok).to_csv(robot_csv, index=False)

    offset_error = np.linalg.norm(p_tt - p_tc - np.einsum("nij,j->ni", R_tc, t_ct), axis=1)
    robot_error = np.linalg.norm(p_rt - ((R_rt @ p_tt.T).T + t_rt), axis=1)
    rotation_error = np.linalg.norm(R_rt_tcp - (R_rt @ R_tt), axis=(1, 2))
    expected_offset_m = float(np.linalg.norm(t_ct))
    z_preservation_error = float(np.max(np.abs(p_rt[:, 2] - p_tt[:, 2])))
    failures: list[str] = []
    if np.max(offset_error) > 1e-10:
        failures.append("camera-to-TCP position composition mismatch")
    if np.max(robot_error) > 1e-10:
        failures.append("tag-to-robot position composition mismatch")
    if np.max(rotation_error) > 1e-10:
        failures.append("tag-to-robot orientation composition mismatch")
    if z_preservation_error > 1e-10:
        failures.append("tag-to-robot transform unexpectedly changes Z")
    if not np.all(np.isfinite(p_rt[ok])):
        failures.append("robot TCP poses contain NaN/inf")
    report = {
        "passed": not failures,
        "rows": int(len(source)),
        "tracked_poses": int(ok.sum()),
        "input": str(world_csv),
        "outputs": {"tcp_tag": str(tag_csv), "tcp_robot": str(robot_csv)},
        "config": str(config_path),
        "config_sha256": hashlib.sha256(cfg_text).hexdigest(),
        "frames": {
            "input": "aruco_tag_13_camera_pose",
            "intermediate": "aruco_tag_13_tcp_pose",
            "output": "robot_base_tcp_link_pose",
        },
        "camera_to_tcp_translation_m": t_ct.tolist(),
        "camera_to_tcp_rotation": R_ct.tolist(),
        "tag_to_robot_translation_m": t_rt.tolist(),
        "tag_to_robot_rotation": R_rt.tolist(),
        "camera_to_tcp_offset_norm_m": expected_offset_m,
        "camera_to_tcp_composition_max_error_m": float(np.max(offset_error)),
        "tag_to_robot_composition_max_error_m": float(np.max(robot_error)),
        "orientation_composition_max_error": float(np.max(rotation_error)),
        "tag_robot_z_preservation_max_error_m": z_preservation_error,
        "failures": failures,
    }
    (out_dir / "tcp_robot_validation.json").write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_csv", type=pathlib.Path,
                    help="world/trajectory_world.csv (tag-frame camera pose)")
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path,
                    default=root / "configs/umi_coordinate_frames.yaml")
    args = ap.parse_args()
    report = transform(args.world_csv, args.config, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
