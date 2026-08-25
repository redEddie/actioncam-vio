#!/usr/bin/env python3
"""Motorless MuJoCo replay of policy chunks through the deploy IK path.

Loads the deploy URDF, solves IK for GT / predicted yaw-free 6D waypoints exactly like
run_live (target rotation = Rz(current FK yaw) * Ry(pitch) * Rx(roll)), and renders an mp4:
green bar = GT chunk, red bar = prediction; red sphere = FK TCP (fingertip — the gripper
assembly has visual-only meshes, so MuJoCo's URDF import does not draw it), orange trail =
TCP path, blue sphere = commanded waypoint.

Also cross-checks IK-lib FK(tcp_link) against MuJoCo's tcp_link frame (expect ~0 error).

Usage (env with mujoco+imageio, MUJOCO_GL=egl):
  python tools/visualization/mujoco_motorless_replay.py \
      --pred ~/work/eval/preds/var_V6_se3relwp_strongaug_step11000.npz \
      --episode 92 --out replay.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
sys.path.insert(0, str(DEPLOY_DIR))
from ik.ik_solver_v7 import DLSInverseKinematicsV7  # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
URDF = DEPLOY_DIR.parent / "urdf" / "so_arm_with_gopro_final.urdf"


def solve_seq(ik, states6, q):
    out = []
    for st in states6:
        T = np.asarray(ik.forward_kinematics(q))
        yaw = Rotation.from_matrix(T[:3, :3]).as_euler("ZYX")[0]
        target = Rotation.from_euler("ZYX", [yaw, st[4], st[3]]).as_matrix()
        q2 = ik.solve(st[:3], target, q)
        if q2 is None or not ik.last_converged:
            out.append(None)
            continue
        q = q2
        out.append(q.copy())
    return out


def add_marker(scene, pos, rgba, size):
    if scene.ngeom < scene.maxgeom:
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0.0, 0.0]),
                            np.asarray(pos, dtype=np.float64), np.eye(3).flatten(),
                            np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="prediction npz from the offline eval framework")
    parser.add_argument("--episode", type=int, default=None, help="hold-out episode index (default: first)")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--out", default="motorless_replay.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    ik = DLSInverseKinematicsV7(str(URDF))
    spec = mujoco.MjSpec.from_file(str(URDF))
    spec.compiler.fusestatic = False
    model = spec.compile()
    data = mujoco.MjData(model)
    model.geom_rgba[:] = np.tile(np.array([0.72, 0.74, 0.78, 1.0]), (model.ngeom, 1))
    model.vis.headlight.ambient[:] = [0.45] * 3
    model.vis.headlight.diffuse[:] = [0.8] * 3
    jadr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
            for i in range(model.njnt)}
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tcp_link")

    # FK cross-check (IK library vs MuJoCo)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(50):
        q = rng.uniform(-1.2, 1.2, 5)
        for n, v in zip(ARM, q):
            data.qpos[jadr[n]] = v
        mujoco.mj_forward(model, data)
        errs.append(np.linalg.norm(data.xpos[tcp] - np.asarray(ik.forward_kinematics(q))[:3, 3]))
    print(f"FK cross-check (IK lib vs MuJoCo tcp_link): max {max(errs) * 1000:.4f} mm")

    pred = np.load(args.pred)
    episode = args.episode if args.episode is not None else int(pred["ep"][0])
    sel = np.where(pred["ep"] == episode)[0][: args.chunks]
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 140, -22, 0.85
    cam.lookat[:] = [0.2, 0.0, 0.1]
    opt = mujoco.MjvOption()

    frames, ok, fail = [], 0, 0
    q0 = np.deg2rad(np.array([0, -40, 60, 40, 0], dtype=float))
    for i in sel:
        for label, states in (("GT", pred["gt_states"][i]), ("PRED", pred["pred_states"][i])):
            trail = []
            for k, qk in enumerate(solve_seq(ik, states, q0)):
                if qk is None:
                    fail += 1
                    continue
                ok += 1
                for n, v in zip(ARM, qk):
                    data.qpos[jadr[n]] = v
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, cam, opt)
                trail.append(data.xpos[tcp].copy())
                for p in trail[:-1]:
                    add_marker(renderer.scene, p, [1, 0.6, 0.2, 0.5], 0.004)
                add_marker(renderer.scene, trail[-1], [1, 0.15, 0.15, 1.0], 0.009)
                add_marker(renderer.scene, states[k][:3], [0.2, 0.4, 1.0, 0.9], 0.006)
                img = renderer.render().copy()
                img[:16, :, :] = (50, 160, 70) if label == "GT" else (200, 60, 60)
                frames.append(img)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"episode {episode}: {len(frames)} frames (IK ok {ok} / fail {fail}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
