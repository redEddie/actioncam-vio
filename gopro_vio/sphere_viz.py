"""Map a fisheye frame onto the unit sphere using the KB4 calibration.

The Kannala-Brandt model is exactly a pixel <-> viewing-direction mapping,
so a frame can be textured onto the unit sphere without any approximation.
Outputs an interactive rerun recording (textured spherical cap + wireframe
globe) and a static PNG preview.

Usage:
  python -m gopro_vio.sphere_viz data/GX014220.MP4 --frame 10224
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def kb4_theta_d(theta, D):
    t2 = theta * theta
    return theta * (1 + D[0]*t2 + D[1]*t2**2 + D[2]*t2**3 + D[3]*t2**4)


def sphere_grid(K, D, img_w, img_h, n_theta=220, n_phi=440, theta_max=None):
    """Build a spherical-cap mesh with per-vertex pixel coordinates.

    Camera frame: +z optical axis, +x right, +y down (OpenCV convention).
    Returns vertices (N,3), uv pixel coords (N,2), valid mask (N,),
    triangles (M,3) using only fully-valid corners.
    """
    if theta_max is None:
        # angle at the farthest image corner
        corners = np.array([[0, 0], [img_w-1, 0], [0, img_h-1],
                            [img_w-1, img_h-1]], np.float64)
        n = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), K,
                                        D.reshape(4, 1)).reshape(-1, 2)
        theta_max = float(np.arctan(np.linalg.norm(n, axis=1)).max()) * 1.02

    th = np.linspace(1e-4, theta_max, n_theta)
    ph = np.linspace(0, 2*np.pi, n_phi)
    TH, PH = np.meshgrid(th, ph, indexing="ij")

    # direction on unit sphere
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)

    # KB4 projection of those directions
    rd = kb4_theta_d(TH, D)
    u = K[0, 0] * rd * np.cos(PH) + K[0, 2]
    v = K[1, 1] * rd * np.sin(PH) + K[1, 2]
    uv = np.stack([u, v], -1).reshape(-1, 2)
    valid = ((uv[:, 0] >= 0) & (uv[:, 0] <= img_w - 1)
             & (uv[:, 1] >= 0) & (uv[:, 1] <= img_h - 1))

    idx = np.arange(n_theta * n_phi).reshape(n_theta, n_phi)
    a = idx[:-1, :-1].ravel(); b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel();  d = idx[:-1, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    tri_ok = valid[tris].all(axis=1)
    return verts, uv, valid, tris[tri_ok], theta_max


def wireframe_globe(n=12, r=1.0):
    """Lat/long circles of the full unit sphere for context."""
    lines = []
    t = np.linspace(0, 2*np.pi, 90)
    for lat in np.linspace(-np.pi/2, np.pi/2, n)[1:-1]:
        lines.append(np.stack([r*np.cos(lat)*np.cos(t), r*np.cos(lat)*np.sin(t),
                               np.full_like(t, r*np.sin(lat))], 1))
    for lon in np.linspace(0, np.pi, n // 2, endpoint=False):
        lines.append(np.stack([r*np.sin(t)*np.cos(lon), r*np.sin(t)*np.sin(lon),
                               r*np.cos(t)], 1))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--calib", default="calibration/intrinsics.json")
    ap.add_argument("-o", "--out", default=None,
                    help="output dir (default output/<video>/sphere)")
    ap.add_argument("--tex-width", type=int, default=1352,
                    help="texture resolution for the rerun mesh")
    args = ap.parse_args()

    calib = json.loads(pathlib.Path(args.calib).read_text())
    K = np.array(calib["K"]); D = np.array(calib["D"])

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {args.frame}")
    H, W = img.shape[:2]

    name = pathlib.Path(args.video).stem
    out = pathlib.Path(args.out) if args.out else pathlib.Path(f"output/{name}/sphere")
    out.mkdir(parents=True, exist_ok=True)

    verts, uv, valid, tris, theta_max = sphere_grid(K, D, W, H)
    fov_cov = (1 - np.cos(theta_max)) / 2
    print(f"[sphere] theta_max={np.degrees(theta_max):.1f}° "
          f"(diagonal FOV {2*np.degrees(theta_max):.1f}°), "
          f"covers {fov_cov*100:.1f}% of the sphere")

    # ---------- rerun: textured mesh ----------
    import rerun as rr
    rrd = out / f"sphere_f{args.frame}.rrd"
    rr.init(f"gopro_sphere/{name}")
    rr.save(str(rrd))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    s = args.tex_width / W
    tex = cv2.cvtColor(cv2.resize(img, (args.tex_width, round(H*s))),
                       cv2.COLOR_BGR2RGB)
    rr.log("sphere/image_cap", rr.Mesh3D(
        vertex_positions=verts.astype(np.float32),
        triangle_indices=tris.astype(np.uint32),
        vertex_texcoords=(uv / [W - 1, H - 1]).astype(np.float32),
        albedo_texture=tex), static=True)
    rr.log("sphere/globe", rr.LineStrips3D(
        wireframe_globe(), colors=[120, 120, 120], radii=0.002), static=True)
    rr.log("sphere/camera_center", rr.Arrows3D(
        origins=[[0, 0, 0]]*3, vectors=np.eye(3)*0.3,
        colors=[[255, 80, 80], [80, 255, 80], [80, 80, 255]]), static=True)
    print(f"[rerun] {rrd}  →  열기: rerun {rrd}")

    # ---------- static PNG preview ----------
    # display frame: x right, y = camera forward (optical axis), z up
    # (camera frame is x right, y DOWN, z forward -> [x, z, -y])
    nt, npph = 130, 260
    v2, uv2, val2, _, _ = sphere_grid(K, D, W, H, nt, npph, theta_max)
    disp = np.stack([v2[:, 0], v2[:, 2], -v2[:, 1]], 1)
    Xg = disp[:, 0].reshape(nt, npph)
    Yg = disp[:, 1].reshape(nt, npph)
    Zg = disp[:, 2].reshape(nt, npph)
    ui = np.clip(uv2[:, 0], 0, W-1).astype(int)
    vi = np.clip(uv2[:, 1], 0, H-1).astype(int)
    rgba = np.zeros((nt, npph, 4))
    rgba[..., :3] = (img[vi, ui][:, ::-1] / 255.0).reshape(nt, npph, 3)
    rgba[..., 3] = val2.reshape(nt, npph)
    face = 0.25 * (rgba[:-1, :-1] + rgba[1:, :-1] + rgba[1:, 1:] + rgba[:-1, 1:])

    fig = plt.figure(figsize=(14, 7))
    # (title, elev, azim, mirror_fix): front view flips the x-axis so the
    # content reads like the photo — i.e. as seen from the sphere center
    views = [("front (as seen from inside)", 8, 90, True),
             ("three-quarter (outside)", 20, 140, False)]
    for i, (vtitle, elev, azim, mirror) in enumerate(views):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        ax.plot_surface(Xg, Yg, Zg, facecolors=face, rstride=1, cstride=1,
                        shade=False, linewidth=0, antialiased=False)
        u_ = np.linspace(0, 2*np.pi, 40); v_ = np.linspace(0, np.pi, 20)
        ax.plot_wireframe(np.outer(np.cos(u_), np.sin(v_)),
                          np.outer(np.sin(u_), np.sin(v_)),
                          np.outer(np.ones_like(u_), np.cos(v_)),
                          color="gray", alpha=0.15, lw=0.5)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        if mirror:
            ax.invert_xaxis()
        ax.set_title(vtitle)
    fig.suptitle(f"{name} frame {args.frame} on the unit sphere — "
                 f"diagonal FOV {2*np.degrees(theta_max):.0f}°, "
                 f"{fov_cov*100:.0f}% of sphere")
    fig.tight_layout()
    png = out / f"sphere_f{args.frame}.png"
    fig.savefig(png, dpi=130)
    print(f"[done] {png}")


if __name__ == "__main__":
    main()
