#!/usr/bin/env python3
"""
SLAM-style interactive 3D view: camera trajectory WITH orientation (frustums
showing where the camera looks), optional map-point cloud, in the SLAM map frame
or the ArUco id13 world frame. Writes a self-contained interactive HTML.

    python dataset/plot_trajectory_slam.py \
        --traj data/maps/mapping_..._traj_frames.txt \
        --mappoints data/maps/mapping_..._traj_mappoints.txt \
        --t_world_map calibration/tx_slam_tag_d435i_envmap2.json \
        --out data/maps/slam_view.html

traj: EuRoC "ts tx ty tz qx qy qz qw" = camera pose Twc (OpenCV optical frame:
+Z forward, +X right, +Y down). Frustums open along +Z (view direction).
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import plotly.graph_objects as go


def quat_to_R(q):  # xyzw
    x, y, z, w = q
    n = np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w = x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="EuRoC camera trajectory (Twc)")
    ap.add_argument("--mappoints", default=None)
    ap.add_argument("--t_world_map", default=None, help="tx_slam_tag.json -> id13 world frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cams", type=int, default=40, help="number of camera frustums to draw")
    ap.add_argument("--frustum", type=float, default=0.0,
                    help="frustum depth in m (0 = auto from trajectory extent)")
    a = ap.parse_args()

    T = np.loadtxt(a.traj)
    if T.ndim == 1:
        T = T[None, :]
    pos = T[:, 1:4]
    quat = T[:, 4:8]

    Twm = None
    if a.t_world_map:
        Twm = np.array(json.load(open(a.t_world_map))["T_world_map"], float)
        Rwm, twm = Twm[:3, :3], Twm[:3, 3]
        pos = (Rwm @ pos.T).T + twm

    world = Twm is not None
    extent = (pos.max(0) - pos.min(0)).max()
    fd = a.frustum if a.frustum > 0 else max(extent * 0.05, 0.02)

    fig = go.Figure()

    # map points
    if a.mappoints:
        M = np.loadtxt(a.mappoints)
        if M.ndim == 1:
            M = M[None, :]
        if Twm is not None:
            M = (Rwm @ M.T).T + twm
        lo, hi = np.percentile(M, 1, axis=0), np.percentile(M, 99, axis=0)
        keep = np.all((M >= lo-0.5) & (M <= hi+0.5), axis=1)
        Mv = M[keep] if keep.sum() > 10 else M
        fig.add_trace(go.Scatter3d(
            x=Mv[:,0], y=Mv[:,1], z=Mv[:,2], mode="markers",
            marker=dict(size=1.5, color="lightgray"), name="map points",
            hoverinfo="skip"))

    # camera path, colored by time
    t = np.linspace(0, 1, len(pos))
    fig.add_trace(go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2], mode="lines+markers",
        line=dict(width=3, color="royalblue"),
        marker=dict(size=2, color=t, colorscale="Plasma",
                    colorbar=dict(title="time"), showscale=True),
        name="camera path",
        hovertemplate="frame %{marker.color:.2f}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>"))

    # camera frustums (view direction): apex at cam center, opening along +Z
    base = np.array([[-0.4,-0.4,1],[0.4,-0.4,1],[0.4,0.4,1],[-0.4,0.4,1]]) * fd
    fx, fy, fz = [], [], []
    idx = np.linspace(0, len(pos)-1, min(a.n_cams, len(pos))).astype(int)
    for i in idx:
        R = quat_to_R(quat[i])
        if Twm is not None:
            R = Rwm @ R
        c = pos[i]
        corners = (R @ base.T).T + c           # 4 frustum corners in world
        # 4 apex->corner edges + base loop
        segs = [c, corners[0], c, corners[1], c, corners[2], c, corners[3],
                corners[0], corners[1], corners[2], corners[3], corners[0]]
        for p in segs:
            fx.append(p[0]); fy.append(p[1]); fz.append(p[2])
        fx.append(None); fy.append(None); fz.append(None)   # break between frustums
    fig.add_trace(go.Scatter3d(
        x=fx, y=fy, z=fz, mode="lines",
        line=dict(width=2, color="crimson"), name="camera view (frustum)",
        hoverinfo="skip"))

    # world origin triad (id13) when in world frame
    if world:
        L = max(extent*0.12, 0.03)
        for vec, col in zip(np.eye(3), ["red","green","blue"]):
            fig.add_trace(go.Scatter3d(
                x=[0,vec[0]*L], y=[0,vec[1]*L], z=[0,vec[2]*L], mode="lines",
                line=dict(width=6,color=col), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode="markers",
            marker=dict(size=5,color="black",symbol="diamond"),
            name="id13 origin", hovertemplate="id13 origin<extra></extra>"))

    fig.update_layout(
        title=f"SLAM camera trajectory + view direction ({'id13 world' if world else 'map'} frame): "
              f"{os.path.basename(a.traj)}",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.write_html(a.out, include_plotlyjs=True, full_html=True)
    print(f"poses {len(pos)} | frustums {len(idx)} | -> {a.out}")
    print("drag=rotate, scroll=zoom; red pyramids show where the camera looked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
