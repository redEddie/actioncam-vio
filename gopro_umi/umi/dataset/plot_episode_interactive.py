#!/usr/bin/env python3
"""
Interactive 3D viewer for UMI episodes (world / ArUco id13 frame).

Writes a single self-contained HTML you can open in any browser and rotate,
zoom, pan, and toggle individual episodes on/off in the legend. Hover a point
to read its gripper width and frame index.

    # one or many episodes into one interactive page
    python dataset/plot_episode_interactive.py data/episodes/demoA*.csv --out demos.html
    # single episode, points colored by gripper width
    python dataset/plot_episode_interactive.py data/episodes/demoA01.csv --out a01.html

Episode CSV columns: ts_ns tx ty tz qx qy qz qw gripper_width_m  (world frame).
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import plotly.graph_objects as go

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def load(path):
    D = np.loadtxt(path)
    if D.ndim == 1:
        D = D[None, :]
    P = D[:, 1:4]
    w = D[:, 8] * 100 if D.shape[1] >= 9 else np.full(len(P), np.nan)  # cm
    return P, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", help="episode CSV path(s) or glob(s)")
    ap.add_argument("--out", required=True, help="output .html")
    a = ap.parse_args()

    paths = []
    for pat in a.episodes:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("no episode files matched"); return 1
    single = len(paths) == 1

    fig = go.Figure()
    allP = []
    for i, p in enumerate(paths):
        P, w = load(p)
        allP.append(P)
        name = os.path.splitext(os.path.basename(p))[0]
        col = PALETTE[i % len(PALETTE)]
        # path line
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines",
            line=dict(width=4, color=col), name=name, legendgroup=name,
            hoverinfo="skip"))
        # points (colored by width if single episode, else solid per-episode color)
        marker = (dict(size=3, color=w, colorscale="Viridis", showscale=True,
                       colorbar=dict(title="grip width (cm)"))
                  if single and np.isfinite(w).any()
                  else dict(size=3, color=col))
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", marker=marker,
            name=name, legendgroup=name, showlegend=False,
            customdata=np.column_stack([np.arange(len(P)), w]),
            hovertemplate=("%{fullData.name}<br>frame %{customdata[0]}<br>"
                           "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f} m<br>"
                           "width=%{customdata[1]:.1f} cm<extra></extra>")))
        # start / end
        fig.add_trace(go.Scatter3d(
            x=[P[0, 0]], y=[P[0, 1]], z=[P[0, 2]], mode="markers",
            marker=dict(size=6, color="green", symbol="circle"),
            name=f"{name} start", legendgroup=name, showlegend=False,
            hovertemplate=f"{name} START<extra></extra>"))
        fig.add_trace(go.Scatter3d(
            x=[P[-1, 0]], y=[P[-1, 1]], z=[P[-1, 2]], mode="markers",
            marker=dict(size=6, color="red", symbol="x"),
            name=f"{name} end", legendgroup=name, showlegend=False,
            hovertemplate=f"{name} END<extra></extra>"))

    # world origin (ArUco id13): RGB axis triad
    allP = np.vstack(allP)
    L = max((allP.max(0) - allP.min(0)).max() * 0.12, 0.03)
    for vec, col, lbl in zip(np.eye(3), ["red", "green", "blue"], ["+x", "+y", "+z"]):
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]*L], y=[0, vec[1]*L], z=[0, vec[2]*L], mode="lines",
            line=dict(width=6, color=col), name=f"id13 {lbl}",
            hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=5, color="black", symbol="diamond"),
        name="id13 origin", hovertemplate="ArUco id13 origin (0,0,0)<extra></extra>"))

    fig.update_layout(
        title=(f"UMI episode (world / id13 frame): {os.path.basename(paths[0])}"
               if single else f"UMI episodes (world / id13 frame): {len(paths)} demos"),
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                   aspectmode="data"),
        legend=dict(itemsizing="constant"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.write_html(a.out, include_plotlyjs=True, full_html=True)
    print(f"episodes: {len(paths)} | -> {a.out}")
    print("open in a browser; drag to rotate, scroll to zoom, click legend to toggle demos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
