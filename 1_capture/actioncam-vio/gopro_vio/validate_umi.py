"""Validate UMI map/demo SLAM outputs before they are promoted.

This is intentionally a thin companion to the upstream-compatible
``gopro_vio.slam`` and ``gopro_vio.world_align`` commands.  It checks tracking
coverage, discontinuities, metric workspace size, exact world-transform
reproduction, and (for mapping runs) ArUco scale-fit quality.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd


def _longest_true_run(mask: np.ndarray) -> int:
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def validate(
    trajectory: pathlib.Path,
    world: pathlib.Path,
    tx_json: pathlib.Path,
    *,
    kind: str,
    slam_log: pathlib.Path | None = None,
    min_tracked_ratio: float = 0.85,
    min_contiguous_ratio: float = 0.70,
    max_world_diagonal_m: float = 1.0,
    max_step_m: float = 0.15,
    max_p99_step_m: float | None = None,
    scale_range: tuple[float, float] = (0.5, 1.5),
    min_tag_samples: int = 20,
    max_median_residual_cm: float = 2.0,
) -> dict:
    raw = pd.read_csv(trajectory)
    out = pd.read_csv(world)
    required_raw = {"timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "state"}
    required_world = {"timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "ok"}
    missing_raw = required_raw - set(raw.columns)
    missing_world = required_world - set(out.columns)
    if missing_raw or missing_world:
        raise ValueError(f"missing columns: raw={sorted(missing_raw)}, world={sorted(missing_world)}")
    if len(raw) != len(out) or len(raw) == 0:
        raise ValueError(f"trajectory row mismatch/empty: raw={len(raw)}, world={len(out)}")

    pos = raw[["x", "y", "z"]].to_numpy(float)
    quat = raw[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
    pw = out[["x", "y", "z"]].to_numpy(float)
    t = raw["timestamp"].to_numpy(float)
    ok = (raw["state"].to_numpy(int) == 2) & (np.linalg.norm(quat, axis=1) > 0.5)
    failures: list[str] = []
    if not np.all(np.isfinite(pos[ok])) or not np.all(np.isfinite(pw[ok])):
        failures.append("tracked poses contain NaN/inf")
    if not np.array_equal(out["ok"].to_numpy(int).astype(bool), ok):
        failures.append("world ok-mask does not match raw tracking state")

    cal = json.loads(tx_json.read_text())
    scale = float(cal["scale_map_per_m"])
    transform = np.asarray(cal["T_world_map_scaled"], float)
    if not np.isfinite(scale) or scale <= 0:
        failures.append(f"invalid scale {scale}")
        reproduction_error = float("inf")
    else:
        predicted = (transform[:3, :3] @ (pos / scale).T).T + transform[:3, 3]
        reproduction_error = float(np.max(np.abs(predicted - pw)))
        if reproduction_error > 1e-8:
            failures.append(f"world transform mismatch ({reproduction_error:.3g} m)")

    tracked = int(ok.sum())
    tracked_ratio = tracked / len(ok)
    longest = _longest_true_run(ok)
    contiguous_ratio = longest / len(ok)
    if tracked_ratio < min_tracked_ratio:
        failures.append(f"tracked ratio {tracked_ratio:.1%} < {min_tracked_ratio:.1%}")
    if contiguous_ratio < min_contiguous_ratio:
        failures.append(
            f"longest contiguous tracked run {contiguous_ratio:.1%} < {min_contiguous_ratio:.1%}")

    if tracked:
        tracked_pw = pw[ok]
        span = np.ptp(tracked_pw, axis=0)
        diagonal = float(np.linalg.norm(span))
    else:
        span = np.full(3, np.nan)
        diagonal = float("inf")
    if diagonal > max_world_diagonal_m:
        failures.append(f"world bounding diagonal {diagonal:.3f} m > {max_world_diagonal_m:.3f} m")

    adjacent = ok[:-1] & ok[1:] & (np.diff(t) > 0) & (np.diff(t) < 0.1)
    steps = np.linalg.norm(np.diff(pw, axis=0)[adjacent], axis=1)
    step_max = float(steps.max(initial=0.0))
    step_p99 = float(np.percentile(steps, 99)) if len(steps) else 0.0
    if step_max > max_step_m:
        failures.append(f"single-frame step {step_max:.3f} m > {max_step_m:.3f} m")
    if max_p99_step_m is not None and step_p99 > max_p99_step_m:
        failures.append(f"p99 frame step {step_p99:.3f} m > {max_p99_step_m:.3f} m")

    fail_to_track = 0
    relocalized = 0
    atlas_loaded = None
    if slam_log and slam_log.exists():
        log_text = slam_log.read_text(errors="replace")
        fail_to_track = len(re.findall(r"Fail to track local map", log_text))
        relocalized = len(re.findall(r"Relocalized!!", log_text))
        atlas_loaded = "Atlas loaded!" in log_text
        if kind == "demo" and not atlas_loaded:
            failures.append("SLAM log does not confirm 'Atlas loaded!'")

    if kind == "map":
        lo, hi = scale_range
        if not lo <= scale <= hi:
            failures.append(f"map scale {scale:.6g} outside [{lo}, {hi}]")
        samples = int(cal.get("n_samples", 0))
        median_residual = float(cal.get("residual_cm", {}).get("median", float("inf")))
        if samples < min_tag_samples:
            failures.append(f"tag inliers {samples} < {min_tag_samples}")
        if median_residual > max_median_residual_cm:
            failures.append(
                f"median tag residual {median_residual:.2f} cm > {max_median_residual_cm:.2f} cm")

    return {
        "passed": not failures,
        "kind": kind,
        "rows": int(len(raw)),
        "tracked_poses": tracked,
        "tracked_ratio": tracked_ratio,
        "longest_contiguous_tracked_ratio": contiguous_ratio,
        "world_span_m": span.tolist(),
        "world_bounding_diagonal_m": diagonal,
        "max_adjacent_step_m": step_max,
        "p99_adjacent_step_m": step_p99,
        "scale_map_per_m": scale,
        "transform_reproduction_max_error_m": reproduction_error,
        "fail_to_track_count": fail_to_track,
        "relocalized_count": relocalized,
        "atlas_loaded": atlas_loaded,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", type=pathlib.Path, required=True)
    ap.add_argument("--world", type=pathlib.Path, required=True)
    ap.add_argument("--tx", type=pathlib.Path, required=True)
    ap.add_argument("--kind", choices=("map", "demo"), required=True)
    ap.add_argument("--slam-log", type=pathlib.Path)
    ap.add_argument("-o", "--out", type=pathlib.Path, required=True)
    ap.add_argument("--min-tracked-ratio", type=float, default=0.85)
    ap.add_argument("--min-contiguous-ratio", type=float, default=0.70)
    ap.add_argument("--max-world-diagonal-m", type=float, default=1.0)
    ap.add_argument("--max-step-m", type=float, default=0.15)
    ap.add_argument("--max-p99-step-m", type=float)
    args = ap.parse_args()
    report = validate(
        args.trajectory, args.world, args.tx, kind=args.kind,
        slam_log=args.slam_log,
        min_tracked_ratio=args.min_tracked_ratio,
        min_contiguous_ratio=args.min_contiguous_ratio,
        max_world_diagonal_m=args.max_world_diagonal_m,
        max_step_m=args.max_step_m,
        max_p99_step_m=args.max_p99_step_m,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
