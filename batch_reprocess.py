"""Rebuild the HERO13 UMI map and every episode from original MP4 files.

The individual processing commands intentionally follow upstream
redEddie/actioncam-vio (``gopro_vio.extract``, ``slam``, ``aruco_detect``,
``world_align`` and ``gripper_width``).  This wrapper adds project-specific
camera selection, map/episode provenance, quality gates, resumable staging,
and promotion only after every selected output passes validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

from gopro_vio.validate_umi import validate
from gopro_vio.tcp_robot_transform import transform as transform_tcp_robot


ROOT = pathlib.Path(__file__).resolve().parent
UPSTREAM_REPOSITORY = "https://github.com/redEddie/actioncam-vio"
UPSTREAM_COMMIT = "d77cee58c597f6710d954a0df178d20ff2c9d7ef"
FRAME_CONFIG = ROOT / "configs/umi_coordinate_frames.yaml"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], *, allow_rescued_slam: pathlib.Path | None = None) -> None:
    print("\n▶", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode == 0:
        return
    if allow_rescued_slam and allow_rescued_slam.exists():
        print(
            f"⚠ SLAM exited {proc.returncode}, but upstream rescue trajectory exists; "
            "validation will decide whether it is usable.",
            flush=True,
        )
        return
    raise subprocess.CalledProcessError(proc.returncode, cmd)


def module(*args: str) -> list[str]:
    return [sys.executable, "-m", *map(str, args)]


def write_manifest(
    out: pathlib.Path,
    *,
    kind: str,
    video: pathlib.Path,
    calib: pathlib.Path,
    extr: pathlib.Path,
    mask: pathlib.Path,
    atlas: pathlib.Path,
    report: dict,
    tcp_report: dict | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": kind,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "video": {
            "path": str(video.relative_to(ROOT)),
            "size_bytes": video.stat().st_size,
            "mtime_ns": video.stat().st_mtime_ns,
        },
        "camera": "hero13black",
        "calibration_sha256": sha256(calib),
        "imu_extrinsics_sha256": sha256(extr),
        "gripper_mask_sha256": sha256(mask),
        "map_atlas_sha256": sha256(atlas),
        "validation": report,
    }
    if tcp_report is not None:
        payload["tcp_robot_transform"] = tcp_report
    (out / "processing_manifest.json").write_text(json.dumps(payload, indent=2))


def process_map(
    out: pathlib.Path,
    video: pathlib.Path,
    calib: pathlib.Path,
    extr: pathlib.Path,
    mask: pathlib.Path,
    *,
    features: int,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    run(module("gopro_vio.extract", video, "-o", out))
    trajectory = out / "slam/camera_trajectory.csv"
    run(
        module(
            "gopro_vio.slam", video,
            "--imu", out / "imu.csv", "-o", out / "slam",
            "--calib", calib, "--extr", extr, "--mask", mask,
            "--init_tag_size", "0.10", "--width", "960",
            "--features", str(features), "--fps-div", "2",
        ),
        allow_rescued_slam=trajectory,
    )
    atlas = out / "slam/map_atlas.osa"
    if not atlas.exists():
        raise RuntimeError("mapping SLAM did not produce map_atlas.osa")
    run(module(
        "gopro_vio.aruco_detect", video, "--calib", calib,
        "-o", out / "tags.pkl", "--step", "2", "--ids", "13",
    ))
    run(module(
        "gopro_vio.world_align", out / "tags.pkl", trajectory,
        "-o", out / "world", "--scale-range", "0.5", "1.5",
        "--min-inliers", "20", "--max-median-residual-cm", "2.0",
    ))
    tcp_report = transform_tcp_robot(
        out / "world/trajectory_world.csv", FRAME_CONFIG, out / "world")
    report = validate(
        trajectory, out / "world/trajectory_world.csv",
        out / "world/tx_slam_tag.json", kind="map",
        slam_log=out / "slam/slam_log.txt",
        # This mapping clip has a deliberate initialization sweep.  Initial
        # untracked frames are acceptable; after initialization the map must
        # remain tracked in one continuous run covering at least 70%.
        min_tracked_ratio=0.70, min_contiguous_ratio=0.70,
        max_world_diagonal_m=2.0, max_step_m=0.40,
        max_p99_step_m=0.10,
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("map validation failed: " + "; ".join(report["failures"]))
    write_manifest(
        out, kind="map", video=video, calib=calib, extr=extr, mask=mask,
        atlas=atlas, report=report, tcp_report=tcp_report,
    )
    return report


def process_episode(
    name: str,
    video: pathlib.Path,
    out: pathlib.Path,
    map_out: pathlib.Path,
    calib: pathlib.Path,
    extr: pathlib.Path,
    mask: pathlib.Path,
    existing_out: pathlib.Path | None = None,
    *,
    features: int,
    min_tracked_ratio: float,
    max_world_diagonal_m: float,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    run(module("gopro_vio.extract", video, "-o", out))
    trajectory = out / "slam/camera_trajectory.csv"
    atlas = map_out / "slam/map_atlas.osa"
    run(
        module(
            "gopro_vio.slam", video,
            "--imu", out / "imu.csv", "-o", out / "slam",
            "--calib", calib, "--extr", extr, "--mask", mask,
            "--load-map", atlas, "--init_tag_size", "0.10",
            "--width", "960", "--features", str(features), "--fps-div", "2",
        ),
        allow_rescued_slam=trajectory,
    )
    tx = map_out / "world/tx_slam_tag.json"
    run(module(
        "gopro_vio.world_align", "--apply", tx, trajectory,
        "-o", out / "world",
    ))
    tcp_report = transform_tcp_robot(
        out / "world/trajectory_world.csv", FRAME_CONFIG, out / "world")
    report = validate(
        trajectory, out / "world/trajectory_world.csv", tx, kind="demo",
        slam_log=out / "slam/slam_log.txt",
        min_tracked_ratio=min_tracked_ratio, min_contiguous_ratio=0.70,
        max_world_diagonal_m=max_world_diagonal_m, max_step_m=0.15,
        max_p99_step_m=0.05,
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"{name} validation failed: " + "; ".join(report["failures"]))

    # Finger-marker detections are computed directly from the original MP4
    # and do not depend on SLAM/map coordinates.  Preserve the already valid
    # HERO13 detections when present; regenerate only if they are missing.
    old_tags = existing_out / "tags.pkl" if existing_out else None
    old_gripper = existing_out / "gripper" if existing_out else None
    if old_tags and old_tags.exists() and old_gripper and old_gripper.exists():
        shutil.copy2(old_tags, out / "tags.pkl")
        shutil.copytree(old_gripper, out / "gripper")
        print(f"✓ preserved SLAM-independent gripper detections from {existing_out}")
    else:
        run(module(
            "gopro_vio.aruco_detect", video, "--calib", calib,
            "-o", out / "tags.pkl", "--step", "1", "--ids", "0", "1",
        ))
        run(module("gopro_vio.gripper_width", out / "tags.pkl", "-o", out / "gripper"))
    write_manifest(
        out, kind="demo", video=video, calib=calib, extr=extr, mask=mask,
        atlas=atlas, report=report, tcp_report=tcp_report,
    )
    return report


def promote(staging: pathlib.Path, result_root: pathlib.Path, names: list[str]) -> None:
    targets = ["map_result", *names]
    for name in targets:
        source = staging / name
        destination = result_root / name
        if not source.exists():
            raise RuntimeError(f"cannot promote missing staged output: {source}")
        if destination.exists():
            shutil.rmtree(destination)
        source.replace(destination)
        print(f"✓ promoted {name} -> {destination}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map-video", type=pathlib.Path, default=ROOT / "Episode/map.MP4")
    ap.add_argument("--result-root", type=pathlib.Path, default=ROOT / "Episode_result")
    ap.add_argument("--staging", type=pathlib.Path,
                    default=ROOT / "Episode_result/.reprocess_work")
    ap.add_argument("--episodes", nargs="*", type=int,
                    help="episode numbers to process (default: all in mapping log)")
    ap.add_argument("--features", type=int, default=2500,
                    help="ORB features; upstream default workflow uses 2500")
    ap.add_argument("--min-tracked-ratio", type=float, default=0.85)
    ap.add_argument("--max-world-diagonal-m", type=float, default=1.0)
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="independent ORB-SLAM3 attempts per map/episode")
    ap.add_argument("--reuse-staged-map", action="store_true")
    ap.add_argument("--force-episodes", action="store_true",
                    help="rebuild episodes even when a validated staged result exists")
    ap.add_argument("--promote", action="store_true",
                    help="replace existing results only after all selected outputs pass")
    args = ap.parse_args()

    map_video = args.map_video.resolve()
    result_root = args.result_root.resolve()
    staging = args.staging.resolve()
    calib = ROOT / "cameras/hero13black/calibration/intrinsics.json"
    extr = ROOT / "cameras/hero13black/calibration/imu_extrinsics.json"
    mask = ROOT / "cameras/hero13black/calibration/gripper_mask.png"
    mapping_file = result_root / "episode_mapping_log.json"
    mapping = json.loads(mapping_file.read_text())
    selected = {
        f"episode_{n}" for n in args.episodes
    } if args.episodes else set(mapping)
    unknown = selected - set(mapping)
    if unknown:
        raise SystemExit(f"unknown episodes: {sorted(unknown)}")
    names = sorted(selected, key=lambda x: int(x.split("_")[1]))

    required = [map_video, calib, extr, mask, FRAME_CONFIG, mapping_file]
    required.extend(ROOT / "Episode" / mapping[name] for name in names)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))

    staging.mkdir(parents=True, exist_ok=True)
    status_path = staging / "reprocess_status.json"
    status = {
        "upstream_commit": UPSTREAM_COMMIT,
        "map": "pending",
        "episodes": {name: "pending" for name in names},
        "attempts": {"map": [], "episodes": {name: [] for name in names}},
    }
    status_path.write_text(json.dumps(status, indent=2))

    map_out = staging / "map_result"
    if args.reuse_staged_map and (map_out / "processing_manifest.json").exists():
        print(f"✓ reusing validated staged map: {map_out}")
    else:
        if map_out.exists():
            shutil.rmtree(map_out)
        attempts_root = staging / ".attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            candidate = attempts_root / f"map_{attempt}"
            if candidate.exists():
                shutil.rmtree(candidate)
            print(f"\n===== map attempt {attempt}/{args.max_attempts} =====", flush=True)
            try:
                report = process_map(
                    candidate, map_video, calib, extr, mask,
                    features=args.features,
                )
                status["attempts"]["map"].append({"attempt": attempt, "report": report})
                candidate.replace(map_out)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                report_path = candidate / "validation.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else None
                status["attempts"]["map"].append(
                    {"attempt": attempt, "error": str(exc), "report": report})
                status_path.write_text(json.dumps(status, indent=2))
                print(f"✗ map attempt {attempt} rejected: {exc}", flush=True)
                if candidate.exists():
                    shutil.rmtree(candidate)
        if last_error is not None:
            raise RuntimeError(
                f"map failed all {args.max_attempts} attempts") from last_error
    status["map"] = "passed"
    status_path.write_text(json.dumps(status, indent=2))

    atlas_hash = sha256(map_out / "slam/map_atlas.osa")
    failed_names: list[str] = []
    for index, name in enumerate(names, 1):
        print(f"\n===== [{index}/{len(names)}] {name}: {mapping[name]} =====", flush=True)
        out = staging / name
        manifest_path = out / "processing_manifest.json"
        if not args.force_episodes and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("map_atlas_sha256") == atlas_hash and
                        manifest.get("validation", {}).get("passed") is True):
                    status["episodes"][name] = "passed (reused staged output)"
                    status_path.write_text(json.dumps(status, indent=2))
                    print(f"✓ reusing validated staged output: {out}", flush=True)
                    continue
            except (OSError, ValueError, TypeError):
                pass
        if out.exists():
            shutil.rmtree(out)
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            candidate = staging / ".attempts" / f"{name}_{attempt}"
            if candidate.exists():
                shutil.rmtree(candidate)
            print(f"\n--- {name} attempt {attempt}/{args.max_attempts} ---", flush=True)
            try:
                report = process_episode(
                    name, ROOT / "Episode" / mapping[name], candidate, map_out,
                    calib, extr, mask, existing_out=result_root / name,
                    features=args.features,
                    min_tracked_ratio=args.min_tracked_ratio,
                    max_world_diagonal_m=args.max_world_diagonal_m,
                )
                status["attempts"]["episodes"][name].append(
                    {"attempt": attempt, "report": report})
                candidate.replace(out)
                status["episodes"][name] = "passed"
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                report_path = candidate / "validation.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else None
                status["attempts"]["episodes"][name].append(
                    {"attempt": attempt, "error": str(exc), "report": report})
                status_path.write_text(json.dumps(status, indent=2))
                print(f"✗ {name} attempt {attempt} rejected: {exc}", flush=True)
                if candidate.exists():
                    shutil.rmtree(candidate)
        if last_error is not None:
            status["episodes"][name] = f"failed after {args.max_attempts} attempts: {last_error}"
            status_path.write_text(json.dumps(status, indent=2))
            failed_names.append(name)
            print(f"✗ {name} failed all {args.max_attempts} attempts; continuing", flush=True)
        status_path.write_text(json.dumps(status, indent=2))

    if failed_names:
        raise RuntimeError(
            "selected episodes failed validation: " + ", ".join(failed_names))

    if args.promote:
        promote(staging, result_root, names)
        status["promoted"] = True
        (result_root / "reprocess_status.json").write_text(json.dumps(status, indent=2))
    else:
        print(f"\nAll staged outputs passed. Re-run with --reuse-staged-map --promote "
              f"to replace existing results. Status: {status_path}")


if __name__ == "__main__":
    main()
