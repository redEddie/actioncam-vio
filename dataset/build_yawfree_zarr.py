#!/usr/bin/env python3
"""Derive the canonical yaw-free 6D Zarr from the protected source Zarr.

The rotation convention is recovered from the audited historical builder with
SHA-256 ``0291fe82cdc2ed9337e5dd403f5f6e69c26d2fc22fda0b809fee6bbc20af8ff0``:
rotation vector -> rotation matrix -> intrinsic Euler ZYX [yaw, pitch, roll],
then store only [roll, pitch].  RGB, position, gripper, and episode-boundary
arrays are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from numcodecs import get_codec
import numpy as np
from scipy.spatial.transform import Rotation


TRUSTED_SOURCE_SHA256 = (
    "0291fe82cdc2ed9337e5dd403f5f6e69c26d2fc22fda0b809fee6bbc20af8ff0"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _metadata(path: Path) -> dict[str, Any]:
    return json.loads((path / ".zarray").read_text())


def _read_v2_array(path: Path) -> np.ndarray:
    meta = _metadata(path)
    shape = tuple(int(value) for value in meta["shape"])
    chunks = tuple(int(value) for value in meta["chunks"])
    if len(shape) not in {1, 2} or chunks[1:] != shape[1:]:
        raise ValueError(f"unsupported source array chunking: {path}: {chunks}")
    dtype = np.dtype(meta["dtype"])
    codec = get_codec(meta["compressor"]) if meta.get("compressor") else None
    output = np.empty(shape, dtype=dtype)
    chunk_count = (shape[0] + chunks[0] - 1) // chunks[0]
    for chunk_index in range(chunk_count):
        name = str(chunk_index) if len(shape) == 1 else f"{chunk_index}.0"
        payload = (path / name).read_bytes()
        decoded = codec.decode(payload) if codec is not None else payload
        chunk = np.frombuffer(decoded, dtype=dtype).reshape(chunks)
        start = chunk_index * chunks[0]
        stop = min(start + chunks[0], shape[0])
        output[start:stop] = chunk[: stop - start]
    return output


def _write_v2_array(path: Path, values: np.ndarray, chunks: tuple[int, ...]) -> None:
    values = np.ascontiguousarray(values)
    path.mkdir(parents=True)
    compressor_config = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1}
    codec = get_codec(compressor_config)
    meta = {
        "chunks": list(chunks),
        "compressor": compressor_config,
        "dtype": values.dtype.str,
        "fill_value": 0.0,
        "filters": None,
        "order": "C",
        "shape": list(values.shape),
        "zarr_format": 2,
    }
    (path / ".zarray").write_text(json.dumps(meta, indent=4, sort_keys=True) + "\n")
    (path / ".zattrs").write_text("{}\n")
    for chunk_index, start in enumerate(range(0, len(values), chunks[0])):
        chunk = np.zeros(chunks, dtype=values.dtype)
        source = values[start : start + chunks[0]]
        chunk[: len(source)] = source
        chunk = np.ascontiguousarray(chunk)
        name = str(chunk_index) if values.ndim == 1 else f"{chunk_index}.0"
        (path / name).write_bytes(codec.encode(chunk))


def yawfree_roll_pitch(rotvec: np.ndarray) -> np.ndarray:
    value = np.asarray(rotvec, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError("rotation vectors must be finite [N,3]")
    euler_zyx = Rotation.from_matrix(Rotation.from_rotvec(value).as_matrix()).as_euler("ZYX")
    result = np.column_stack((euler_zyx[:, 2], euler_zyx[:, 1])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Euler conversion produced NaN/Inf")
    return result


def validate_source(source: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    attrs = json.loads((source / ".zattrs").read_text())
    expected_attrs = {
        "pose_source": "trajectory_tcp_robot.csv",
        "pose_frame": "robot_base_tcp_link_pose",
        "coordinate_transforms_applied_by_builder": "none",
    }
    for key, expected in expected_attrs.items():
        if attrs.get(key) != expected:
            raise ValueError(f"source attr {key!r} must be {expected!r}")
    image_meta = _metadata(source / "data" / "camera0_rgb")
    frames = int(image_meta["shape"][0])
    if tuple(image_meta["shape"][1:]) != (224, 224, 3):
        raise ValueError("source RGB must be [N,224,224,3]")
    positions = _read_v2_array(source / "data" / "robot0_eef_pos").astype(np.float32)
    rotvec = _read_v2_array(source / "data" / "robot0_eef_rot_axis_angle")
    gripper = _read_v2_array(source / "data" / "robot0_gripper_width").astype(np.float32)
    ends = _read_v2_array(source / "meta" / "episode_ends").astype(np.int64)
    if positions.shape != (frames, 3) or rotvec.shape != (frames, 3):
        raise ValueError("source pose arrays do not match RGB frame count")
    if gripper.shape != (frames, 1) or ends.ndim != 1 or int(ends[-1]) != frames:
        raise ValueError("source gripper or episode boundaries are invalid")
    if np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("source episodes must contain positive frame counts")
    if not np.isfinite(positions).all() or not np.isfinite(gripper).all():
        raise ValueError("source contains NaN/Inf")
    if float(gripper.min()) < -1e-6 or float(gripper.max()) > 1.0 + 1e-6:
        raise ValueError("source gripper is outside [0,1]")
    roll_pitch = yawfree_roll_pitch(rotvec)
    return {
        "frames": frames,
        "episodes": len(ends),
        "positions": positions,
        "gripper": gripper,
        "ends": ends,
        "roll_pitch": roll_pitch,
    }


def _existing_output_summary(output: Path) -> dict[str, Any] | None:
    if not output.exists():
        return None
    attrs = json.loads((output / ".zattrs").read_text())
    images = _metadata(output / "data" / "camera0_rgb")
    ends = _read_v2_array(output / "meta" / "episode_ends")
    return {
        "frames": int(images["shape"][0]),
        "episodes": len(ends),
        "source_zarr": attrs.get("source_zarr"),
        "yaw_removed": attrs.get("yaw_removed"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "artifacts/zarr/replay_buffer.zarr")
    parser.add_argument(
        "--output", "-o", type=Path, default=root / "artifacts/zarr/replay_buffer_yawfree.zarr"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or output.suffix != ".zarr" or output == Path(output.anchor):
        raise ValueError("unsafe yaw-free output path")
    validated = validate_source(source)
    existing = _existing_output_summary(output)
    report = {
        "project_root": str(project_root()),
        "source": str(source),
        "output": str(output),
        "source_frames": validated["frames"],
        "source_episodes": validated["episodes"],
        "trusted_source_sha256": TRUSTED_SOURCE_SHA256,
        "rotation_conversion": "rotvec -> matrix -> intrinsic Euler ZYX -> [roll,pitch]",
        "existing_output": existing,
        "existing_output_matches_source_scope": existing is None
        or (
            existing["frames"] == validated["frames"]
            and existing["episodes"] == validated["episodes"]
        ),
    }
    if args.dry_run:
        return report
    if existing is not None and not report["existing_output_matches_source_scope"]:
        raise ValueError(
            "refusing to replace a yaw-free artifact with a different frame/episode scope; "
            "use a separate --output or recover the matching source Zarr"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite explicitly")
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        (temporary / "data").mkdir(parents=True)
        (temporary / "meta").mkdir()
        (temporary / ".zgroup").write_text('{"zarr_format": 2}\n')
        for group in ("data", "meta"):
            source_group = source / group
            target_group = temporary / group
            for metadata_name in (".zgroup", ".zattrs"):
                metadata = source_group / metadata_name
                if metadata.is_file():
                    shutil.copy2(metadata, target_group / metadata_name)
        for relative in (
            "data/camera0_rgb",
            "data/robot0_eef_pos",
            "data/robot0_gripper_width",
            "meta/episode_ends",
        ):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / relative, destination)
        _write_v2_array(
            temporary / "data" / "robot0_eef_roll_pitch",
            validated["roll_pitch"],
            (100, 2),
        )
        attrs = {
            "pose_source": "trajectory_tcp_robot.csv",
            "pose_frame": "robot_base_tcp_roll_pitch_pose",
            "task_dof": "xyz_roll_pitch",
            "coordinate_transforms_applied_by_builder": "none",
            "rotation_conversion": (
                "rotvec -> rotation_matrix -> Euler intrinsic ZYX [yaw,pitch,roll]; "
                "stored [roll,pitch] radians"
            ),
            "yaw_removed": True,
            "source_zarr": "artifacts/zarr/replay_buffer.zarr",
            "trusted_transform_sha256": TRUSTED_SOURCE_SHA256,
        }
        (temporary / ".zattrs").write_text(json.dumps(attrs, indent=2, sort_keys=True) + "\n")
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise ValueError(f"refusing to replace unsafe output: {output}")
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = build(parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
