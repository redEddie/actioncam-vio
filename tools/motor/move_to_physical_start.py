#!/usr/bin/env python3
"""Move only five arm joints to the canonical recorded physical start.

The target is loaded from ``deploy/config/physical_start.json`` through the
validated runtime configuration. No IK result, Cartesian target, model action,
or gripper goal participates in target generation. The execute path preserves
the existing torque state and does not configure PID or other registers.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "deploy"
LEROBOT_SRC = PROJECT_ROOT / "etc" / "third_party" / "lerobot" / "src"
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ARRIVAL_TOLERANCE_DEG = 1.0
MAX_READ_DISCONTINUITY_DEG = 30.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(report), indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def _finite_arm(values: Mapping[str, float], name: str) -> np.ndarray:
    result = np.asarray([values[joint] for joint in ARM_JOINTS], dtype=np.float64)
    if result.shape != (5,) or not np.isfinite(result).all():
        raise RuntimeError(f"{name} did not return five finite arm positions")
    return result


def read_snapshot(bus: Any, mapping: Any, clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    started = float(clock())
    normalized = {name: float(value) for name, value in bus.sync_read("Present_Position", normalize=True).items()}
    raw_ticks = {name: float(value) for name, value in bus.sync_read("Present_Position", normalize=False).items()}
    finished = float(clock())
    arm_raw_deg = _finite_arm(normalized, "normalized Present_Position")
    arm_ticks = _finite_arm(raw_ticks, "raw Present_Position")
    gripper_raw = float(raw_ticks["gripper"])
    if not np.isfinite(gripper_raw):
        raise RuntimeError("gripper Present_Position is nonfinite")
    return {
        "timestamp_monotonic": finished,
        "read_started_monotonic": started,
        "read_duration_s": finished - started,
        "arm_raw_deg": arm_raw_deg,
        "arm_urdf_deg": mapping.raw_degrees_to_urdf_degrees(arm_raw_deg),
        "arm_raw_ticks": arm_ticks,
        "gripper_raw": gripper_raw,
    }


def target_preflight(runtime: Any, mapping: Any) -> dict[str, Any]:
    target_raw = np.asarray(runtime.start_raw_deg, dtype=np.float64)
    target_urdf = np.asarray(runtime.start_urdf_deg, dtype=np.float64)
    mapping.validate_joint_order(runtime.joint_order)
    reconstructed_raw = mapping.urdf_degrees_to_raw_degrees(target_urdf)
    target_ticks = mapping.arm_degrees_to_raw_ticks(target_raw)
    ranges = {
        name: {
            "motor_id": int(mapping.calibration[name]["id"]),
            "range_min": int(mapping.calibration[name]["range_min"]),
            "target_tick": int(target_ticks[name]),
            "range_max": int(mapping.calibration[name]["range_max"]),
            "inside": bool(
                int(mapping.calibration[name]["range_min"])
                <= target_ticks[name]
                <= int(mapping.calibration[name]["range_max"])
            ),
        }
        for name in ARM_JOINTS
    }
    finite = bool(np.isfinite(target_raw).all() and np.isfinite(target_urdf).all())
    mapping_consistency = bool(np.allclose(reconstructed_raw, target_raw, atol=1e-9, rtol=0.0))
    representable = all(item["inside"] and 0 <= item["target_tick"] <= 4095 for item in ranges.values())
    return {
        "target_source": str(runtime.physical_start_path),
        "joint_order": list(runtime.joint_order),
        "target_raw_deg": target_raw,
        "target_urdf_deg": target_urdf,
        "target_raw_ticks": target_ticks,
        "finite": finite,
        "mapping_consistency": mapping_consistency,
        "motor_calibration_ranges": ranges,
        "raw_representable": representable,
        "result": "PASS" if finite and mapping_consistency and representable else "FAIL",
    }


def execute_controlled_move(
    bus: Any,
    mapping: Any,
    target_raw_deg: np.ndarray,
    *,
    duration_s: float,
    steps: int,
    settle_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    persist: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if duration_s <= 0.0 or steps <= 0 or settle_s < 0.0:
        raise ValueError("move duration/steps must be positive and settle must be nonnegative")
    target = np.asarray(target_raw_deg, dtype=np.float64)
    if target.shape != (5,) or not np.isfinite(target).all():
        raise ValueError("start target must contain five finite RAW-degree values")

    before = read_snapshot(bus, mapping, clock)
    report: dict[str, Any] = {
        "pre_move": before,
        "move": {
            "duration_configured_s": duration_s,
            "steps": steps,
            "interpolation": "linear in canonical RAW degrees",
            "goal_position_arm_write_attempts": 0,
            "goal_position_arm_physical_writes": 0,
            "gripper_goal_writes": 0,
            "pid_writes": 0,
            "torque_writes": 0,
            "other_register_writes": 0,
            "tracking": [],
        },
    }
    if persist is not None:
        persist(report)

    start = float(clock())
    interval = duration_s / steps
    previous_actual = np.asarray(before["arm_raw_deg"], dtype=np.float64)
    for step in range(1, steps + 1):
        desired = before["arm_raw_deg"] + (step / steps) * (target - before["arm_raw_deg"])
        goal = {name: float(value) for name, value in zip(ARM_JOINTS, desired, strict=True)}
        report["move"]["goal_position_arm_write_attempts"] += 1
        bus.sync_write("Goal_Position", goal, normalize=True)
        report["move"]["goal_position_arm_physical_writes"] += 1

        actual_values = {
            name: float(value)
            for name, value in bus.sync_read("Present_Position", normalize=True).items()
        }
        actual = _finite_arm(actual_values, "movement Present_Position")
        discontinuity = np.abs(actual - previous_actual)
        if np.max(discontinuity) > MAX_READ_DISCONTINUITY_DEG:
            raise RuntimeError(
                f"unexpected Present_Position discontinuity: {discontinuity.tolist()} degrees"
            )
        report["move"]["tracking"].append(
            {
                "step": step,
                "timestamp_monotonic": float(clock()),
                "desired_raw_deg": desired,
                "actual_raw_deg": actual,
                "tracking_error_deg": actual - desired,
                "read_discontinuity_deg": discontinuity,
            }
        )
        previous_actual = actual
        if persist is not None:
            persist(report)
        deadline = start + step * interval
        sleeper(max(0.0, deadline - float(clock())))
        if float(clock()) - start > duration_s + 2.0:
            raise TimeoutError("controlled start move exceeded its duration allowance")

    report["move"]["actual_duration_s"] = float(clock()) - start
    sleeper(settle_s)
    report["post_move"] = read_snapshot(bus, mapping, clock)
    if persist is not None:
        persist(report)
    return report


def compare_ik_limits(target_urdf: np.ndarray, actual_urdf: np.ndarray) -> dict[str, Any]:
    # Deliberately imported/constructed only after physical readback: IK never
    # creates or changes the recorded start movement target.
    from ik.ik_solver_v7 import DLSInverseKinematicsV7

    ik = DLSInverseKinematicsV7(str(DEPLOY_DIR / "urdf" / "so_arm_with_gopro_final.urdf"))
    comparison = {}
    for index, name in enumerate(ARM_JOINTS):
        physical = tuple(float(value) for value in np.rad2deg(ik.PHYSICAL_JOINT_LIMITS_RAD[name]))
        safe = tuple(float(value) for value in np.rad2deg(ik.safe_limits[name]))
        nominal = float(target_urdf[index])
        actual = float(actual_urdf[index])
        comparison[name] = {
            "nominal_start_deg": nominal,
            "actual_start_deg": actual,
            "ik_physical_min_deg": physical[0],
            "ik_physical_max_deg": physical[1],
            "ik_safe_min_deg": safe[0],
            "ik_safe_max_deg": safe[1],
            "nominal_inside_physical": physical[0] <= nominal <= physical[1],
            "actual_inside_physical": physical[0] <= actual <= physical[1],
            "nominal_inside_safe": safe[0] <= nominal <= safe[1],
            "actual_inside_safe": safe[0] <= actual <= safe[1],
        }
    return {
        "source": str(DEPLOY_DIR / "ik" / "ik_solver_v7.py"),
        "variable": "DLSInverseKinematicsV7.PHYSICAL_JOINT_LIMITS_RAD / instance.safe_limits",
        "safe_margin": "90% centered joint range",
        "joints": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-one-arm-move", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))
    from config.runtime_config import load_runtime_config
    from control.joint_mapping import JointMapping

    runtime = load_runtime_config(DEPLOY_DIR / "config" / "deployment.yaml")
    mapping = JointMapping(runtime.motor_mapping_path, runtime.motor_calibration)
    preflight = target_preflight(runtime, mapping)
    if preflight["result"] != "PASS":
        raise SystemExit(f"canonical start preflight failed: {preflight}")
    if not args.execute:
        print(json.dumps({"mode": "DRY_RUN", "preflight": preflight}, indent=2, default=_json_default))
        return 0
    if not args.confirm_one_arm_move:
        raise SystemExit("hardware move blocked without --confirm-one-arm-move")

    motor_config = runtime.deployment["motor"]
    if args.port is not None and args.port != motor_config["port"]:
        raise SystemExit("noncanonical motor port override is forbidden")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "archive"
        / "generated_results"
        / "deployment"
        / f"step4a_start_limit_check_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "summary.json"
    report: dict[str, Any] = {
        "status": "PREFLIGHT_PASS",
        "wall_started": datetime.now().astimezone().isoformat(),
        "preflight": preflight,
        "target_generated_by_ik": False,
        "camera_access": False,
        "ssh_access": False,
        "smolvla_access": False,
    }
    _write_report(result_path, report)

    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(
        SO100FollowerConfig(
            port=motor_config["port"],
            use_degrees=True,
            id=runtime.motor_calibration.stem,
            calibration_dir=runtime.motor_calibration.parent,
            disable_torque_on_disconnect=False,
        )
    )
    connected = False

    def persist(move_report: dict[str, Any]) -> None:
        report["movement"] = move_report
        _write_report(result_path, report)

    try:
        follower.bus.connect(handshake=True)
        connected = True
        report["status"] = "CONNECTED"
        _write_report(result_path, report)
        movement = execute_controlled_move(
            follower.bus,
            mapping,
            np.asarray(runtime.start_raw_deg, dtype=np.float64),
            duration_s=float(motor_config["start_move_duration_s"]),
            steps=int(motor_config["start_move_steps"]),
            settle_s=float(motor_config["settle_duration_s"]),
            persist=persist,
        )
        post = movement["post_move"]
        target_urdf = np.asarray(runtime.start_urdf_deg, dtype=np.float64)
        error = np.asarray(post["arm_urdf_deg"], dtype=np.float64) - target_urdf
        max_error = float(np.max(np.abs(error)))
        report["arrival"] = {
            "error_urdf_deg": error,
            "abs_error_urdf_deg": np.abs(error),
            "max_abs_error_deg": max_error,
            "tolerance_deg": ARRIVAL_TOLERANCE_DEG,
            "tolerance_source": "active hardware diagnostic strict 1-degree gate",
            "result": "PASS" if max_error <= ARRIVAL_TOLERANCE_DEG else "FAIL",
        }
        report["ik_limit_comparison"] = compare_ik_limits(target_urdf, post["arm_urdf_deg"])
        joints = report["ik_limit_comparison"]["joints"]
        conflict = any(
            not item["nominal_inside_physical"]
            or not item["actual_inside_physical"]
            or not item["nominal_inside_safe"]
            or not item["actual_inside_safe"]
            for item in joints.values()
        )
        report["canonical_start_ik_limit_conflict"] = conflict
        report["status"] = "PASS" if report["arrival"]["result"] == "PASS" else "START_ARRIVAL_FAIL"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if connected:
            follower.bus.disconnect(disable_torque=False)
        report["wall_finished"] = datetime.now().astimezone().isoformat()
        _write_report(result_path, report)
        print(json.dumps(report, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
