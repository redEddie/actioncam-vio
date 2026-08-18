#!/usr/bin/env python3
"""Read-only SO-101 URDF-zero and SmolVLA-start measurement.

This tool deliberately reproduces the calibration environment used by
``deploy_smolvla_yawfree.py`` before importing LeRobot.  It never sends a
motor goal, changes torque, calibrates, configures a motor, or runs IK/policy
code.  Hardware reads require ``--measure``; without it the script performs
only its static provenance audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
DEPLOYMENT_CACHE_ROOT = PROJECT_ROOT / "etc" / "smolvla_cache"
LEROBOT_SRC = DEPLOY_DIR / "Teleop" / "lerobot" / "src"
RESULT_PATH = DEPLOY_DIR / "deployment_domain_motor_urdf_zero_and_start.json"
SOURCE_ZERO_PATH = DEPLOY_DIR / "yawfree_user_defined_joint_zero.json"
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

# Keep this order identical to deploy_smolvla_yawfree.py: these variables are
# set before any LeRobot import.  Do not overwrite the more-specific LeRobot
# environment variables: deployment itself does not overwrite them either.
os.environ["HF_HOME"] = str(DEPLOYMENT_CACHE_ROOT)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="perform the two interactive, read-only captures")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--target-hz", type=float, default=30.0)
    parser.add_argument("--overwrite-result", action="store_true", help="replace an existing result JSON after a new capture")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_calibration_path() -> Path:
    """Mirror LeRobot constants.py without importing LeRobot first."""
    lerobot_home = Path(os.environ.get("HF_LEROBOT_HOME", str(DEPLOYMENT_CACHE_ROOT / "lerobot"))).expanduser()
    calibration_root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", str(lerobot_home / "calibration"))).expanduser()
    return calibration_root / "robots" / "so_follower" / "None.json"


def expected_deployment_calibration_path() -> Path:
    return DEPLOYMENT_CACHE_ROOT / "lerobot" / "calibration" / "robots" / "so_follower" / "None.json"


def load_source_sign_metadata() -> np.ndarray:
    data = json.loads(SOURCE_ZERO_PATH.read_text())
    joints = data.get("joints", {})
    if set(joints) != set(ARM_JOINTS):
        raise RuntimeError(f"invalid source joint-zero file: {SOURCE_ZERO_PATH}")
    sign = np.asarray([float(joints[name]["sign"]) for name in ARM_JOINTS], dtype=np.float64)
    if not np.isin(sign, (-1.0, 1.0)).all():
        raise RuntimeError(f"invalid source sign metadata: {sign.tolist()}")
    return sign


def provenance() -> dict:
    resolved = resolved_calibration_path().resolve()
    expected = expected_deployment_calibration_path().resolve()
    if resolved != expected:
        raise RuntimeError(
            "CALIBRATION_DOMAIN_MATCH=FAIL: resolved calibration differs from deployment path: "
            f"resolved={resolved} expected={expected}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"deployment calibration does not exist: {resolved}")
    return {
        "deployment_cache_root": str(DEPLOYMENT_CACHE_ROOT.resolve()),
        "deployment_calibration_path": str(resolved),
        "deployment_calibration_sha256": sha256(resolved),
        "deployment_calibration_id": None,
        "deployment_robot_config": "SO100FollowerConfig(port=/dev/ttyACM0, use_degrees=True)",
        "position_read_function": "follower.bus.sync_read('Present_Position')",
        "position_normalization": "normalize=True (default); Feetech MotorNormMode.DEGREES",
        "raw_position_read_function": "follower.bus.sync_read('Present_Position', normalize=False)",
        "raw_position_representation": "LeRobot-decoded signed Present_Position register value; no normalization",
        "encoder_units": "degrees",
    }


def print_static_audit(info: dict) -> None:
    print("STATIC_READ_ONLY_AUDIT = PASS")
    print("ACTIVE_CACHE_ROOT =", info["deployment_cache_root"])
    print("RESOLVED_CALIBRATION_PATH =", info["deployment_calibration_path"])
    print("RESOLVED_CALIBRATION_SHA256 =", info["deployment_calibration_sha256"])
    print("CALIBRATION_DOMAIN_MATCH = PASS")
    print("POSITION_READ_FUNCTION =", info["position_read_function"])
    print("POSITION_NORMALIZATION =", info["position_normalization"])
    print("RAW_POSITION_READ_FUNCTION =", info["raw_position_read_function"])
    print("SIGN_APPLICATION_STATUS = UNVERIFIED")
    print("URDF_START_STATUS = DEFERRED_UNTIL_TRAINING_COORDINATE_AUDIT")
    print("FORBIDDEN_OPERATIONS = no goal write / no torque operation / no calibration operation / no IK / no controller")


def wait_for_space(prompt: str) -> None:
    print(prompt, flush=True)
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                key = sys.stdin.read(1)
                if key == "\x03":
                    raise KeyboardInterrupt
                if key == " ":
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def vector_from_reading(values: dict[str, float]) -> np.ndarray:
    return np.asarray([float(values[name]) for name in ARM_JOINTS], dtype=np.float64)


def sample_positions(follower, duration_s: float, target_hz: float) -> tuple[dict, int]:
    sample_count = int(round(duration_s * target_hz))
    if sample_count < 1:
        raise ValueError("duration-s * target-hz must yield at least one sample")
    interval_s = 1.0 / target_hz
    normalized_samples: list[np.ndarray] = []
    raw_samples: list[np.ndarray] = []

    for _ in range(sample_count):
        started = time.monotonic()
        # This is the exact deployment representation, followed immediately by
        # an unnormalized read for provenance only.
        normalized = follower.bus.sync_read("Present_Position")
        raw = follower.bus.sync_read("Present_Position", normalize=False)
        normalized_samples.append(vector_from_reading(normalized))
        raw_samples.append(vector_from_reading(raw))
        time.sleep(max(0.0, interval_s - (time.monotonic() - started)))

    normalized_array = np.asarray(normalized_samples, dtype=np.float64)
    raw_array = np.asarray(raw_samples, dtype=np.float64)

    def statistics(values: np.ndarray) -> dict:
        return {
            "median": np.median(values, axis=0).tolist(),
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "sample_count": int(values.shape[0]),
        }

    return {
        "deployment_encoder_deg": statistics(normalized_array),
        "raw_present_position": statistics(raw_array),
        "raw_position_representation": "LeRobot-decoded signed Present_Position register value; no normalization",
        "raw_read_order": "immediately after each normalized Present_Position read",
    }, sample_count * 2


def print_capture(label: str, capture: dict) -> None:
    normalized = capture["deployment_encoder_deg"]
    raw = capture["raw_present_position"]
    print(f"{label}_DEPLOYMENT_ENCODER_MEDIAN_DEG = {np.round(normalized['median'], 4).tolist()}")
    print(f"{label}_DEPLOYMENT_ENCODER_STD_DEG = {np.round(normalized['std'], 4).tolist()}")
    print(f"{label}_RAW_PRESENT_POSITION_MEDIAN = {np.round(raw['median'], 4).tolist()}")


def main() -> int:
    args = parse_args()
    if args.duration_s <= 0.0 or args.target_hz <= 0.0:
        raise ValueError("duration-s and target-hz must be positive")
    source_sign = load_source_sign_metadata()
    info = provenance()
    print_static_audit(info)
    print("SIGN_SOURCE = existing deployment source: yawfree_user_defined_joint_zero.json / deploy_smolvla_yawfree.py")
    print("SIGN_VALUE =", source_sign.tolist())
    if not args.measure:
        print("NO_HARDWARE_CONNECTION: add --measure only when the operator is ready for the two read-only captures.")
        return 0
    if RESULT_PATH.exists() and not args.overwrite_result:
        raise FileExistsError(f"result already exists: {RESULT_PATH}; inspect it or add --overwrite-result for a new measurement")

    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))
    # Import happens only after the deployment-equivalent cache environment is
    # established and provenance has passed.
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower_path = follower.calibration_fpath.resolve()
    if follower_path != Path(info["deployment_calibration_path"]):
        raise RuntimeError(
            "CALIBRATION_DOMAIN_MATCH=FAIL after LeRobot import: "
            f"follower={follower_path} expected={info['deployment_calibration_path']}"
        )
    if sha256(follower_path) != info["deployment_calibration_sha256"]:
        raise RuntimeError("CALIBRATION_DOMAIN_MATCH=FAIL after LeRobot import: SHA256 changed")

    position_reads = 0
    follower.bus.connect(handshake=True)
    try:
        print("\n[STEP 1]")
        wait_for_space(
            "로봇을 URDF 기준 모든 5 joint가 0°인 물리적 ㄱ자 자세로 맞추십시오.\n"
            "준비되면 SPACE BAR를 누르십시오. (읽기만 수행하며 motor command는 없습니다.)"
        )
        zero_capture, zero_reads = sample_positions(follower, args.duration_s, args.target_hz)
        position_reads += zero_reads
        print_capture("ZERO", zero_capture)

        print("\n[STEP 2]")
        wait_for_space(
            "이제 실제 SmolVLA inference 시작자세로 맞추십시오.\n"
            "준비되면 SPACE BAR를 누르십시오. (읽기만 수행하며 motor command는 없습니다.)"
        )
        start_capture, start_reads = sample_positions(follower, args.duration_s, args.target_hz)
        position_reads += start_reads
        print_capture("START", start_capture)
    finally:
        # False is required: the bus implementation only disables torque when
        # this argument is True.
        if follower.bus.is_connected:
            follower.bus.disconnect(disable_torque=False)

    zero = np.asarray(zero_capture["deployment_encoder_deg"]["median"], dtype=np.float64)
    start = np.asarray(start_capture["deployment_encoder_deg"]["median"], dtype=np.float64)
    start_minus_zero = start - zero

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "joint_order": list(ARM_JOINTS),
        "sign_source": "existing deployment source: yawfree_user_defined_joint_zero.json / deploy_smolvla_yawfree.py",
        "sign_value": source_sign.tolist(),
        "sign_application_status": "UNVERIFIED",
        "urdf_start_status": "DEFERRED_UNTIL_TRAINING_COORDINATE_AUDIT",
        **info,
        "encoder_zero_at_urdf_zero_deg": zero.tolist(),
        "encoder_start_deg": start.tolist(),
        "start_minus_zero_deg": start_minus_zero.tolist(),
        "zero_raw_present_position": zero_capture["raw_present_position"]["median"],
        "start_raw_present_position": start_capture["raw_present_position"]["median"],
        "formula": {
            "encoder_to_urdf": "q_urdf = sign * (encoder_deg - encoder_zero_deg)",
            "urdf_to_encoder": "encoder_deg = q_urdf * sign + encoder_zero_deg",
        },
        "measurement_statistics": {
            "zero": zero_capture,
            "start": start_capture,
            "position_read_transactions": position_reads,
            "goal_position_writes": 0,
            "torque_writes": 0,
            "robot_moved_by_script": False,
        },
        "validation": {
            "calibration_domain_match": "PASS",
            "measurement_coordinate_transform_applied": "NO",
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\nMEASUREMENT COMPLETE")
    print("DEPLOYMENT_ENCODER_ZERO_DEG =", np.round(zero, 4).tolist())
    print("DEPLOYMENT_ENCODER_START_DEG =", np.round(start, 4).tolist())
    print("START_MINUS_ZERO_DEG =", np.round(start_minus_zero, 4).tolist())
    print("SIGN_APPLICATION_STATUS = UNVERIFIED")
    print("URDF_START = NOT YET DEFINED")
    print("RESULT_PATH =", RESULT_PATH)
    print("POSITION_READS =", position_reads)
    print("GOAL_POSITION_WRITES = 0")
    print("TORQUE_WRITES = 0")
    print("ROBOT_MOVED_BY_SCRIPT = NO")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("INTERRUPTED: no motor command, torque write, or calibration write was performed.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
