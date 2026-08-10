#!/usr/bin/env python3
"""Read-only live monitor for one fixed yaw-free policy start pose.

Fixed reference: episode 12, frame 0.  It is the medoid of the 56 recorded
episode starts (the real recorded start nearest to the group centre), so it is
a stable canonical reference rather than an arithmetic mean pose that no
episode actually used.

The program reads only ``Present_Position`` and computes FK.  It contains no
motor goal, torque, camera, policy, or IK command path.  Green means the
*measured* encoder-FK value is within the displayed tolerance, not that the
robot has moved or that the camera scene matches episode 12.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import deploy_smolvla_yawfree as deploy


REFERENCE_EPISODE = 12
REFERENCE_FRAME = 0
TRAJECTORY_PATH = (
    deploy.PROJECT_ROOT
    / f"1_data_pipeline/actioncam-vio/Episode_result/episode_{REFERENCE_EPISODE}/world/trajectory_tcp_robot.csv"
)
RECORD_PATH = deploy.DEPLOY_DIR / "yawfree_canonical_start_record.json"
# Canonical-start positional display/record gate.  The user-selected per-axis
# tolerance is +/-7 mm.  The corresponding vector threshold is the enclosing
# cube corner, 7*sqrt(3)=12.12 mm, so three green XYZ axes can also make the
# overall XYZ state green and recordable.
XYZ_AXIS_TOL_MM = 7.0
XYZ_VECTOR_TOL_MM = XYZ_AXIS_TOL_MM * np.sqrt(3.0)  # 12.124 mm
ROLL_PITCH_AXIS_TOL_DEG = 5.0
ROLL_PITCH_VECTOR_TOL_DEG = 5.0

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until q/ESC/Ctrl-C")
    parser.add_argument("--no-gui", action="store_true", help="terminal output only")
    parser.add_argument(
        "--record-on-ready", action=argparse.BooleanOptionalAction, default=True,
        help="save the measured arm encoder angles once when every start-pose gate passes (default: true)",
    )
    return parser.parse_args()


def read_reference() -> tuple[np.ndarray, np.ndarray, float]:
    if not TRAJECTORY_PATH.is_file():
        raise FileNotFoundError(TRAJECTORY_PATH)
    with TRAJECTORY_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if REFERENCE_FRAME >= len(rows):
        raise RuntimeError(f"reference frame {REFERENCE_FRAME} is absent from {TRAJECTORY_PATH}")
    row = rows[REFERENCE_FRAME]
    if str(row.get("ok", "1")).strip() not in ("1", "true", "True", "1.0"):
        raise RuntimeError("fixed reference frame is marked invalid")
    xyz = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
    quat = np.array([float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])])
    euler = Rotation.from_quat(quat).as_euler("ZYX", degrees=False)
    return xyz, euler[[2, 1]], float(row["timestamp"])


def wrap_radians(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi


def colour_terminal(ok: bool, value: str) -> str:
    return f"{GREEN if ok else RED}{value}{RESET}"


def panel(reference_xyz: np.ndarray, reference_rp: np.ndarray, current_xyz: np.ndarray, current_rp: np.ndarray) -> np.ndarray:
    """Build an OpenCV panel; green/red is per coordinate tolerance."""
    canvas = np.full((590, 1050, 3), 25, dtype=np.uint8)
    white, muted = (235, 235, 235), (175, 175, 175)
    green, red, amber = (55, 220, 55), (55, 55, 235), (0, 190, 255)
    cv2.putText(canvas, "Yaw-free canonical start alignment  |  READ-ONLY", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.86, white, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Reference: episode 12 / frame 0 (real medoid start); yaw ignored", (22, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.60, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "green = current FK is within tolerance", (22, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.60, amber, 1, cv2.LINE_AA)
    cv2.putText(canvas, "axis", (25, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.66, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "target", (175, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.66, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "current FK", (400, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.66, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "target - current", (645, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.66, muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "tolerance", (895, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.66, muted, 1, cv2.LINE_AA)

    delta_xyz_mm = (reference_xyz - current_xyz) * 1000.0
    delta_rp_deg = np.rad2deg(wrap_radians(reference_rp - current_rp))
    rows = [
        ("X", reference_xyz[0], current_xyz[0], delta_xyz_mm[0], "m", f"+/-{XYZ_AXIS_TOL_MM:.2f} mm", abs(delta_xyz_mm[0]) <= XYZ_AXIS_TOL_MM),
        ("Y", reference_xyz[1], current_xyz[1], delta_xyz_mm[1], "m", f"+/-{XYZ_AXIS_TOL_MM:.2f} mm", abs(delta_xyz_mm[1]) <= XYZ_AXIS_TOL_MM),
        ("Z", reference_xyz[2], current_xyz[2], delta_xyz_mm[2], "m", f"+/-{XYZ_AXIS_TOL_MM:.2f} mm", abs(delta_xyz_mm[2]) <= XYZ_AXIS_TOL_MM),
        ("roll", np.rad2deg(reference_rp[0]), np.rad2deg(current_rp[0]), delta_rp_deg[0], "deg", f"+/-{ROLL_PITCH_AXIS_TOL_DEG:.0f} deg", abs(delta_rp_deg[0]) <= ROLL_PITCH_AXIS_TOL_DEG),
        ("pitch", np.rad2deg(reference_rp[1]), np.rad2deg(current_rp[1]), delta_rp_deg[1], "deg", f"+/-{ROLL_PITCH_AXIS_TOL_DEG:.0f} deg", abs(delta_rp_deg[1]) <= ROLL_PITCH_AXIS_TOL_DEG),
    ]
    for index, (name, target, current, error, unit, tolerance, ok) in enumerate(rows):
        y = 190 + index * 60
        colour = green if ok else red
        cv2.putText(canvas, name, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.76, colour, 2, cv2.LINE_AA)
        if unit == "m":
            target_text, current_text, error_text = f"{target:+.5f} m", f"{current:+.5f} m", f"{error:+.2f} mm"
        else:
            target_text, current_text, error_text = f"{target:+.2f} deg", f"{current:+.2f} deg", f"{error:+.2f} deg"
        cv2.putText(canvas, target_text, (175, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, current_text, (400, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, error_text, (645, y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, tolerance, (895, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1, cv2.LINE_AA)

    xyz_norm = float(np.linalg.norm(delta_xyz_mm))
    rp_norm = float(np.linalg.norm(delta_rp_deg))
    overall_ok = xyz_norm <= XYZ_VECTOR_TOL_MM and rp_norm <= ROLL_PITCH_VECTOR_TOL_DEG
    overall_colour = green if overall_ok else red
    text = f"overall: XYZ norm {xyz_norm:.2f} mm / {XYZ_VECTOR_TOL_MM:.0f} mm, roll-pitch norm {rp_norm:.2f} deg / {ROLL_PITCH_VECTOR_TOL_DEG:.0f} deg"
    cv2.putText(canvas, text, (22, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.62, overall_colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, "READY TO COMPARE POLICY INPUT" if overall_ok else "NOT YET AT CANONICAL START", (22, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.78, overall_colour, 2, cv2.LINE_AA)
    return canvas


def terminal_status(reference_xyz: np.ndarray, reference_rp: np.ndarray, current_xyz: np.ndarray, current_rp: np.ndarray) -> None:
    delta_xyz_mm = (reference_xyz - current_xyz) * 1000.0
    delta_rp_deg = np.rad2deg(wrap_radians(reference_rp - current_rp))
    target = [*reference_xyz, *np.rad2deg(reference_rp)]
    current = [*current_xyz, *np.rad2deg(current_rp)]
    delta = [*delta_xyz_mm, *delta_rp_deg]
    units = ["m", "m", "m", "deg", "deg"]
    thresholds = [XYZ_AXIS_TOL_MM] * 3 + [ROLL_PITCH_AXIS_TOL_DEG] * 2
    labels = ["X", "Y", "Z", "roll", "pitch"]
    print("\nREFERENCE ep12 frame0 | target-current: positive means target needs increase")
    for i, label in enumerate(labels):
        ok = abs(delta[i]) <= thresholds[i]
        error_unit = "mm" if i < 3 else "deg"
        tolerance_text = f"{thresholds[i]:.2f}" if i < 3 else f"{thresholds[i]:.0f}"
        line = f"{label:5s} target={target[i]:+9.5f} {units[i]:3s} current={current[i]:+9.5f} {units[i]:3s} error={delta[i]:+8.2f} {error_unit:3s} tol=+/-{tolerance_text} {error_unit}"
        print(colour_terminal(ok, line))
    xyz_norm = float(np.linalg.norm(delta_xyz_mm))
    rp_norm = float(np.linalg.norm(delta_rp_deg))
    ready = xyz_norm <= XYZ_VECTOR_TOL_MM and rp_norm <= ROLL_PITCH_VECTOR_TOL_DEG
    print(colour_terminal(ready, f"OVERALL: XYZ={xyz_norm:.2f}/{XYZ_VECTOR_TOL_MM:.0f} mm, RP={rp_norm:.2f}/{ROLL_PITCH_VECTOR_TOL_DEG:.0f} deg"), flush=True)


def start_ready(reference_xyz: np.ndarray, reference_rp: np.ndarray, current_xyz: np.ndarray, current_rp: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """Return the exact gate used for persistent start-angle recording."""
    delta_xyz_mm = (reference_xyz - current_xyz) * 1000.0
    delta_rp_deg = np.rad2deg(wrap_radians(reference_rp - current_rp))
    components_ok = (
        np.all(np.abs(delta_xyz_mm) <= XYZ_AXIS_TOL_MM)
        and np.all(np.abs(delta_rp_deg) <= ROLL_PITCH_AXIS_TOL_DEG)
    )
    vectors_ok = (
        float(np.linalg.norm(delta_xyz_mm)) <= XYZ_VECTOR_TOL_MM
        and float(np.linalg.norm(delta_rp_deg)) <= ROLL_PITCH_VECTOR_TOL_DEG
    )
    return bool(components_ok and vectors_ok), delta_xyz_mm, delta_rp_deg


def record_canonical_start(
    observed_deg: dict[str, float], observed_raw: dict[str, float], solver, reference_xyz: np.ndarray,
    reference_rp: np.ndarray, current_xyz: np.ndarray, current_rp: np.ndarray,
) -> None:
    """Persist only a measured encoder snapshot whose TCP passed every gate."""
    ready, delta_xyz_mm, delta_rp_deg = start_ready(reference_xyz, reference_rp, current_xyz, current_rp)
    if not ready:
        raise RuntimeError("internal error: refusing to record a start pose that did not pass its FK gate")
    record = {
        "format": "yawfree_canonical_start_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "episode": REFERENCE_EPISODE,
            "frame": REFERENCE_FRAME,
            "trajectory_csv": str(TRAJECTORY_PATH),
            "target_xyz_m": reference_xyz.tolist(),
            "target_roll_pitch_deg": np.rad2deg(reference_rp).tolist(),
            "yaw": "intentionally free and not recorded as a TCP requirement",
        },
        "tolerance": {
            "xyz_axis_mm": float(XYZ_AXIS_TOL_MM),
            "xyz_vector_mm": XYZ_VECTOR_TOL_MM,
            "roll_pitch_axis_deg": ROLL_PITCH_AXIS_TOL_DEG,
            "roll_pitch_vector_deg": ROLL_PITCH_VECTOR_TOL_DEG,
        },
        "measured_at_record": {
            "arm_joint_order": list(solver.joint_names),
            "arm_joint_deg": {name: float(observed_deg[name]) for name in solver.joint_names},
            "gripper_present_deg": float(observed_deg["gripper"]),
            "raw_present_position": {name: float(value) for name, value in observed_raw.items()},
            "fk_xyz_m": current_xyz.tolist(),
            "fk_roll_pitch_deg": np.rad2deg(current_rp).tolist(),
            "target_minus_current_xyz_mm": delta_xyz_mm.tolist(),
            "target_minus_current_roll_pitch_deg": delta_rp_deg.tolist(),
        },
    }
    RECORD_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(
        "CORRECT_START_ARM_JOINT_DEG:",
        {name: round(float(observed_deg[name]), 3) for name in solver.joint_names},
    )
    print(f"CANONICAL_START_RECORDED: {RECORD_PATH}")


def main() -> int:
    args = parse_args()
    if not 0 < args.hz <= 20:
        raise ValueError("--hz must be in (0, 20]")
    if args.frames < 0:
        raise ValueError("--frames must be >= 0")
    reference_xyz, reference_rp, timestamp = read_reference()
    lerobot_src = deploy.DEPLOY_DIR / "Teleop" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    solver = deploy.load_calibrated_solver(limit_margin_ratio=0.95)
    follower = SO100Follower(SO100FollowerConfig(port=args.port, use_degrees=True))
    follower.bus.connect(handshake=True)
    print(f"READ-ONLY START ALIGNMENT. Fixed target: episode 12 frame 0 at t={timestamp:.6f}s. q/ESC/Ctrl-C exits.")
    if args.record_on_ready:
        print(f"When every gate is green, arm encoder snapshot will be saved once to {RECORD_PATH}.")
    else:
        print("Recording disabled: comparison only.")
    try:
        count = 0
        recorded = False
        while args.frames == 0 or count < args.frames:
            observed = follower.bus.sync_read("Present_Position")
            raw_observed = follower.bus.sync_read("Present_Position", normalize=False)
            q_rad = np.deg2rad(np.array([float(observed[name]) for name in solver.joint_names], dtype=np.float64))
            pose = solver.forward_kinematics(q_rad)
            current_xyz = pose[:3, 3]
            euler = Rotation.from_matrix(pose[:3, :3]).as_euler("ZYX", degrees=False)
            current_rp = euler[[2, 1]]
            terminal_status(reference_xyz, reference_rp, current_xyz, current_rp)
            ready, _, _ = start_ready(reference_xyz, reference_rp, current_xyz, current_rp)
            if ready and args.record_on_ready and not recorded:
                record_canonical_start(
                    {name: float(value) for name, value in observed.items()},
                    {name: float(value) for name, value in raw_observed.items()},
                    solver, reference_xyz, reference_rp, current_xyz, current_rp,
                )
                recorded = True
            if not args.no_gui:
                cv2.imshow("UMI fixed start-pose alignment (read-only)", panel(reference_xyz, reference_rp, current_xyz, current_rp))
                key = cv2.waitKey(max(1, int(round(1000.0 / args.hz)))) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                time.sleep(1.0 / args.hz)
            count += 1
    finally:
        if not args.no_gui:
            cv2.destroyAllWindows()
        follower.bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
