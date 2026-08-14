from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEPLOY = ROOT / "4_deploy"
if str(DEPLOY) not in sys.path:
    sys.path.insert(0, str(DEPLOY))

from inference.camera import FrameSnapshot
from inference.observation import ActualStateSnapshot, build_observation
from trajectory.chunk_scheduler import ChunkScheduler


def _called_attributes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_full_shadow_evaluator_has_no_motor_write_or_start_move_call():
    path = DEPLOY / "evaluation" / "live_shadow" / "step4_full_shadow.py"
    called = _called_attributes(path)
    assert "write_goal_ticks" not in called
    assert "move_to_start_ticks" not in called
    assert "configure_and_verify_pid" not in called
    assert "enable_torque" not in called
    assert "disable_torque" not in called


def test_run_live_shadow_no_longer_stops_before_ik_preview():
    source = (DEPLOY / "run_live.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    control_tick = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_control_tick"
    )
    shadow_references = [
        node for node in ast.walk(control_tick) if isinstance(node, ast.Attribute) and node.attr == "SHADOW"
    ]
    assert shadow_references == []


def test_scheduler_reports_nonnegative_reanchor_and_overlap_timings():
    scheduler = ChunkScheduler()
    frame = FrameSnapshot(np.zeros((256, 256, 3), dtype=np.uint8), 1.0, 1, (1080, 1920, 3))
    actual = ActualStateSnapshot(1.0, np.zeros(5), np.zeros(5), np.zeros(6), 0.0)
    scheduler.register_request(build_observation(0, frame, actual))
    result = scheduler.handle_response(0, np.zeros((1, 15, 6)), 1.1, np.zeros(6))
    assert result["status"] == "INTEGRATED"
    assert set(result["timing_s"]) == {"reanchor", "overlap", "total"}
    assert all(value >= 0.0 for value in result["timing_s"].values())
