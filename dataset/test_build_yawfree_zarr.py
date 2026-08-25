#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "dataset/build_yawfree_zarr.py"
SPEC = importlib.util.spec_from_file_location("canonical_yawfree_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
yawfree = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = yawfree
SPEC.loader.exec_module(yawfree)


class YawFreeBuilderTest(unittest.TestCase):
    def test_trusted_euler_zyx_projection(self) -> None:
        euler = np.asarray(
            [
                [0.8, -0.2, 0.4],
                [-1.1, 0.3, -0.5],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        rotvec = Rotation.from_euler("ZYX", euler).as_rotvec()
        actual = yawfree.yawfree_roll_pitch(rotvec)
        expected = euler[:, [2, 1]].astype(np.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_project_relative_defaults_and_existing_scope_audit(self) -> None:
        args = yawfree.parse_args(["--dry-run"])
        self.assertEqual(
            args.source.resolve(),
            PROJECT_ROOT / "artifacts/zarr/replay_buffer.zarr",
        )
        self.assertEqual(
            args.output.resolve(),
            PROJECT_ROOT / "artifacts/zarr/replay_buffer_yawfree.zarr",
        )
        if not args.source.exists():
            self.skipTest("local artifacts store not present (data lives outside git)")
        # Verify canonical 56-ep source audit
        report = yawfree.build(args)
        self.assertEqual(report["source_frames"], 55_751)
        self.assertEqual(report["source_episodes"], 56)

        # Verify migrated 77-ep canonical artifact in bootstrap workspace
        migrated_output = PROJECT_ROOT / "artifacts/datasets/202608161336/02_zarr/replay_buffer_yawfree.zarr"
        if migrated_output.exists():
            args_migrated = yawfree.parse_args(["--dry-run", "-o", str(migrated_output)])
            report_migrated = yawfree.build(args_migrated)
            self.assertFalse(report_migrated["existing_output_matches_source_scope"])
            self.assertEqual(report_migrated["existing_output"]["frames"], 74_109)
            self.assertEqual(report_migrated["existing_output"]["episodes"], 77)


if __name__ == "__main__":
    unittest.main()
