#!/usr/bin/env python3
"""Deprecated compatibility entry point.

Use ``../../build_umi_zarr.py`` directly.  This wrapper intentionally delegates
to it so this historical path cannot re-apply camera-to-TCP offsets.
"""
from pathlib import Path
import runpy
import sys

sys.argv[0] = str(Path(__file__).resolve().parents[2] / "build_umi_zarr.py")
runpy.run_path(sys.argv[0], run_name="__main__")
