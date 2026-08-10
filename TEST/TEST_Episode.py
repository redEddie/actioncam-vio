"""Compatibility entry point for the validated HERO13 batch pipeline."""
from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    raise SystemExit(subprocess.call(
        [sys.executable, str(root / "batch_reprocess.py"), "--promote"],
        cwd=root,
    ))


if __name__ == "__main__":
    main()
