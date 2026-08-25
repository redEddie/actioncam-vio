"""Explicit-lifecycle JSON-lines telemetry with no import-time file creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class RuntimeLogger:
    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self._stream = None

    def open(self) -> "RuntimeLogger":
        if self._stream is not None:
            raise RuntimeError("telemetry logger is already open")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_path.open("x", encoding="utf-8")
        return self

    def log(self, record: Mapping[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("telemetry logger is not open")
        forbidden = {"password", "passwd", "sshpass"}
        if forbidden.intersection(str(key).lower() for key in record):
            raise ValueError("secret-bearing telemetry fields are forbidden")
        self._stream.write(json.dumps(dict(record), separators=(",", ":"), default=_json_default) + "\n")
        self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "RuntimeLogger":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"telemetry value is not JSON serializable: {type(value).__name__}")


__all__ = ["RuntimeLogger"]
