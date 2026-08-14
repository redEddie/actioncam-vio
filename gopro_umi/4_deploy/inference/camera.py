"""Explicit-lifecycle camera capture and geometric RGB preparation."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameSnapshot:
    image_rgb: np.ndarray
    timestamp: float
    sequence: int
    source_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        image = np.asarray(self.image_rgb, dtype=np.uint8).copy()
        if image.shape != (256, 256, 3):
            raise ValueError(f"transport image must have shape (256,256,3), got {image.shape}")
        image.setflags(write=False)
        object.__setattr__(self, "image_rgb", image)


def prepare_transport_rgb(bgr: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    source = np.asarray(bgr)
    if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
        raise CameraError("camera frame must be a non-empty BGR HxWx3 array")
    rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    side = min(rgb.shape[:2])
    top = (rgb.shape[0] - side) // 2
    left = (rgb.shape[1] - side) // 2
    square = rgb[top : top + side, left : left + side]
    return cv2.resize(square, size, interpolation=cv2.INTER_AREA)


class LatestFrameCamera:
    """Latest-only capture thread; no device is opened until ``start``."""

    def __init__(
        self,
        device: int,
        width: int,
        height: int,
        stale_timeout_s: float,
        *,
        capture_factory: Callable[..., Any] = cv2.VideoCapture,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        self.stale_timeout_s = float(stale_timeout_s)
        if self.stale_timeout_s <= 0.0:
            raise ValueError("stale_timeout_s must be positive")
        self._capture_factory = capture_factory
        self._clock = clock
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: FrameSnapshot | None = None
        self._failure: Exception | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._thread is not None:
            raise CameraError("camera is already started")
        capture = self._capture_factory(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = self._capture_factory(self.device)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"cannot open camera {self.device}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-latest", daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, bgr = self._capture.read()
                if not ok:
                    raise CameraError("camera frame read failed")
                timestamp = float(self._clock())
                self._sequence += 1
                snapshot = FrameSnapshot(prepare_transport_rgb(bgr), timestamp, self._sequence, tuple(bgr.shape))
                with self._lock:
                    self._latest = snapshot
        except Exception as exc:
            self._failure = exc
            self._stop.set()

    def latest(self, now: float | None = None) -> FrameSnapshot:
        if self._failure is not None:
            raise CameraError("camera capture failed") from self._failure
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            raise CameraError("camera has not produced a frame")
        current = float(self._clock() if now is None else now)
        if current - snapshot.timestamp > self.stale_timeout_s:
            raise CameraError("latest camera frame is stale")
        return snapshot

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
        self._thread = None
        self._capture = None


__all__ = ["CameraError", "FrameSnapshot", "LatestFrameCamera", "prepare_transport_rgb"]
