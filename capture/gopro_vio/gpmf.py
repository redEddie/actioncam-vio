"""GPMF (GoPro Metadata Format) KLV parser.

Format reference: https://github.com/gopro/gpmf-parser
Each KLV item: 4-byte FourCC key | 1-byte type | 1-byte struct size |
2-byte big-endian repeat count | payload padded to 4 bytes.
Type 0x00 marks a nested container.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

_SCALARS = {
    ord("b"): ("b", 1), ord("B"): ("B", 1),
    ord("s"): ("h", 2), ord("S"): ("H", 2),
    ord("l"): ("i", 4), ord("L"): ("I", 4),
    ord("f"): ("f", 4), ord("d"): ("d", 8),
    ord("j"): ("q", 8), ord("J"): ("Q", 8),
}


@dataclass
class KLV:
    key: str
    type: int
    ssize: int
    repeat: int
    raw: bytes
    children: list["KLV"] = field(default_factory=list)

    def find(self, key: str) -> "KLV | None":
        for c in self.children:
            if c.key == key:
                return c
        return None

    def find_all(self, key: str) -> list["KLV"]:
        return [c for c in self.children if c.key == key]

    @property
    def value(self):
        """Decode payload according to type."""
        t = self.type
        if t == 0:
            return self.children
        if t in _SCALARS:
            fmt, width = _SCALARS[t]
            n = (self.ssize // width) * self.repeat
            vals = struct.unpack(f">{n}{fmt}", self.raw[: n * width])
            per = self.ssize // width
            if per > 1:
                return [vals[i * per:(i + 1) * per] for i in range(self.repeat)]
            return list(vals)
        if t in (ord("c"), ord("U")):
            return self.raw[: self.ssize * self.repeat].decode("latin-1").rstrip("\x00")
        if t == ord("F"):
            return [self.raw[i * 4:(i + 1) * 4].decode("latin-1") for i in range(self.repeat)]
        if t == ord("q"):  # Q15.16 fixed point
            n = (self.ssize // 4) * self.repeat
            vals = struct.unpack(f">{n}i", self.raw[: n * 4])
            return [v / 65536.0 for v in vals]
        return self.raw


def parse(buf: bytes) -> list[KLV]:
    return _parse_level(buf, 0, len(buf))


def _parse_level(buf: bytes, start: int, end: int) -> list[KLV]:
    items = []
    pos = start
    while pos + 8 <= end:
        key = buf[pos:pos + 4].decode("latin-1")
        if key == "\x00\x00\x00\x00":
            break
        t, ssize = buf[pos + 4], buf[pos + 5]
        repeat = struct.unpack_from(">H", buf, pos + 6)[0]
        dlen = ssize * repeat
        padded = (dlen + 3) & ~3
        raw = buf[pos + 8: pos + 8 + dlen]
        klv = KLV(key, t, ssize, repeat, raw)
        if t == 0:
            klv.children = _parse_level(buf, pos + 8, pos + 8 + dlen)
        items.append(klv)
        pos += 8 + padded
    return items


@dataclass
class SensorStream:
    """Concatenated sensor samples over the whole video with timestamps."""
    key: str
    name: str
    units: str
    data: np.ndarray          # (N, C) scaled to SI units, raw GPMF axis order
    t: np.ndarray             # (N,) seconds on the video/track timeline
    orientation: str = ""     # ORIN string if present, e.g. "ZXY"


def _apply_scale(vals, scal) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if scal is None:
        return arr
    s = np.asarray(scal, dtype=np.float64).ravel()
    if s.size == 1:
        return arr / s[0]
    return arr / s[None, : arr.shape[1]]


def extract_sensor(payloads: list[bytes], payload_times: list[float],
                   payload_durations: list[float], fourcc: str) -> SensorStream | None:
    """Extract one sensor stream (e.g. ACCL / GYRO) across all payloads.

    Per-sample timestamps: a least-squares linear fit of payload start time vs
    cumulative sample index (the sensor clock is far more stable than the
    payload packaging), following gpmf-parser's GetGPMFSampleRate approach.
    """
    chunks, counts = [], []
    name = units = orin = ""
    start_indices, start_times = [], []
    total = 0
    for payload, t0, dur in zip(payloads, payload_times, payload_durations):
        for devc in parse(payload):
            if devc.key != "DEVC":
                continue
            for strm in devc.find_all("STRM"):
                node = strm.find(fourcc)
                if node is None:
                    continue
                scal = strm.find("SCAL")
                data = _apply_scale(node.value, scal.value if scal else None)
                if not name:
                    stnm = strm.find("STNM")
                    siun = strm.find("SIUN") or strm.find("UNIT")
                    orin_node = strm.find("ORIN")
                    name = stnm.value if stnm else fourcc
                    units = siun.value if siun else ""
                    orin = orin_node.value if orin_node else ""
                chunks.append(data)
                counts.append(len(data))
                start_indices.append(total)
                start_times.append(t0)
                total += len(data)
    if not chunks:
        return None

    data = np.concatenate(chunks, axis=0)
    idx = np.asarray(start_indices, dtype=np.float64)
    ts = np.asarray(start_times, dtype=np.float64)
    if len(idx) >= 2:
        # t = a * sample_index + b
        a, b = np.polyfit(idx, ts, 1)
        t = a * np.arange(total) + b
    else:
        dur = payload_durations[0] if payload_durations else 1.0
        t = start_times[0] + np.arange(total) * (dur / max(total, 1))
    return SensorStream(fourcc, name, units, data, t, orin)
