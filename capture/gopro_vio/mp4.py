"""Minimal ISO-BMFF (MP4) parser — just enough to locate GoPro's `gpmd`
telemetry track samples and basic video track info, without external tools.
"""
from __future__ import annotations

import datetime as _dt
import struct
from dataclasses import dataclass, field

_CONTAINERS = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"udta",
}


@dataclass
class TrackInfo:
    handler: str = ""
    codec: str = ""
    timescale: int = 0
    duration: float = 0.0          # seconds
    width: int = 0
    height: int = 0
    sample_sizes: list[int] = field(default_factory=list)
    sample_offsets: list[int] = field(default_factory=list)
    sample_times: list[float] = field(default_factory=list)   # decode time, s
    sample_durations: list[float] = field(default_factory=list)  # s


@dataclass
class Mp4Info:
    creation_time: _dt.datetime | None = None
    tracks: list[TrackInfo] = field(default_factory=list)

    def find_track(self, codec: str) -> TrackInfo | None:
        for t in self.tracks:
            if t.codec == codec:
                return t
        return None


def _iter_boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, btype = struct.unpack_from(">I4s", buf, pos)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            break
        yield btype, pos + hdr, pos + size
        pos += size


def _mp4_epoch(seconds: int) -> _dt.datetime | None:
    if seconds == 0:
        return None
    epoch = _dt.datetime(1904, 1, 1, tzinfo=_dt.timezone.utc)
    return epoch + _dt.timedelta(seconds=seconds)


def _parse_stts(buf, s, e):
    count = struct.unpack_from(">I", buf, s + 4)[0]
    entries = []
    for i in range(count):
        n, delta = struct.unpack_from(">II", buf, s + 8 + i * 8)
        entries.append((n, delta))
    return entries


def _parse_stsz(buf, s, e):
    uniform, count = struct.unpack_from(">II", buf, s + 4)
    if uniform:
        return [uniform] * count
    return list(struct.unpack_from(f">{count}I", buf, s + 12))


def _parse_stco(buf, s, e, large=False):
    count = struct.unpack_from(">I", buf, s + 4)[0]
    fmt = ">%d%s" % (count, "Q" if large else "I")
    return list(struct.unpack_from(fmt, buf, s + 8))


def _parse_stsc(buf, s, e):
    count = struct.unpack_from(">I", buf, s + 4)[0]
    entries = []
    for i in range(count):
        first_chunk, spc, _desc = struct.unpack_from(">III", buf, s + 8 + i * 12)
        entries.append((first_chunk, spc))
    return entries


def parse(path: str) -> Mp4Info:
    with open(path, "rb") as f:
        # Load only moov into memory; find it by scanning top-level boxes.
        moov = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            size, btype = struct.unpack(">I4s", hdr)
            hdr_len = 8
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                hdr_len = 16
            elif size == 0:
                raise ValueError("box with size 0 not supported")
            if btype == b"moov":
                moov = f.read(size - hdr_len)
                break
            f.seek(size - hdr_len, 1)
    if moov is None:
        raise ValueError(f"No moov box found in {path}")

    info = Mp4Info()
    for btype, s, e in _iter_boxes(moov, 0, len(moov)):
        if btype == b"mvhd":
            version = moov[s]
            if version == 1:
                ct = struct.unpack_from(">Q", moov, s + 4)[0]
            else:
                ct = struct.unpack_from(">I", moov, s + 4)[0]
            info.creation_time = _mp4_epoch(ct)
        elif btype == b"trak":
            info.tracks.append(_parse_trak(moov, s, e))
    return info


def _parse_trak(buf: bytes, start: int, end: int) -> TrackInfo:
    tr = TrackInfo()
    stts = stsz = stco = stsc = None

    def walk(s, e):
        nonlocal stts, stsz, stco, stsc
        for btype, bs, be in _iter_boxes(buf, s, e):
            if btype in _CONTAINERS:
                walk(bs, be)
            elif btype == b"tkhd":
                version = buf[bs]
                off = bs + (32 if version == 0 else 44)
                # width/height are 16.16 fixed at the end of tkhd
                w, h = struct.unpack_from(">II", buf, be - 8)
                tr.width, tr.height = w >> 16, h >> 16
            elif btype == b"mdhd":
                version = buf[bs]
                if version == 1:
                    ts, dur = struct.unpack_from(">IQ", buf, bs + 20)
                else:
                    ts, dur = struct.unpack_from(">II", buf, bs + 12)
                tr.timescale = ts
                tr.duration = dur / ts if ts else 0.0
            elif btype == b"hdlr":
                tr.handler = buf[bs + 8:bs + 12].decode("ascii", "replace")
            elif btype == b"stsd":
                # first sample entry's fourcc
                tr.codec = buf[bs + 12:bs + 16].decode("ascii", "replace")
            elif btype == b"stts":
                stts = _parse_stts(buf, bs, be)
            elif btype == b"stsz":
                stsz = _parse_stsz(buf, bs, be)
            elif btype == b"stco":
                stco = _parse_stco(buf, bs, be, large=False)
            elif btype == b"co64":
                stco = _parse_stco(buf, bs, be, large=True)
            elif btype == b"stsc":
                stsc = _parse_stsc(buf, bs, be)

    walk(start, end)

    if stsz:
        tr.sample_sizes = stsz
        # decode timestamps from stts
        t = 0
        times, durs = [], []
        for n, delta in (stts or []):
            for _ in range(n):
                times.append(t / tr.timescale)
                durs.append(delta / tr.timescale)
                t += delta
        tr.sample_times = times[: len(stsz)]
        tr.sample_durations = durs[: len(stsz)]

        # resolve file offsets via stsc/stco
        if stco and stsc:
            offsets = []
            n_chunks = len(stco)
            # expand stsc to per-chunk samples-per-chunk
            spc_per_chunk = []
            for i, (first_chunk, spc) in enumerate(stsc):
                last = stsc[i + 1][0] - 1 if i + 1 < len(stsc) else n_chunks
                spc_per_chunk += [spc] * (last - first_chunk + 1)
            si = 0
            for chunk_idx, chunk_off in enumerate(stco):
                off = chunk_off
                for _ in range(spc_per_chunk[chunk_idx]):
                    if si >= len(stsz):
                        break
                    offsets.append(off)
                    off += stsz[si]
                    si += 1
            tr.sample_offsets = offsets
    return tr


def read_track_samples(path: str, track: TrackInfo) -> list[bytes]:
    """Read raw sample payloads for a track (used for the small gpmd track)."""
    out = []
    with open(path, "rb") as f:
        for off, size in zip(track.sample_offsets, track.sample_sizes):
            f.seek(off)
            out.append(f.read(size))
    return out
