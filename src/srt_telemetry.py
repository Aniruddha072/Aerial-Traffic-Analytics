"""Parser for DJI-style SRT telemetry, extracted from the hackathon video's
embedded subtitle stream (not a sidecar file -- see docs/decisions.md
Decision 4: `ffmpeg -i <video> -map 0:s:0 <out>.srt`).

Confirmed format (2026-08-22, against the real Intersection_Merged.MP4
extraction): each block is a frame count + timestamp on one line
("FrameCnt: <n> <date> <time>"), followed by a bracket-field line including
[latitude: ..] [longitude: ..] [rel_alt: .. abs_alt: ..] (also carries
[gb_yaw/gb_pitch/gb_roll: ..] -- not parsed here, not needed for L1's
LGV/HGV split, but useful for L4's spatial grounding later).
"""

import re
from dataclasses import dataclass


@dataclass
class TelemetryFrame:
    frame_idx: int
    timestamp_ms: int
    latitude: float
    longitude: float
    rel_altitude_m: float
    abs_altitude_m: float


_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->.*?\n"
    r".*?FrameCnt:\s*(\d+).*?\n"
    r".*?latitude:\s*([-\d.]+)\].*?longitude:\s*([-\d.]+)\].*?"
    r"rel_alt:\s*([-\d.]+)\s+abs_alt:\s*([-\d.]+)",
    re.DOTALL,
)


def parse_srt(path):
    """Parse a DJI-style SRT telemetry file into a list of TelemetryFrame, ordered by frame_idx.

    Raises ValueError if no telemetry blocks parse -- usually means the SRT
    format differs from the layout assumed here; inspect the raw file and
    adjust _BLOCK_RE.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    frames = []
    for match in _BLOCK_RE.finditer(content):
        (_, hh, mm, ss, ms, srt_cnt, lat, lon, rel_alt, abs_alt) = match.groups()
        timestamp_ms = (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1000 + int(ms)
        frames.append(
            TelemetryFrame(
                frame_idx=int(srt_cnt),
                timestamp_ms=timestamp_ms,
                latitude=float(lat),
                longitude=float(lon),
                rel_altitude_m=float(rel_alt),
                abs_altitude_m=float(abs_alt),
            )
        )

    if not frames:
        raise ValueError(
            f"No telemetry blocks parsed from {path} -- SRT format may differ from "
            "the DJI layout this parser assumes. Inspect the raw file and adjust "
            "_BLOCK_RE in src/srt_telemetry.py."
        )

    frames.sort(key=lambda f: f.frame_idx)
    return frames


def altitude_at_timestamp(frames, timestamp_ms):
    """Return the rel_altitude_m of the telemetry frame closest to timestamp_ms."""
    if not frames:
        raise ValueError("frames is empty")
    closest = min(frames, key=lambda f: abs(f.timestamp_ms - timestamp_ms))
    return closest.rel_altitude_m
