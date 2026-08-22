"""Heuristic quality checks for tracking output, run without ground truth.

1. ID switches: a track_id whose class changes mid-life, or whose bbox
   center implies an impossible real-world speed between consecutive
   frames, most likely got linked to a different real object.
2. Oversized boxes: a box whose real-world size (via the same GSD math as
   class_mapping.classify_truck) is implausible for its class -- most
   likely two or more objects merged into one detection.

Neither check proves the failure mode; both are heuristics meant to
surface candidates for the manual spot-check requested alongside them.
"""

import csv
from collections import defaultdict

from src.class_mapping import bbox_length_m
from src.srt_telemetry import altitude_at_timestamp

# Generous real-world length ceilings per class, in meters. Anything past
# this for its class is more likely two objects merged than one real object
# of that class this large. Deliberately generous -- false positives here
# just mean "worth a manual look," not "confirmed broken."
MAX_LENGTH_M = {
    "pedestrian": 2.5,
    "motorcycle": 4.0,
    "car": 6.5,
    "bus": 18.0,
    "LGV": 7.0,
    "HGV": 16.0,
}

# A real road vehicle at an intersection/road segment isn't exceeding this.
MAX_PLAUSIBLE_SPEED_KMH = 150.0


def _load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _center(row):
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return (x1 + x2) / 2, (y1 + y2) / 2


def find_id_switches(csv_path, frames, frame_width_px, hfov_deg=84.0, max_speed_kmh=MAX_PLAUSIBLE_SPEED_KMH):
    """Return a list of {track_id, frame, reason, detail} for suspected ID switches.

    frames: telemetry frames from srt_telemetry.parse_srt(), or [] to skip
    the speed check entirely (only class-change is checked in that case).
    """
    by_track = defaultdict(list)
    for row in _load_rows(csv_path):
        by_track[row["track_id"]].append(row)

    flags = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda r: int(r["frame"]))

        for prev, cur in zip(rows, rows[1:]):
            if cur["class"] != prev["class"]:
                flags.append(
                    {
                        "track_id": track_id,
                        "frame": int(cur["frame"]),
                        "reason": "class_change",
                        "detail": f"{prev['class']} -> {cur['class']}",
                    }
                )

            if not frames:
                continue

            dt_ms = int(cur["timestamp_ms"]) - int(prev["timestamp_ms"])
            if dt_ms <= 0:
                continue

            x1p, y1p = _center(prev)
            x2p, y2p = _center(cur)
            pixel_dist = ((x2p - x1p) ** 2 + (y2p - y1p) ** 2) ** 0.5

            altitude_m = altitude_at_timestamp(frames, int(cur["timestamp_ms"]))
            length_m = bbox_length_m((0, 0, pixel_dist, 0), altitude_m, frame_width_px, hfov_deg)
            speed_kmh = (length_m / (dt_ms / 1000.0)) * 3.6

            if speed_kmh > max_speed_kmh:
                flags.append(
                    {
                        "track_id": track_id,
                        "frame": int(cur["frame"]),
                        "reason": "impossible_speed",
                        "detail": f"{speed_kmh:.0f} km/h implied",
                    }
                )

    return flags


def find_oversized_boxes(csv_path, frames, frame_width_px, hfov_deg=84.0, max_length_m=None):
    """Return a list of {track_id, frame, class, length_m} for boxes whose
    real-world size exceeds MAX_LENGTH_M for their class -- candidates for
    "multiple objects in one box"."""
    thresholds = max_length_m or MAX_LENGTH_M
    if not frames:
        return []

    flags = []
    for row in _load_rows(csv_path):
        limit = thresholds.get(row["class"])
        if limit is None:
            continue

        bbox = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        altitude_m = altitude_at_timestamp(frames, int(row["timestamp_ms"]))
        length_m = bbox_length_m(bbox, altitude_m, frame_width_px, hfov_deg)

        if length_m > limit:
            flags.append(
                {
                    "track_id": row["track_id"],
                    "frame": int(row["frame"]),
                    "class": row["class"],
                    "length_m": round(length_m, 1),
                }
            )

    return flags
