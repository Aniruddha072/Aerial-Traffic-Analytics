"""Per-track velocity and acceleration in real-world units (L2: Object-Level
Insight). Pure math over an existing L1 tracks CSV plus telemetry -- no
model, no re-detection.
"""

import csv
from collections import defaultdict

from src.class_mapping import bbox_length_m
from src.srt_telemetry import altitude_at_timestamp

KINEMATICS_FIELDS = ["track_id", "frame", "timestamp_ms", "velocity_mps", "acceleration_mps2"]


def _center(row):
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return (x1 + x2) / 2, (y1 + y2) / 2


def compute_kinematics(csv_path, frames, frame_width_px, hfov_deg=84.0):
    """Return a list of {track_id, frame, timestamp_ms, velocity_mps,
    acceleration_mps2} for every row in csv_path, ordered by track then frame.

    velocity_mps is None for a track's first point (no prior point to diff
    against). acceleration_mps2 is None until a track has two velocity
    samples. Real-world distance uses the same ground-sample-distance math
    as the LGV/HGV split (class_mapping.bbox_length_m), applied to
    center-to-center displacement instead of bbox size.
    """
    by_track = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_track[row["track_id"]].append(row)

    results = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda r: int(r["frame"]))

        prev_center = None
        prev_ts = None
        prev_velocity = None
        prev_v_ts = None

        for row in rows:
            ts = int(row["timestamp_ms"])
            center = _center(row)

            velocity_mps = None
            if prev_center is not None:
                dt_s = (ts - prev_ts) / 1000.0
                if dt_s > 0:
                    px_dist = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
                    altitude_m = altitude_at_timestamp(frames, ts)
                    dist_m = bbox_length_m((0, 0, px_dist, 0), altitude_m, frame_width_px, hfov_deg)
                    velocity_mps = dist_m / dt_s

            acceleration_mps2 = None
            if velocity_mps is not None and prev_velocity is not None:
                dt_s = (ts - prev_v_ts) / 1000.0
                if dt_s > 0:
                    acceleration_mps2 = (velocity_mps - prev_velocity) / dt_s

            results.append(
                {
                    "track_id": track_id,
                    "frame": int(row["frame"]),
                    "timestamp_ms": ts,
                    "velocity_mps": velocity_mps,
                    "acceleration_mps2": acceleration_mps2,
                }
            )

            prev_center = center
            prev_ts = ts
            if velocity_mps is not None:
                prev_velocity = velocity_mps
                prev_v_ts = ts

    return results


def write_kinematics_csv(rows, output_csv_path):
    import os
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KINEMATICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
