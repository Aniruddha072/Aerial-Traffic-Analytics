"""Origin-destination / turning-movement analysis and time-windowed flow
counts (L3 prep: Aggregate Insight). Pure geometry and counting over an
existing L1 tracks CSV -- no model, no video decode.
"""

import csv
from collections import Counter, defaultdict

from shapely.geometry import Point, Polygon


def _center(row):
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return (x1 + x2) / 2, (y1 + y2) / 2


def _zone_for_point(point, zone_polygons):
    for name, poly in zone_polygons.items():
        if poly.contains(Point(point)):
            return name
    return None


def compute_od_matrix(csv_path, zones):
    """For every track, determine which named zone (from `zones`, a dict of
    {name: [(x,y), ...]} polygons) it first appears in (entry) and last
    appears in (exit). Returns {"od_counts": Counter of (entry, exit) pairs,
    "unmatched": count of tracks whose entry or exit point falls outside
    every zone}.
    """
    zone_polygons = {name: Polygon(pts) for name, pts in zones.items()}

    by_track = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_track[row["track_id"]].append(row)

    od_counts = Counter()
    unmatched = 0
    for rows in by_track.values():
        rows.sort(key=lambda r: int(r["frame"]))
        entry_zone = _zone_for_point(_center(rows[0]), zone_polygons)
        exit_zone = _zone_for_point(_center(rows[-1]), zone_polygons)
        if entry_zone is None or exit_zone is None:
            unmatched += 1
            continue
        od_counts[(entry_zone, exit_zone)] += 1

    return {"od_counts": od_counts, "unmatched": unmatched}


def compute_od_matrix_from_labels(csv_path, zone_labels, mask_width, mask_height, frame_width_px, frame_height_px):
    """Same idea as compute_od_matrix, but zones come from a pixel-label
    array (src.zone_assignment.assign_zones_bfs output) instead of hand-drawn
    polygons -- geodesic nearest-arm assignment through the actual road
    shape, not straight-line point-in-polygon. Label -1 means "not on the
    road region at all" (counted as unmatched).
    """
    scale_x = mask_width / frame_width_px
    scale_y = mask_height / frame_height_px

    def zone_for(point):
        x, y = point
        mx = min(max(int(x * scale_x), 0), mask_width - 1)
        my = min(max(int(y * scale_y), 0), mask_height - 1)
        label = zone_labels[my, mx]
        return int(label) if label >= 0 else None

    by_track = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_track[row["track_id"]].append(row)

    od_counts = Counter()
    unmatched = 0
    for rows in by_track.values():
        rows.sort(key=lambda r: int(r["frame"]))
        entry_zone = zone_for(_center(rows[0]))
        exit_zone = zone_for(_center(rows[-1]))
        if entry_zone is None or exit_zone is None:
            unmatched += 1
            continue
        od_counts[(entry_zone, exit_zone)] += 1

    return {"od_counts": od_counts, "unmatched": unmatched}


def time_windowed_counts(csv_path, window_ms):
    """Bucket unique tracks into time windows of window_ms, keyed by each
    track's first-seen timestamp. Returns {window_index: {class: count}}."""
    first_seen = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["track_id"]
            ts = int(row["timestamp_ms"])
            if tid not in first_seen or ts < first_seen[tid][0]:
                first_seen[tid] = (ts, row["class"])

    buckets = defaultdict(lambda: defaultdict(int))
    for ts, cls in first_seen.values():
        window = ts // window_ms
        buckets[window][cls] += 1

    return {w: dict(counts) for w, counts in buckets.items()}
