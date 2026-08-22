"""Aggregate-level summaries over L1/L2 output (L3 prep: Aggregate Insight).

Pure data analysis over CSVs already on disk -- no model, no video decode.
Built ahead of L3's exact brief unlocking, using the well-known traffic
aggregate categories (counts, speed profiles, density/flow) as a safe bet.
"""

import csv
from collections import defaultdict


def class_counts(csv_path):
    """Return {class: unique_track_count} -- counts distinct tracked objects,
    not rows (a track spans many rows, one per frame it's visible in)."""
    tracks_by_class = defaultdict(set)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tracks_by_class[row["class"]].add(row["track_id"])
    return {cls: len(ids) for cls, ids in tracks_by_class.items()}


def speed_profile_summary(kinematics_csv_path):
    """Return speed distribution stats (m/s) from a kinematics CSV
    (src.kinematics.compute_kinematics output): n, mean, percentiles, max."""
    speeds = []
    with open(kinematics_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("velocity_mps")
            if v:
                speeds.append(float(v))

    speeds.sort()
    n = len(speeds)
    if n == 0:
        return {"n": 0}

    def pct(p):
        return speeds[min(int(n * p / 100), n - 1)]

    return {
        "n": n,
        "mean_mps": sum(speeds) / n,
        "p10_mps": pct(10),
        "p25_mps": pct(25),
        "p50_mps": pct(50),
        "p75_mps": pct(75),
        "p90_mps": pct(90),
        "p95_mps": pct(95),
        "max_mps": speeds[-1],
    }
