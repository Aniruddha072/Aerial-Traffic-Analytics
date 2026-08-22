"""Render an L3 demo clip: boxes colored by which road arm/zone the track
entered from, visually demonstrating the turning-movement classification.
"""

import csv
import os
from collections import defaultdict

import cv2
import numpy as np

ZONE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 128, 255),
]


def _center(row):
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
    return (x1 + x2) / 2, (y1 + y2) / 2


def render_l3_zone_demo(video_path, tracks_csv, zone_labels_npy, output_path, start_s, end_s, frame_width_px=3840, frame_height_px=2160):
    labels = np.load(zone_labels_npy)
    mask_h, mask_w = labels.shape
    scale_x = mask_w / frame_width_px
    scale_y = mask_h / frame_height_px

    def zone_for(x, y):
        mx = min(max(int(x * scale_x), 0), mask_w - 1)
        my = min(max(int(y * scale_y), 0), mask_h - 1)
        z = labels[my, mx]
        return int(z) if z >= 0 else None

    by_track = defaultdict(list)
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_track[row["track_id"]].append(row)

    entry_zone = {}
    rows_by_frame = defaultdict(list)
    for tid, rows in by_track.items():
        rows.sort(key=lambda r: int(r["frame"]))
        cx, cy = _center(rows[0])
        z = zone_for(cx, cy)
        if z is not None:
            entry_zone[tid] = z
        for row in rows:
            rows_by_frame[int(row["frame"])].append(row)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        for row in rows_by_frame.get(frame_idx, []):
            tid = row["track_id"]
            if tid not in entry_zone:
                continue
            color = ZONE_COLORS[entry_zone[tid] % len(ZONE_COLORS)]
            x1, y1, x2, y2 = int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"arm{entry_zone[tid]}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
