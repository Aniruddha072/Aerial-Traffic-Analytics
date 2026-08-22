"""Render an L2 demo clip: boxes annotated with track ID, speed (km/h), and
fine-grained class where available. Reuses the L1 CSV for boxes, the
kinematics CSV for speed, and the fine-grained CSV for the sub-type label.
"""

import csv
import os
from collections import defaultdict

import cv2


def render_l2_demo_clip(video_path, tracks_csv, kinematics_csv, fine_grained_csv, output_path, start_s, end_s):
    tracks_by_frame = defaultdict(list)
    with open(tracks_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tracks_by_frame[int(row["frame"])].append(row)

    speed_by_track_frame = {}
    with open(kinematics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["velocity_mps"]:
                speed_by_track_frame[(row["track_id"], int(row["frame"]))] = float(row["velocity_mps"])

    fine_grained_by_track = {}
    with open(fine_grained_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fine_grained_by_track[row["track_id"]] = row["fine_grained_class"]

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
        for row in tracks_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = int(float(row["x1"])), int(float(row["y1"])), int(float(row["x2"])), int(float(row["y2"]))
            track_id = row["track_id"]
            has_fg = track_id in fine_grained_by_track
            color = (0, 165, 255) if has_fg else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if has_fg else 2)

            speed = speed_by_track_frame.get((track_id, frame_idx))
            label_parts = [f"#{track_id}", row["class"]]
            if speed is not None:
                label_parts.append(f"{speed * 3.6:.0f}km/h")
            cv2.putText(frame, " ".join(label_parts), (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if has_fg:
                cv2.putText(frame, fine_grained_by_track[track_id], (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
