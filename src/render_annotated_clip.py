"""Render a short annotated clip (boxes + track ID + class) for validation and demo material."""

import csv
import os
from collections import defaultdict

import cv2


def load_rows_by_frame(csv_path):
    """Load a tracks CSV into a dict of frame_idx -> list of row dicts."""
    rows_by_frame = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_frame[int(row["frame"])].append(row)
    return rows_by_frame


def render_annotated_clip(video_path, csv_path, output_path, start_s, end_s):
    """Render frames from start_s to end_s of video_path, with detections from
    csv_path burned in as boxes + "track_id class" labels, to output_path (mp4)."""
    rows_by_frame = load_rows_by_frame(csv_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError(f"Could not read a valid FPS from {video_path}")
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
            x1 = int(float(row["x1"]))
            y1 = int(float(row["y1"]))
            x2 = int(float(row["x2"]))
            y2 = int(float(row["y2"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{row["track_id"]} {row["class"]}'
            cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return output_path
