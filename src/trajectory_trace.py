"""Draw each track's full path as a line on a single frame, for manual
audit of tracking continuity -- a smooth line means a stable ID; a line
that jumps means an ID switch. Complements render_annotated_clip.py's
per-frame boxes with a single-glance view of a track's whole lifetime.
"""

import colorsys
import csv
import os
from collections import defaultdict

import cv2


def load_tracks_by_id(csv_path):
    """Load a tracks CSV into {track_id: [(frame, (center_x, center_y)), ...]},
    each track's points ordered by frame."""
    tracks = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            tracks[row["track_id"]].append((int(row["frame"]), center))

    for track_id in tracks:
        tracks[track_id].sort(key=lambda p: p[0])
    return dict(tracks)


def _color_for_track(track_id):
    """Deterministic distinct-ish BGR color per track_id, via hash -> hue."""
    hue = (hash(track_id) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def render_trajectory_trace(video_path, csv_path, output_path, track_ids=None, background_frame=0):
    """Draw the full path of each track in track_ids (or all tracks, if None)
    as a polyline over a single background frame from video_path, saved to
    output_path (image file)."""
    tracks = load_tracks_by_id(csv_path)
    if track_ids is None:
        track_ids = list(tracks.keys())

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {background_frame} from {video_path}")

    for track_id in track_ids:
        points = tracks.get(track_id)
        if not points or len(points) < 2:
            continue
        color = _color_for_track(track_id)
        pts = [(int(x), int(y)) for _, (x, y) in points]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, pts[0], 5, color, -1)  # start
        cv2.drawMarker(frame, pts[-1], color, cv2.MARKER_TILTED_CROSS, 12, 2)  # end

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path
