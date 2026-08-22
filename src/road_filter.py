"""Post-process a tracks CSV to drop detections outside the actual road area.

Pure post-processing on already-computed detections -- no re-detection, no
model calls. Motivated by false positives observed on building rooftops in
the Intersection_Merged trajectory trace (docs/decisions.md Decision 6).
"""

import csv
import os

import cv2
from shapely.geometry import Point, Polygon


def filter_to_road(input_csv_path, polygon_points, output_csv_path, padding_px=15):
    """Keep only rows whose bbox center falls within polygon_points, expanded
    by padding_px to avoid clipping legitimate roadside pedestrians/vehicles
    right at the edge.

    polygon_points is either a list of (x,y) pixel coordinates for a single
    polygon, or an already-built shapely geometry (e.g. a unary_union of
    several simpler per-arm polygons -- easier to trace accurately than one
    complex hand-traced outline).

    Returns (kept_count, dropped_count).
    """
    polygon = polygon_points if hasattr(polygon_points, "buffer") else Polygon(polygon_points)
    if padding_px:
        polygon = polygon.buffer(padding_px)

    with open(input_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept_rows = []
    dropped = 0
    for row in rows:
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        if polygon.contains(center):
            kept_rows.append(row)
        else:
            dropped += 1

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return len(kept_rows), dropped


def filter_to_road_mask(
    input_csv_path,
    mask_path,
    output_csv_path,
    frame_width_px,
    frame_height_px,
    white_threshold=127,
):
    """Keep only rows whose bbox center falls on the white (road) region of a
    binary road mask image (e.g. a black/white segmentation from an external
    source). The mask may be a different resolution than the video frame --
    coordinates are scaled to match. Returns (kept_count, dropped_count).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image at {mask_path}")
    mask_h, mask_w = mask.shape
    scale_x = mask_w / frame_width_px
    scale_y = mask_h / frame_height_px

    with open(input_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept_rows = []
    dropped = 0
    for row in rows:
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        mx = min(max(int(cx * scale_x), 0), mask_w - 1)
        my = min(max(int(cy * scale_y), 0), mask_h - 1)
        if mask[my, mx] >= white_threshold:
            kept_rows.append(row)
        else:
            dropped += 1

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return len(kept_rows), dropped
