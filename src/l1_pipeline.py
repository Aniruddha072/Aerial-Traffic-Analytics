"""L1 detection + tracking pipeline: YOLOv8n + BoT-SORT -> per-track CSV."""

import csv
import os

from src.class_mapping import (
    COCO_CLASS_NAMES,
    map_coco_class,
    classify_truck,
    classify_truck_fallback,
)

CSV_FIELDS = ["track_id", "frame", "timestamp_ms", "class", "x1", "y1", "x2", "y2", "conf"]


def build_rows_from_frame(
    frame_idx,
    timestamp_ms,
    track_ids,
    class_ids,
    boxes_xyxy,
    confs,
    altitude_m=None,
    frame_width_px=None,
    hfov_deg=84.0,
):
    """Convert one frame's raw detections into rubric-class CSV rows.

    track_ids, class_ids, boxes_xyxy, confs are parallel sequences, one entry
    per detection (matches ultralytics' Results.boxes shape after .track():
    .id, .cls, .xyxy, .conf). class_ids are COCO class indices.

    Truck detections use classify_truck() when altitude_m and frame_width_px
    are both given, else classify_truck_fallback() against same-frame car
    bboxes (dropped if no cars are present to compare against).
    """
    car_bboxes_this_frame = [
        tuple(box)
        for cid, box in zip(class_ids, boxes_xyxy)
        if COCO_CLASS_NAMES.get(cid) == "car"
    ]

    rows = []
    for track_id, class_id, box, conf in zip(track_ids, class_ids, boxes_xyxy, confs):
        coco_name = COCO_CLASS_NAMES.get(class_id)
        if coco_name is None:
            continue

        if coco_name == "truck":
            if altitude_m is not None and frame_width_px is not None:
                rubric_class = classify_truck(tuple(box), altitude_m, frame_width_px, hfov_deg)
            else:
                if not car_bboxes_this_frame:
                    continue
                rubric_class = classify_truck_fallback(tuple(box), car_bboxes_this_frame)
        else:
            rubric_class = map_coco_class(coco_name)
            if rubric_class is None:
                continue

        x1, y1, x2, y2 = box
        rows.append(
            {
                "track_id": int(track_id),
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "class": rubric_class,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(conf),
            }
        )
    return rows


def write_tracks_csv(rows, output_csv_path):
    """Write rows to output_csv_path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
