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


def reclassify_truck_rows(input_csv_path, frames, frame_width_px, output_csv_path, hfov_deg=84.0):
    """Re-derive LGV/HGV classification for existing truck rows using real altitude.

    Rows already classified LGV or HGV came from classify_truck_fallback() at
    detection time (no altitude was available then). This re-runs the more
    precise altitude-based classify_truck() now that real telemetry (`frames`,
    a list of objects with .timestamp_ms and .rel_altitude_m -- as returned by
    src.srt_telemetry.parse_srt()) is available, without re-running detection.
    Non-truck rows (car/bus/motorcycle/pedestrian) pass through unchanged.
    """
    from src.srt_telemetry import altitude_at_timestamp

    with open(input_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["class"] not in ("LGV", "HGV"):
            continue
        altitude_m = altitude_at_timestamp(frames, int(row["timestamp_ms"]))
        bbox = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        row["class"] = classify_truck(bbox, altitude_m, frame_width_px, hfov_deg)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def detect_track(
    video_path,
    output_csv_path,
    srt_path=None,
    conf_threshold=0.15,
    hfov_deg=84.0,
    device="0",
    model_path="yolov8s.pt",
    imgsz=1280,
):
    """Run YOLOv8 + BoT-SORT over video_path and write per-track detections to output_csv_path.

    If srt_path is given, altitude-based LGV/HGV classification is used;
    otherwise the same-frame relative-area fallback is used. device="0" uses
    the first CUDA GPU (decisions.md 1.1: local RTX 4050 is primary compute).

    model_path/imgsz default to yolov8s.pt at 1280px (decisions.md 3) rather
    than yolov8n.pt at the default 640px -- on 4K source footage, 640px
    shrinks small objects (pedestrians, motorcycles, distant vehicles) past
    the point the model can see them. We're decode-bound not compute-bound
    (spec: GPU sits at 15-35% utilization), so the larger model/imgsz costs
    little extra wall time here.
    """
    import cv2
    from ultralytics import YOLO

    telemetry = None
    if srt_path is not None:
        from src.srt_telemetry import parse_srt
        telemetry = parse_srt(srt_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Could not read a valid FPS from {video_path}")

    model = YOLO(model_path)
    results_stream = model.track(
        source=video_path,
        tracker="botsort.yaml",
        conf=conf_threshold,
        imgsz=imgsz,
        classes=list(COCO_CLASS_NAMES.keys()),
        persist=True,
        stream=True,
        verbose=False,
        device=device,
    )

    all_rows = []
    frame_width_px = None

    for frame_idx, result in enumerate(results_stream):
        if frame_width_px is None:
            frame_width_px = result.orig_shape[1]

        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue

        timestamp_ms = int(frame_idx * (1000.0 / fps))

        altitude_m = None
        if telemetry is not None:
            from src.srt_telemetry import altitude_at_timestamp
            altitude_m = altitude_at_timestamp(telemetry, timestamp_ms)

        rows = build_rows_from_frame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            track_ids=boxes.id.tolist(),
            class_ids=boxes.cls.tolist(),
            boxes_xyxy=boxes.xyxy.tolist(),
            confs=boxes.conf.tolist(),
            altitude_m=altitude_m,
            frame_width_px=frame_width_px,
            hfov_deg=hfov_deg,
        )
        all_rows.extend(rows)

    write_tracks_csv(all_rows, output_csv_path)
    return output_csv_path
