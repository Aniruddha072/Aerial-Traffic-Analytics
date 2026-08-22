"""SAHI-sliced detection + BoT-SORT tracking, restricted to given ROIs.

Motivated by organizer guidance to explore SOD (small object detection) and
SAHI. Full-frame SAHI found 4-5x more detections than single-pass YOLO but at
~15-25x the per-frame cost; restricting slicing to the two actual road
regions in Multi_Road_Merged.MP4 keeps ~90% of the detection gain at roughly
half the cost (see docs/decisions.md Decision 5).

ReID is disabled for the BoT-SORT tracker here: it requires a forward-pass
hook into Ultralytics' own predict() loop that SAHI's separate per-tile
inference doesn't provide. Multi_Road_Merged's camera is confirmed static
(telemetry: gimbal yaw/pitch/roll essentially constant across the whole
video), so motion-only tracking is a much smaller downgrade here than it
would be on a panning camera -- this is not a general substitute for BoT-SORT
with ReID, just a fit for this specific static-camera video.
"""

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def build_tracker(device="0"):
    """Build a BoT-SORT tracker with ReID disabled (see module docstring)."""
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
    cfg.device = device
    cfg.with_reid = False
    return BOTSORT(args=cfg)


def sahi_detect_rois(frame, sahi_model, rois, slice_size=640, overlap_ratio=0.2):
    """Run SAHI-sliced detection restricted to each ROI in rois (list of
    (x1,y1,x2,y2) full-frame pixel rects), merged into one Boxes object in
    full-frame coordinates.
    """
    from sahi.predict import get_sliced_prediction

    orig_shape = frame.shape[:2]  # (height, width)
    rows = []
    for x1, y1, x2, y2 in rois:
        crop = frame[y1:y2, x1:x2]
        result = get_sliced_prediction(
            crop,
            sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        for pred in result.object_prediction_list:
            bx1, by1, bx2, by2 = pred.bbox.to_xyxy()
            rows.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, pred.score.value, pred.category.id])

    if not rows:
        return Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape)
    return Boxes(np.array(rows, dtype=np.float32), orig_shape)


def sahi_track_video(
    video_path,
    output_csv_path,
    rois,
    srt_path=None,
    conf_threshold=0.15,
    frame_width_px=3840,
    hfov_deg=84.0,
    device="0",
    max_frames=None,
):
    """Run ROI-restricted SAHI detection + motion-only BoT-SORT tracking over
    video_path, writing per-track detections to output_csv_path in the same
    schema as l1_pipeline.detect_track(). max_frames caps processing for
    quick segment tests; None processes the whole video.
    """
    import cv2
    from sahi import AutoDetectionModel
    from ultralytics import YOLO

    from src.l1_pipeline import build_rows_from_frame, write_tracks_csv

    telemetry = None
    if srt_path is not None:
        from src.srt_telemetry import parse_srt
        telemetry = parse_srt(srt_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError(f"Could not read a valid FPS from {video_path}")

    model = YOLO("yolov8s.pt")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=model,
        confidence_threshold=conf_threshold,
        device=f"cuda:{device}" if device != "cpu" else "cpu",
    )
    tracker = build_tracker(device=device)

    all_rows = []
    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        boxes = sahi_detect_rois(frame, sahi_model, rois)
        tracked = tracker.update(boxes, img=frame)  # rows: x1,y1,x2,y2,track_id,score,cls,idx

        timestamp_ms = int(frame_idx * (1000.0 / fps))
        altitude_m = None
        if telemetry is not None:
            from src.srt_telemetry import altitude_at_timestamp
            altitude_m = altitude_at_timestamp(telemetry, timestamp_ms)

        if len(tracked):
            rows = build_rows_from_frame(
                frame_idx=frame_idx,
                timestamp_ms=timestamp_ms,
                track_ids=tracked[:, 4].tolist(),
                class_ids=[int(c) for c in tracked[:, 6].tolist()],
                boxes_xyxy=tracked[:, 0:4].tolist(),
                confs=tracked[:, 5].tolist(),
                altitude_m=altitude_m,
                frame_width_px=frame_width_px,
                hfov_deg=hfov_deg,
            )
            all_rows.extend(rows)

        frame_idx += 1

    cap.release()
    write_tracks_csv(all_rows, output_csv_path)
    return output_csv_path
