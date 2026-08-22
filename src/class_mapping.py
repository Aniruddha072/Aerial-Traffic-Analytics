"""COCO class -> hackathon rubric class mapping, plus LGV/HGV split heuristic."""

import math

COCO_CLASS_NAMES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COCO_TO_RUBRIC = {
    "person": "pedestrian",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    # "truck" is intentionally absent -- route it through classify_truck()
    # instead, since the LGV/HGV split needs bbox/altitude context this
    # dict doesn't have.
}

# Real-world vehicle length threshold, in meters, separating LGV from HGV.
# Calibrate against real footage once available (spec 3, open question).
HGV_LENGTH_THRESHOLD_M = 7.0


def map_coco_class(coco_class_name):
    """Map a COCO class name to a rubric class name, or None if it should be dropped."""
    if coco_class_name == "truck":
        raise ValueError("truck must be classified via classify_truck(), not map_coco_class()")
    return COCO_TO_RUBRIC.get(coco_class_name)


def bbox_length_m(bbox_xyxy, altitude_m, frame_width_px, hfov_deg):
    """Estimate a bbox's longer side in real-world meters via ground sample distance.

    GSD (meters/pixel) = 2 * altitude_m * tan(hfov_deg/2) / frame_width_px
    """
    x1, y1, x2, y2 = bbox_xyxy
    longer_px = max(abs(x2 - x1), abs(y2 - y1))
    gsd = (2 * altitude_m * math.tan(math.radians(hfov_deg / 2))) / frame_width_px
    return longer_px * gsd


def classify_truck(bbox_xyxy, altitude_m, frame_width_px, hfov_deg, threshold_m=HGV_LENGTH_THRESHOLD_M):
    """Classify a 'truck'-class detection as LGV or HGV using real-world scale (spec 3, primary path)."""
    length_m = bbox_length_m(bbox_xyxy, altitude_m, frame_width_px, hfov_deg)
    return "LGV" if length_m < threshold_m else "HGV"


def classify_truck_fallback(bbox_xyxy, car_bboxes_same_frame):
    """Fallback LGV/HGV split when altitude telemetry isn't available (spec 3, fallback path).

    More than 2.5x the median same-frame car bbox area -> HGV, else LGV.
    """
    if not car_bboxes_same_frame:
        raise ValueError("classify_truck_fallback requires at least one car bbox in the same frame")

    def area(b):
        x1, y1, x2, y2 = b
        return abs(x2 - x1) * abs(y2 - y1)

    areas = sorted(area(b) for b in car_bboxes_same_frame)
    median_car_area = areas[len(areas) // 2]
    truck_area = area(bbox_xyxy)
    return "HGV" if truck_area > 2.5 * median_car_area else "LGV"
