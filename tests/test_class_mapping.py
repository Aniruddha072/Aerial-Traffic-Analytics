import math
import pytest
from src.class_mapping import (
    map_coco_class,
    bbox_length_m,
    classify_truck,
    classify_truck_fallback,
)


def test_map_coco_class_known_classes():
    assert map_coco_class("person") == "pedestrian"
    assert map_coco_class("car") == "car"
    assert map_coco_class("motorcycle") == "motorcycle"
    assert map_coco_class("bus") == "bus"


def test_map_coco_class_drops_bicycle():
    assert map_coco_class("bicycle") is None


def test_map_coco_class_rejects_truck():
    with pytest.raises(ValueError):
        map_coco_class("truck")


def test_bbox_length_m_known_geometry():
    bbox = (0, 0, 100, 40)
    length = bbox_length_m(bbox, altitude_m=100, frame_width_px=1920, hfov_deg=84)
    expected_gsd = (2 * 100 * math.tan(math.radians(42))) / 1920
    assert length == pytest.approx(100 * expected_gsd, rel=1e-6)


def test_classify_truck_lgv_below_threshold():
    result = classify_truck((0, 0, 20, 10), altitude_m=100, frame_width_px=1920, hfov_deg=84)
    assert result == "LGV"


def test_classify_truck_hgv_above_threshold():
    result = classify_truck((0, 0, 400, 100), altitude_m=100, frame_width_px=1920, hfov_deg=84)
    assert result == "HGV"


def test_classify_truck_fallback_hgv_large_relative_area():
    car_bboxes = [(0, 0, 50, 25), (0, 0, 48, 24), (0, 0, 52, 26)]
    truck_bbox = (0, 0, 200, 20)
    assert classify_truck_fallback(truck_bbox, car_bboxes) == "HGV"


def test_classify_truck_fallback_lgv_small_relative_area():
    car_bboxes = [(0, 0, 50, 25), (0, 0, 48, 24), (0, 0, 52, 26)]
    truck_bbox = (0, 0, 55, 28)
    assert classify_truck_fallback(truck_bbox, car_bboxes) == "LGV"


def test_classify_truck_fallback_raises_without_cars():
    with pytest.raises(ValueError):
        classify_truck_fallback((0, 0, 100, 50), [])
