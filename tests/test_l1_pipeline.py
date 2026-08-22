import csv
from src.l1_pipeline import build_rows_from_frame, write_tracks_csv, CSV_FIELDS


def test_build_rows_from_frame_maps_known_classes():
    rows = build_rows_from_frame(
        frame_idx=10,
        timestamp_ms=333,
        track_ids=[1, 2],
        class_ids=[0, 2],  # person, car
        boxes_xyxy=[[10, 10, 20, 20], [30, 30, 60, 50]],
        confs=[0.9, 0.8],
    )
    assert len(rows) == 2
    assert rows[0] == {
        "track_id": 1, "frame": 10, "timestamp_ms": 333, "class": "pedestrian",
        "x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 20.0, "conf": 0.9,
    }
    assert rows[1]["class"] == "car"


def test_build_rows_from_frame_drops_bicycle():
    rows = build_rows_from_frame(
        frame_idx=0, timestamp_ms=0,
        track_ids=[1], class_ids=[1], boxes_xyxy=[[0, 0, 10, 10]], confs=[0.5],
    )
    assert rows == []


def test_build_rows_from_frame_truck_uses_altitude_path():
    rows = build_rows_from_frame(
        frame_idx=0, timestamp_ms=0,
        track_ids=[1], class_ids=[7], boxes_xyxy=[[0, 0, 20, 10]], confs=[0.7],
        altitude_m=100, frame_width_px=1920, hfov_deg=84.0,
    )
    assert len(rows) == 1
    assert rows[0]["class"] == "LGV"


def test_build_rows_from_frame_truck_uses_fallback_without_altitude():
    rows = build_rows_from_frame(
        frame_idx=0, timestamp_ms=0,
        track_ids=[1, 2],
        class_ids=[7, 2],  # truck, car
        boxes_xyxy=[[0, 0, 200, 20], [0, 0, 50, 25]],
        confs=[0.7, 0.6],
    )
    truck_rows = [r for r in rows if r["track_id"] == 1]
    assert len(truck_rows) == 1
    assert truck_rows[0]["class"] == "HGV"


def test_build_rows_from_frame_truck_dropped_without_altitude_or_cars():
    rows = build_rows_from_frame(
        frame_idx=0, timestamp_ms=0,
        track_ids=[1], class_ids=[7], boxes_xyxy=[[0, 0, 200, 20]], confs=[0.7],
    )
    assert rows == []


def test_write_tracks_csv_round_trip(tmp_path):
    rows = [
        {"track_id": 1, "frame": 0, "timestamp_ms": 0, "class": "car",
         "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "conf": 0.95},
    ]
    out_path = tmp_path / "out" / "video_tracks.csv"
    write_tracks_csv(rows, str(out_path))

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDS
        read_rows = list(reader)
    assert len(read_rows) == 1
    assert read_rows[0]["class"] == "car"
    assert read_rows[0]["track_id"] == "1"
