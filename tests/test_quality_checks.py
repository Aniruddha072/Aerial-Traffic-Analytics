import pytest
from src.quality_checks import find_id_switches, find_oversized_boxes


def test_find_id_switches_flags_impossible_speed(tmp_path):
    # track 1 jumps 2000px between two consecutive frames (33ms apart) --
    # at 100m altitude that's a wildly impossible speed for a car
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,20,10,0.9\n"
        "1,1,33,car,2000,0,2020,10,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    class FakeFrame:
        def __init__(self, timestamp_ms, rel_altitude_m):
            self.timestamp_ms = timestamp_ms
            self.rel_altitude_m = rel_altitude_m

    frames = [FakeFrame(0, 100), FakeFrame(33, 100)]

    switches = find_id_switches(str(csv_path), frames, frame_width_px=1920)

    assert len(switches) == 1
    assert switches[0]["track_id"] == "1"
    assert switches[0]["frame"] == 1


def test_find_id_switches_flags_class_change(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,20,10,0.9\n"
        "1,1,33,pedestrian,22,0,42,10,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    switches = find_id_switches(str(csv_path), frames=[], frame_width_px=1920)

    assert len(switches) == 1
    assert switches[0]["reason"] == "class_change"


def test_find_id_switches_ignores_plausible_motion(tmp_path):
    # small, plausible frame-to-frame motion -- should not be flagged
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,20,10,0.9\n"
        "1,1,33,car,5,0,25,10,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    class FakeFrame:
        def __init__(self, timestamp_ms, rel_altitude_m):
            self.timestamp_ms = timestamp_ms
            self.rel_altitude_m = rel_altitude_m

    frames = [FakeFrame(0, 100), FakeFrame(33, 100)]

    switches = find_id_switches(str(csv_path), frames, frame_width_px=1920)
    assert switches == []


def test_find_oversized_boxes_flags_implausible_car_length(tmp_path):
    # a "car" box 500px wide at 100m altitude works out to ~47m real length --
    # no real car is that long, this is almost certainly merged detections
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,500,20,0.9\n"
        "2,0,0,car,0,0,50,20,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    class FakeFrame:
        def __init__(self, timestamp_ms, rel_altitude_m):
            self.timestamp_ms = timestamp_ms
            self.rel_altitude_m = rel_altitude_m

    frames = [FakeFrame(0, 100)]

    oversized = find_oversized_boxes(str(csv_path), frames, frame_width_px=1920)

    assert len(oversized) == 1
    assert oversized[0]["track_id"] == "1"


def test_find_oversized_boxes_ignores_normal_sizes(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,pedestrian,0,0,10,20,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    class FakeFrame:
        def __init__(self, timestamp_ms, rel_altitude_m):
            self.timestamp_ms = timestamp_ms
            self.rel_altitude_m = rel_altitude_m

    frames = [FakeFrame(0, 100)]

    oversized = find_oversized_boxes(str(csv_path), frames, frame_width_px=1920)
    assert oversized == []
