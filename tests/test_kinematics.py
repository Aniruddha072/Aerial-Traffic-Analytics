import csv
from src.kinematics import compute_kinematics


class FakeFrame:
    def __init__(self, timestamp_ms, rel_altitude_m):
        self.timestamp_ms = timestamp_ms
        self.rel_altitude_m = rel_altitude_m


def test_compute_kinematics_velocity_from_known_displacement(tmp_path):
    # bbox centers 100px apart, 1000ms apart, at 100m altitude, 1920px wide frame
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "1,1,1000,car,100,0,110,10,0.9\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    frames = [FakeFrame(0, 100), FakeFrame(1000, 100)]

    rows = compute_kinematics(str(csv_path), frames, frame_width_px=1920)

    assert len(rows) == 2
    assert rows[0]["velocity_mps"] is None  # first point of a track has no prior point
    assert rows[1]["velocity_mps"] is not None
    assert rows[1]["velocity_mps"] > 0


def test_compute_kinematics_acceleration_from_changing_velocity(tmp_path):
    # accelerating: displacement grows each step (50px, then 100px), constant 1000ms steps
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "1,1,1000,car,50,0,60,10,0.9\n"
        "1,2,2000,car,150,0,160,10,0.9\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    frames = [FakeFrame(0, 100), FakeFrame(1000, 100), FakeFrame(2000, 100)]

    rows = compute_kinematics(str(csv_path), frames, frame_width_px=1920)

    assert rows[0]["acceleration_mps2"] is None
    assert rows[1]["acceleration_mps2"] is None  # needs two velocity samples
    assert rows[2]["acceleration_mps2"] is not None
    assert rows[2]["acceleration_mps2"] > 0  # speeding up


def test_compute_kinematics_single_point_track_has_no_velocity(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    frames = [FakeFrame(0, 100)]

    rows = compute_kinematics(str(csv_path), frames, frame_width_px=1920)

    assert len(rows) == 1
    assert rows[0]["velocity_mps"] is None
    assert rows[0]["acceleration_mps2"] is None
