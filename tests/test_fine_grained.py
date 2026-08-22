import csv
from src.fine_grained import select_representative_frames


def test_select_representative_frames_picks_largest_box_per_track(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"       # area 100
        "1,5,166,car,0,0,50,50,0.9\n"     # area 2500 -- largest for track 1
        "1,10,333,car,0,0,20,20,0.9\n"    # area 400
        "2,0,0,pedestrian,0,0,5,5,0.8\n"  # only point for track 2
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    reps = select_representative_frames(str(csv_path))

    assert reps["1"]["frame"] == 5
    assert reps["1"]["class"] == "car"
    assert reps["1"]["bbox"] == (0.0, 0.0, 50.0, 50.0)
    assert reps["2"]["frame"] == 0


def test_select_representative_frames_can_exclude_classes(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "2,0,0,pedestrian,0,0,5,5,0.8\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    reps = select_representative_frames(str(csv_path), exclude_classes={"pedestrian"})

    assert "1" in reps
    assert "2" not in reps
