from src.aggregate_insight import class_counts, speed_profile_summary


def test_class_counts_counts_unique_tracks_not_rows(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "1,1,33,car,1,1,11,11,0.9\n"   # same track, second row -- shouldn't double count
        "2,0,0,pedestrian,5,5,15,15,0.8\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    counts = class_counts(str(csv_path))

    assert counts == {"car": 1, "pedestrian": 1}


def test_speed_profile_summary_computes_percentiles(tmp_path):
    csv_content = "track_id,frame,timestamp_ms,velocity_mps,acceleration_mps2\n"
    for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        csv_content += f"1,{i},{i*33},{v},\n"
    csv_path = tmp_path / "kin.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    summary = speed_profile_summary(str(csv_path))

    assert summary["n"] == 5
    assert summary["mean_mps"] == 3.0
    assert summary["p50_mps"] == 3.0
    assert summary["max_mps"] == 5.0
