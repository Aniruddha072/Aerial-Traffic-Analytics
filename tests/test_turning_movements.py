import csv
from src.turning_movements import compute_od_matrix, time_windowed_counts

ZONES = {
    "north": [(0, 0), (100, 0), (100, 30), (0, 30)],
    "south": [(0, 70), (100, 70), (100, 100), (0, 100)],
}


def test_compute_od_matrix_counts_entry_exit_pairs(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,10,10,20,20,0.9\n"      # center (15,15) -- north
        "1,5,166,car,45,45,55,55,0.9\n"    # center (50,50) -- middle, no zone
        "1,10,333,car,80,80,90,90,0.9\n"   # center (85,85) -- south
        "2,0,0,car,10,10,20,20,0.9\n"      # center (15,15) -- north (only point, entry=exit=north)
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    result = compute_od_matrix(str(csv_path), ZONES)

    assert result["od_counts"][("north", "south")] == 1
    assert result["od_counts"][("north", "north")] == 1
    assert result["unmatched"] == 0


def test_time_windowed_counts_buckets_by_class(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "2,0,500,car,0,0,10,10,0.9\n"
        "3,0,5000,pedestrian,0,0,10,10,0.9\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    result = time_windowed_counts(str(csv_path), window_ms=1000)

    assert result[0]["car"] == 2  # both tracks 1 and 2 are unique, first bucket
    assert result[5]["pedestrian"] == 1
