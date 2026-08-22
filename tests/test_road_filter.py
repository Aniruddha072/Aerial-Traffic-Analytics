import csv
from src.road_filter import filter_to_road


def test_filter_to_road_keeps_only_centers_inside_polygon(tmp_path):
    # a simple square road from (0,0) to (100,100)
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]

    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,10,10,20,20,0.9\n"      # center (15,15) -- inside
        "2,0,0,car,200,200,220,220,0.8\n"  # center (210,210) -- outside
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    out_path = tmp_path / "out.csv"

    kept, dropped = filter_to_road(str(csv_path), polygon, str(out_path))

    assert kept == 1
    assert dropped == 1
    with open(out_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["track_id"] == "1"


def test_filter_to_road_respects_padding(tmp_path):
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]

    # center at (105, 50) -- just outside the raw polygon, but within 10px padding
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,100,45,110,55,0.9\n"
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    out_path = tmp_path / "out.csv"

    kept, dropped = filter_to_road(str(csv_path), polygon, str(out_path), padding_px=10)
    assert kept == 1
    assert dropped == 0

    out_path2 = tmp_path / "out2.csv"
    kept2, dropped2 = filter_to_road(str(csv_path), polygon, str(out_path2), padding_px=0)
    assert kept2 == 0
    assert dropped2 == 1
