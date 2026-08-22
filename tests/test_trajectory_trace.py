import os
import cv2
import numpy as np
from src.trajectory_trace import load_tracks_by_id, render_trajectory_trace


def test_load_tracks_by_id_groups_and_orders_by_frame(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,5,166,car,10,10,20,20,0.7\n"
        "1,2,66,car,0,0,10,10,0.9\n"
        "2,0,0,pedestrian,5,5,15,15,0.8\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    tracks = load_tracks_by_id(str(csv_path))

    assert set(tracks.keys()) == {"1", "2"}
    # track 1's points should be ordered by frame (2 before 5), not CSV row order
    assert [p[0] for p in tracks["1"]] == [2, 5]
    centers = tracks["1"]
    assert centers[0][1] == (5.0, 5.0)  # center of (0,0,10,10)
    assert centers[1][1] == (15.0, 15.0)  # center of (10,10,20,20)


def test_render_trajectory_trace_draws_on_background_frame(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (64, 48))
    for _ in range(5):
        writer.write(np.ones((48, 64, 3), dtype="uint8") * 100)
    writer.release()

    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "1,1,33,car,10,10,20,20,0.9\n"
        "1,2,66,car,20,20,30,30,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    output_path = str(tmp_path / "out" / "trace.jpg")
    render_trajectory_trace(video_path, str(csv_path), output_path, track_ids=["1"], background_frame=0)

    assert os.path.exists(output_path)
    img = cv2.imread(output_path)
    assert img is not None
    assert img.shape[:2] == (48, 64)


def test_render_trajectory_trace_all_tracks_when_none_specified(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (64, 48))
    writer.write(np.ones((48, 64, 3), dtype="uint8") * 100)
    writer.release()

    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,0,0,10,10,0.9\n"
        "2,0,0,pedestrian,20,20,30,30,0.8\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    output_path = str(tmp_path / "out" / "trace_all.jpg")
    render_trajectory_trace(video_path, str(csv_path), output_path, track_ids=None, background_frame=0)

    assert os.path.exists(output_path)
