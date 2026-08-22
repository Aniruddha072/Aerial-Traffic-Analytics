import os
import cv2
import numpy as np
from src.render_annotated_clip import load_rows_by_frame, render_annotated_clip


def test_load_rows_by_frame_groups_by_frame(tmp_path):
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,1,2,3,4,0.9\n"
        "2,0,0,pedestrian,5,6,7,8,0.8\n"
        "1,1,33,car,2,3,4,5,0.91\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    rows_by_frame = load_rows_by_frame(str(csv_path))

    assert len(rows_by_frame[0]) == 2
    assert len(rows_by_frame[1]) == 1
    assert rows_by_frame[0][0]["class"] == "car"
    assert rows_by_frame[0][1]["class"] == "pedestrian"


def test_render_annotated_clip_produces_output_video(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (64, 48))
    for _ in range(20):
        frame = (np.ones((48, 64, 3), dtype="uint8") * 127)
        writer.write(frame)
    writer.release()

    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,10,10,30,20,0.9\n"
    )
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    output_path = str(tmp_path / "out" / "clip.mp4")
    render_annotated_clip(video_path, str(csv_path), output_path, start_s=0, end_s=1)

    assert os.path.exists(output_path)
    out_cap = cv2.VideoCapture(output_path)
    frame_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_cap.release()
    assert frame_count == 10
