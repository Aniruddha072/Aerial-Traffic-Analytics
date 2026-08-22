import numpy as np
import cv2
from src.road_filter import filter_to_road_mask


def test_filter_to_road_mask_keeps_only_white_region(tmp_path):
    # 100x100 black mask with a white 40x40 square at (30,30)-(70,70)
    mask = np.zeros((100, 100), dtype="uint8")
    mask[30:70, 30:70] = 255
    mask_path = str(tmp_path / "mask.png")
    cv2.imwrite(mask_path, mask)

    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,40,40,50,50,0.9\n"    # center (45,45) -- inside white square
        "2,0,0,car,0,0,10,10,0.8\n"      # center (5,5) -- black region
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    out_path = tmp_path / "out.csv"

    kept, dropped = filter_to_road_mask(str(csv_path), mask_path, str(out_path), frame_width_px=100, frame_height_px=100)

    assert kept == 1
    assert dropped == 1


def test_filter_to_road_mask_scales_to_frame_size(tmp_path):
    # mask is 100x100 but the real frame is 200x200 -- coordinates must scale
    mask = np.zeros((100, 100), dtype="uint8")
    mask[0:50, 0:50] = 255  # white in the top-left quadrant of the mask
    mask_path = str(tmp_path / "mask.png")
    cv2.imwrite(mask_path, mask)

    # in a 200x200 frame, the white region maps to (0,0)-(100,100)
    csv_content = (
        "track_id,frame,timestamp_ms,class,x1,y1,x2,y2,conf\n"
        "1,0,0,car,10,10,30,30,0.9\n"     # center (20,20) -- inside scaled white region
        "2,0,0,car,150,150,170,170,0.8\n" # center (160,160) -- outside
    )
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    out_path = tmp_path / "out.csv"

    kept, dropped = filter_to_road_mask(str(csv_path), mask_path, str(out_path), frame_width_px=200, frame_height_px=200)

    assert kept == 1
    assert dropped == 1
