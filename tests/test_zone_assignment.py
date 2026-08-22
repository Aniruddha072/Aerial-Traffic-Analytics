import numpy as np
import cv2
from src.zone_assignment import detect_border_arms, assign_zones_bfs


def test_detect_border_arms_finds_road_touching_border(tmp_path):
    # 100x100 mask: a plus-shape road touching all 4 borders once each
    mask = np.zeros((100, 100), dtype="uint8")
    mask[45:55, :] = 255   # horizontal arm, touches left+right
    mask[:, 45:55] = 255   # vertical arm, touches top+bottom
    mask_path = str(tmp_path / "mask.png")
    cv2.imwrite(mask_path, mask)

    arms = detect_border_arms(mask_path, min_run=5, gap_tolerance=2)

    assert len(arms) == 4  # top, bottom, left, right


def test_assign_zones_bfs_respects_road_connectivity(tmp_path):
    # an L-shaped road: horizontal arm from left, turning into a vertical
    # arm going down. A straight-line-nearest approach would misassign
    # points near the bend; BFS-through-the-mask should not.
    mask = np.zeros((60, 60), dtype="uint8")
    mask[25:35, 0:40] = 255   # horizontal segment (left arm)
    mask[25:60, 25:35] = 255  # vertical segment (bottom arm), joined at the bend
    mask_path = str(tmp_path / "mask.png")
    cv2.imwrite(mask_path, mask)

    arms = detect_border_arms(mask_path, min_run=5, gap_tolerance=2)
    assert len(arms) == 2

    labels = assign_zones_bfs(mask_path, arms)

    # a point deep in the horizontal arm should belong to the arm touching the left border
    # a point deep in the vertical arm should belong to the arm touching the bottom border
    left_arm = next(a for a in arms if a["border"] == "left")
    bottom_arm = next(a for a in arms if a["border"] == "bottom")

    assert labels[30, 5] == left_arm["id"]
    assert labels[55, 30] == bottom_arm["id"]
