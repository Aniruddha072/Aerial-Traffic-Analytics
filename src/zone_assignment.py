"""Auto-detect road 'arms' from a binary road mask and assign every road
pixel to its nearest arm via BFS through the mask itself (respects the
road's actual curvature, unlike straight-line nearest-zone assignment).
Used for the OD / turning-movement matrix (L3: Aggregate Insight).
"""

from collections import deque

import cv2
import numpy as np


def detect_border_arms(mask_path, min_run=20, gap_tolerance=25):
    """Find each place the road touches the frame border, walking the full
    perimeter as one loop so a run spanning a corner isn't split in two.
    Returns a list of {"id": int, "border": str, "points": [(row,col), ...]}.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = mask >= 127
    h, w = mask_bin.shape

    perimeter = []
    borders = []
    for x in range(w):
        perimeter.append((0, x))
        borders.append("top")
    for y in range(h):
        perimeter.append((y, w - 1))
        borders.append("right")
    for x in range(w - 1, -1, -1):
        perimeter.append((h - 1, x))
        borders.append("bottom")
    for y in range(h - 1, -1, -1):
        perimeter.append((y, 0))
        borders.append("left")

    is_road = [mask_bin[p] for p in perimeter]
    n = len(is_road)
    start = next((k for k in range(n) if not is_road[k]), 0)
    order = list(range(start, n)) + list(range(0, start))

    runs = []
    cur_points, cur_borders = [], []
    gap = 0
    for idx in order:
        if is_road[idx]:
            cur_points.append(perimeter[idx])
            cur_borders.append(borders[idx])
            gap = 0
        else:
            gap += 1
            if gap > gap_tolerance and cur_points:
                if len(cur_points) >= min_run:
                    runs.append((cur_points, cur_borders))
                cur_points, cur_borders = [], []
    if len(cur_points) >= min_run:
        runs.append((cur_points, cur_borders))

    arms = []
    for i, (points, borders_) in enumerate(runs):
        # majority border label for this run (a run can span a corner)
        border = max(set(borders_), key=borders_.count)
        arms.append({"id": i, "border": border, "points": points})
    return arms


def assign_zones_bfs(mask_path, arms):
    """Multi-source BFS from each arm's border pixels, through the road
    mask. Returns an (h, w) int array where each road pixel holds the id of
    the arm reached first (geodesic nearest, following actual road shape);
    non-road pixels are -1."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = mask >= 127
    h, w = mask_bin.shape

    labels = np.full((h, w), -1, dtype=np.int32)
    q = deque()
    for arm in arms:
        for p in arm["points"]:
            if mask_bin[p] and labels[p] == -1:
                labels[p] = arm["id"]
                q.append(p)

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask_bin[nr, nc] and labels[nr, nc] == -1:
                labels[nr, nc] = labels[r, c]
                q.append((nr, nc))

    return labels
