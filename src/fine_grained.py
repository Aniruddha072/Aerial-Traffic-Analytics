"""Fine-grained vehicle classification (L2: Object-Level Insight).

Strategy: pick one representative detection per track (the largest bbox --
closest/clearest view of that object), crop it from the source video, and
run CLIP zero-shot classification against a per-top-level-class label set.
One CLIP call per track, not per frame -- this is orders of magnitude
cheaper than re-running detection.
"""

import csv
from collections import defaultdict

# Candidate fine-grained labels per top-level rubric class. Pedestrians and
# motorcycles aren't split further -- L2 asks for "vehicle classification",
# and there's no meaningful fine-grained split for a pedestrian bbox.
#
# These are top-down-silhouette categories, not side-profile ones (sedan vs
# hatchback vs SUV look nearly identical from directly overhead -- a drone's
# view -- since that distinction lives in grille shape and roofline, cues a
# nadir crop doesn't have). An open truck bed, a van's boxy rectangle, and a
# compact car's smaller rounded footprint are all visible from above, so
# this set is chosen for what a top-down crop can actually discriminate.
FINE_GRAINED_LABELS = {
    "car": ["compact car", "large sedan or SUV", "van or minivan"],
    "bus": ["single-deck bus", "articulated bus with a bend in the middle"],
    "LGV": ["enclosed van", "pickup truck with an open bed"],
    "HGV": ["truck with an open or flatbed cargo area", "truck with a closed box or tanker body"],
}


def select_representative_frames(csv_path, exclude_classes=None):
    """Return {track_id: {"frame": int, "class": str, "bbox": (x1,y1,x2,y2)}}
    picking the largest-area detection per track as its representative frame.
    """
    exclude_classes = exclude_classes or set()
    best = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["class"] in exclude_classes:
                continue
            x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            area = abs(x2 - x1) * abs(y2 - y1)
            track_id = row["track_id"]
            if track_id not in best or area > best[track_id]["_area"]:
                best[track_id] = {
                    "frame": int(row["frame"]),
                    "class": row["class"],
                    "bbox": (x1, y1, x2, y2),
                    "_area": area,
                }
    for v in best.values():
        del v["_area"]
    return best


def classify_tracks(video_path, csv_path, output_csv_path, device="cuda", max_tracks=None):
    """For every track whose top-level class has a fine-grained label set
    (see FINE_GRAINED_LABELS), crop its largest-bbox frame from video_path
    and classify it via CLIP zero-shot against just that class's labels.
    Writes {track_id, class, fine_grained_class, clip_confidence} to
    output_csv_path. Tracks whose class has no fine-grained set (pedestrian,
    motorcycle) are skipped entirely.

    max_tracks caps how many tracks get classified, evenly sampled across
    the video's frame range (not just the first N chronologically) so the
    sample stays representative -- each video seek costs ~2s (random access
    into a compressed 4K stream), which makes classifying every single track
    impractical within a tight time budget. None processes everything.
    """
    import os
    import cv2
    import torch
    import open_clip

    reps = select_representative_frames(csv_path, exclude_classes=None)
    reps = {tid: r for tid, r in reps.items() if r["class"] in FINE_GRAINED_LABELS}

    if max_tracks is not None and len(reps) > max_tracks:
        ordered_ids = sorted(reps.keys(), key=lambda tid: reps[tid]["frame"])
        step = len(ordered_ids) / max_tracks
        sampled_ids = {ordered_ids[int(i * step)] for i in range(max_tracks)}
        reps = {tid: r for tid, r in reps.items() if tid in sampled_ids}

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # Pre-encode text embeddings once per class's label set.
    text_features_by_class = {}
    with torch.no_grad():
        for cls, labels in FINE_GRAINED_LABELS.items():
            prompts = [f"a drone photo looking straight down at a {label} on a road" for label in labels]
            tokens = tokenizer(prompts).to(device)
            feats = model.encode_text(tokens)
            feats /= feats.norm(dim=-1, keepdim=True)
            text_features_by_class[cls] = feats

    # Process in frame order to keep video seeks roughly forward-moving.
    ordered = sorted(reps.items(), key=lambda kv: kv[1]["frame"])
    cap = cv2.VideoCapture(video_path)

    results = []
    with torch.no_grad():
        for track_id, rep in ordered:
            cap.set(cv2.CAP_PROP_POS_FRAMES, rep["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = [int(v) for v in rep["bbox"]]
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            image = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            text_features = text_features_by_class[rep["class"]]
            similarity = (image_features @ text_features.T).squeeze(0)
            best_idx = similarity.argmax().item()

            results.append(
                {
                    "track_id": track_id,
                    "class": rep["class"],
                    "fine_grained_class": FINE_GRAINED_LABELS[rep["class"]][best_idx],
                    "clip_confidence": float(similarity[best_idx].item()),
                }
            )

    cap.release()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "class", "fine_grained_class", "clip_confidence"])
        writer.writeheader()
        writer.writerows(results)

    return output_csv_path
