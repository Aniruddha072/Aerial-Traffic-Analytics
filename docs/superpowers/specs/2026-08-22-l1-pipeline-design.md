# L1 Pipeline Design — Detection & Tracking

Date: 2026-08-22
Status: approved, pending implementation plan

## Problem

L1 of the hackathon quest requires detecting and tracking every road user in two drone videos, classified by mode (car, LGV, HGV, bus, trucks, motorcycle, pedestrian), holding stable identity through occlusion, crossing paths, and long dwell times. No annotations are provided. Max 100 points; must unlock before L2 opens.

## Constraints

- ~7 hour build window, already live, solo entry
- No ground truth → no fine-tuning; pretrained COCO weights are the whole detection budget
- Output must stay generic enough to feed L2 (object-level insight), L3 (aggregate insight), L4 (spatial grounding via SRT telemetry), L5 (network reasoning) without a rewrite
- Local RTX 4050 (6GB VRAM) is the primary compute; Kaggle T4 x2 is a reserve, not the main path

## Architecture

Single-pass script per video: `detect_track(video_path) → YOLOv8n detections (per frame) → BoT-SORT tracker → class-mapping post-process → CSV writer`.

### 1. Detection — YOLOv8n, pretrained COCO weights

Keep only classes relevant to the rubric:

| COCO class | Maps to |
|---|---|
| `person` | pedestrian |
| `car` | car |
| `motorcycle` | motorcycle |
| `bus` | bus |
| `truck` | feeds LGV/HGV split (below) |
| `bicycle` | dropped — not in the rubric's 7 classes |

Confidence threshold ~0.25 to start — drone footage means small, distant objects; better to over-detect and let tracking smooth noise than miss real vehicles. Tune after eyeballing the first annotated clip.

### 2. Tracking — BoT-SORT, not ByteTrack

Ultralytics ships both via `model.track(tracker="botsort.yaml")`. BoT-SORT adds camera-motion compensation and appearance ReID on top of ByteTrack's motion-only association. This matters because the drone's gimbal pans — the whole frame shifts under a purely motion-based tracker — and camera-motion compensation is exactly what "stable identity through occlusion and crossing paths" needs here.

### 3. LGV/HGV split (decided: heuristic, real-world-scale based)

COCO has no LGV/HGV distinction — everything from a pickup to a semi is `truck`. Post-process every `truck`-class track:

- **Primary path:** pull altitude from the SRT telemetry for that frame, convert bbox pixel dimensions to a real-world length estimate via the resulting ground-sample-distance, threshold (approx. <7m → LGV, ≥7m → HGV — exact cutoff calibrated by eyeballing footage once available).
- **Fallback path:** if SRT parsing turns out to be non-trivial (sidecar file missing, format issues), use relative bbox area vs. the median car bbox in the same frame as a cruder real-world-scale proxy.

Which path we're on depends on whether SRT ships as a sidecar file or embedded metadata — confirm once the dataset is fully downloaded.

### 4. Output schema

One CSV per video at `outputs/l1/<video_stem>_tracks.csv`:

| column | meaning |
|---|---|
| `track_id` | stable ID from the tracker |
| `frame` | frame index |
| `timestamp_ms` | for joining against SRT telemetry in L4 |
| `class` | one of the 7 rubric classes |
| `x1,y1,x2,y2` | bbox, pixel coords |
| `conf` | detection confidence |

Generic enough that L4 joins on `timestamp_ms`, L2/L3 aggregate off `track_id` + `class`.

### 5. Repo structure

```
src/
  l1_pipeline.py      # detect_track(video_path) -> writes CSV
  class_mapping.py    # COCO class -> rubric class, LGV/HGV heuristic
outputs/l1/            # gitignored, CSV outputs
```

### 6. Validation without ground truth

No labels exist to score against, so validation is:
- Render a 30-60s annotated clip (boxes + track IDs burned in) per video and eyeball it for obviously wrong detections/ID switches. This clip doubles as raw material for the required video-demo submission — not throwaway work.
- Track-quality proxy: count ID switches (a track's box jumping to a clearly different object) over the sample window.

## Out of scope for L1

- Fine-tuning any model
- Full ground-plane projection (that's L4's job)
- Anything beyond altitude for the size heuristic

## Open question carried into implementation

Whether SRT telemetry ships as a sidecar file alongside the MP4s or needs to be extracted differently — resolve once the dataset finishes downloading, before finalizing the LGV/HGV split code path.
