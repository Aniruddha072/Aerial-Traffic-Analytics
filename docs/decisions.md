# Decisions

Running log of non-obvious choices and why we made them. Dated records — don't rewrite past entries when something changes, add a new one that supersedes it.

## Decision 1 — Environment (pre-event, 2026-08-21/22)

**1.1 — Local GPU + Kaggle as reserve, not primary.** RTX 4050 Laptop (6GB VRAM) handles YOLOv8n fine locally. Kaggle T4 x2 (30 GPU-hrs/week) is kept as a fallback if local VRAM becomes a bottleneck (e.g. heavier models, longer batch runs), not the main compute path — avoids depending on a shared/queued resource for the core pipeline.

**1.2 — Gemini 3.6 Flash, not 2.5 Flash.** The prerequisite guide's own snippet uses `gemini-2.5-flash`, which now 404s ("no longer available to new users"). Switched to `gemini-3.6-flash`, confirmed working. If this drifts again, check aistudio.google.com for the current Flash model name before assuming the code is wrong.

**1.3 — Claude Code Pro, no OpenRouter.** The guide's coding-setup requirement is satisfied by the Claude Code Pro subscription already in place — no need for the guide's OpenRouter/free-model fallback path.

**1.4 — gdown over browser download for the dataset.** The Drive folder has two files totaling 10.6GB; gdown handles Drive's large-file confirm-token redirect automatically and runs unattended in the background, where a browser-driven download would tie up an interactive tab and be harder to resume/monitor.

## Decision 2 — Problem scope (event day, 2026-08-22)

**2.1 — Levels unlock sequentially; build for L1 first, but design with L2–L5 in mind.** The platform locks L2 until L1 is submitted, so there's no benefit to skipping ahead — but L4 (spatial grounding via SRT telemetry) and L5 (network reasoning) both consume the same trajectory output L1 produces. Worth keeping the L1 output schema (track ID, class, per-frame bbox + timestamp) generic enough that later levels don't require a rewrite.

## Decision 3 — L1 detection quality: bigger model + imgsz over default (event day, first full run)

The first full run (YOLOv8n, default imgsz=640, conf=0.25) visibly missed a lot of vehicles, motorcycles, and pedestrians on inspection. Root cause: source footage is 4K (3840x2160); Ultralytics resizes every frame to imgsz before inference, so at 640px a pedestrian that's a few dozen pixels wide in the source becomes a handful of pixels — below what the model can reliably detect. YOLOv8n (nano) also has the weakest small-object recall in the family.

**Fix:** switched to `yolov8s.pt` at `imgsz=1280`, `conf=0.15` (was `yolov8n.pt`/640/0.25). Justified by GPU utilization sitting at 15-35% during the first run — we're decode-bound (4K video decode is CPU-bound), not compute-bound, so a bigger model and larger inference size should cost relatively little extra wall time. Validated on a short segment before committing to a full re-run on both videos (see progress.md for the before/after comparison).

**Not done:** tiled/sliced inference (running detection on cropped high-res tiles instead of one downsized frame) would likely help small-object recall even more, but adds real complexity and per-frame cost; treated as a stretch goal if L1 still underperforms after this change, not a first move.

## Decision 4 — SRT telemetry is embedded, not a sidecar file; parser format corrected

Resolves issue #2. The dataset's file listing (and the dashboard's Resources page) only ever mentioned the two MP4s — no separate `.srt` file. Checked with `ffprobe` and found the telemetry is embedded as a subtitle stream inside each MP4 itself (`Stream #0:1... Subtitle: mov_text`). Extracted via `ffmpeg -i <video> -map 0:s:0 <out>.srt`.

The real format also differs from `src/srt_telemetry.py`'s original assumption: frame count and timestamp share one line (`FrameCnt: <n> <date> <time>`, not `SrtCnt : <n>, DiffTime : ...` on a separate line from the date), and there are extra bracket fields (`ae_meter_md`, `dzoom_ratio`) plus gimbal orientation (`gb_yaw`/`gb_pitch`/`gb_roll`) not present in the originally-assumed layout. `_BLOCK_RE` updated and re-validated against the real extracted files for both videos (11,971 frames for Intersection, 9,140 for Multi_Road — exact match to each video's frame count; altitude a tight ~70.44-70.50m band, consistent with a hovering survey drone).

**Not done:** re-running full detection with `srt_path` wired in — the two full re-runs already in progress (Decision 3's model/imgsz fix) were left running rather than restarted a second time for what only affects the LGV/HGV sub-classification precision, not overall detection/tracking quality. Plan instead: post-process the existing output CSVs, re-classifying only the rows already labeled LGV/HGV (bbox + timestamp already present) using the now-available real altitude data, rather than paying for a third full pipeline run. Gimbal orientation fields aren't parsed yet — not needed for L1, but relevant for L4's ground-plane projection later.

## Decision 5 — SAHI + ROI restriction for Multi_Road_Merged's small-object tracking

FlytBase's own event announcement flagged this directly: teams are struggling to track/trace objects consistently, and recommended exploring SOD (small object detection) and SAHI specifically. A raw comparison confirmed it: full-frame SAHI slicing (640px tiles, 20% overlap) found **4-5x more detections per frame** than our single-pass yolov8s/imgsz=1280 pipeline on `Multi_Road_Merged.MP4` — but at **~15-25x the per-frame cost** (~0.5-0.77s vs ~0.03s), which would put a full run at 75-90 minutes.

**Fix: restrict SAHI slicing to the two actual road regions instead of the whole frame.** Confirmed via telemetry that this video's camera is static (gimbal yaw/pitch/roll essentially constant, altitude constant, across the full 5:05) — so fixed-pixel ROIs hold for the entire video, no re-mapping needed per frame. Mapped the two roads visually (main curving road + a highway) from a reference frame. ROI-restricted SAHI kept ~90% of the detection gain (163/138/139 vs 174/166/146 dets on the same 3 test frames) at roughly half the time (~0.28-0.29s/frame vs ~0.5-0.77s).

**Tracking integration, not just detection.** SAHI only detects, per-tile; feeding its merged per-frame detections into BoT-SORT for actual tracked IDs required bypassing Ultralytics' `model.track()` (which manages its own single-pass detection internally) and driving `ultralytics.trackers.bot_sort.BOTSORT` directly — converting SAHI's `object_prediction_list` into an `ultralytics.engine.results.Boxes` object per frame, calling `tracker.update(boxes, img=frame)` frame-by-frame. Reference: `src/sahi_tracking.py`.

**ReID disabled.** BoT-SORT's default config uses appearance-matching features (`with_reid: true`) extracted via a forward-pass hook into Ultralytics' own predict loop — not available when detection happens through SAHI's separate per-tile inference instead. Since this video's camera is confirmed static (see above), motion-only association is a much smaller downgrade than it would be on a panning shot, which is why BoT-SORT (vs. ByteTrack) was chosen in the first place for the *other* video.

**Validated before committing to the full run** (same pattern as Decision 3): ran the integrated pipeline on a 500-frame segment first. Result: 717 unique tracks vs. 114 from the standard pipeline in the identical window (6.3x), full frame coverage, ~0.35s/frame observed (~54 min projected for the full 9,140-frame video) — then committed to the full run.

**Scope: `Multi_Road_Merged.MP4` only.** `Intersection_Merged.MP4` already has strong results from the standard pipeline (6,689 tracks, full class coverage) and wasn't flagged as a weak point, so it wasn't reprocessed with this approach.
