# Progress

Live build log. One entry per meaningful step — what happened, not a diary.

## 2026-08-21/22 — Environment setup (pre-event)

- [x] venv, PyTorch+CUDA 12.8 confirmed on RTX 4050
- [x] YOLOv8n smoke test — 6 objects detected on test image
- [x] CLIP (open-clip ViT-B/32) loads OK
- [x] Gemini API working (`gemini-3.6-flash`, key verified)
- [x] Kaggle GPU T4 x2 reserve verified (`True` / `Tesla T4`)
- [x] Claude Code Pro confirmed as coding setup

## 2026-08-22 — Event day

- [x] Hackathon went live, dashboard access confirmed (participant HUD, rank #8 at check-in)
- [x] Problem statement pulled from Quest Map: build a traffic analysis agent, trajectories → insight, 5 levels / 650 pts
- [x] Dataset download kicked off (`Intersection_Merged.MP4` + `Multi_Road_Merged.MP4`, ~10.6GB, → `data/`)
- [x] Repo initialized, docs scaffolded
- [ ] L1 — Detection & Tracking: not started
- [ ] L2 — Object-Level Insight: locked (needs L1 submitted)
- [ ] L3 — Aggregate Insight: locked
- [ ] L4 — Spatial Grounding: locked
- [ ] L5 — Network Reasoning: locked

## Next up

Once the dataset finishes downloading: scope the L1 pipeline (YOLOv8n detection + ByteTrack/BoT-SORT tracking, class-mapped to car/LGV/HGV/bus/truck/motorcycle/pedestrian, stable IDs through occlusion).
