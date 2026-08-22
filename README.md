# FlytBase Visual Intelligence Hackathon

Solo entry — build a traffic analysis agent from raw drone footage. No annotations given; the platform scores depth of derived insight, not just detection accuracy.

- **Event day:** 2026-08-22, remote track

## The problem

Fixed traffic cameras are alarm generators: closed event vocabulary, single-approach perspective, coverage stuck wherever a camera was installed. A drone removes all three constraints — full-scene view in one frame, arbitrary derived events, deployable anywhere. The task: extract trajectories from aerial video and derive insight from them. Detection & tracking is the floor, not the objective.

## Levels (650 pts total, sequential unlock)

| Level | Focus | Pts |
|---|---|---|
| L1 | Detection & Tracking — classify by mode, stable ID through occlusion | 100 |
| L2 | Object-Level Insight — fine-grained class, plate reading, kinematics | 150 |
| L3 | Aggregate Insight — counts, OD, speed profiles, queues, density/flow | 200 |
| L4 | Spatial Grounding — SRT telemetry → ground-plane → road network | 100 |
| L5 | Network Reasoning — congestion origin, signal performance, desire lines | 100 |

Each level submits as: code `.zip` (≤200MB), a write-up link, a video demo link.

## Dataset

Two drone videos with SRT telemetry (GPS, altitude, gimbal orientation, per-frame timestamps), downloaded to `data/` (gitignored, ~10.6GB):
- `Intersection_Merged.MP4` — 6.0GB, 6:39
- `Multi_Road_Merged.MP4` — 4.6GB, 5:05

Source: [Drive folder](https://drive.google.com/drive/folders/1YvfPkzp7xZUJN5VmeCswu2xePYCMe4Xv)

## Stack

- **Detection:** YOLOv8n (Ultralytics), local RTX 4050
- **Tracking:** ByteTrack/BoT-SORT (ships with Ultralytics) — for stable IDs through occlusion
- **Reasoning/NL:** Gemini 3.6 Flash (accepts video directly)
- **Embeddings/retrieval:** CLIP (open-clip ViT-B/32) + FAISS
- **Compute reserve:** Kaggle T4 x2 (30 GPU-hrs/week) if local GPU is insufficient

## Setup

See the [Visual Intelligence Playbook](https://claude.ai/code/artifact/addc13d5-8e4e-4784-a7b6-b88c2e2da25e) artifact for the full environment setup runbook. Quick start:

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt
```

`.env` holds `GEMINI_API_KEY` (gitignored).

## Docs

- [docs/decisions.md](docs/decisions.md) — why we chose what we chose
- [docs/superpowers/specs/](docs/superpowers/specs/) — design specs
- [docs/superpowers/plans/](docs/superpowers/plans/) — implementation plans
