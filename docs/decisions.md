# Decisions

Running log of non-obvious choices and why we made them. Dated records — don't rewrite past entries when something changes, add a new one that supersedes it.

## Decision 1 — Environment (pre-event, 2026-08-21/22)

**1.1 — Local GPU + Kaggle as reserve, not primary.** RTX 4050 Laptop (6GB VRAM) handles YOLOv8n fine locally. Kaggle T4 x2 (30 GPU-hrs/week) is kept as a fallback if local VRAM becomes a bottleneck (e.g. heavier models, longer batch runs), not the main compute path — avoids depending on a shared/queued resource for the core pipeline.

**1.2 — Gemini 3.6 Flash, not 2.5 Flash.** The prerequisite guide's own snippet uses `gemini-2.5-flash`, which now 404s ("no longer available to new users"). Switched to `gemini-3.6-flash`, confirmed working. If this drifts again, check aistudio.google.com for the current Flash model name before assuming the code is wrong.

**1.3 — Claude Code Pro, no OpenRouter.** The guide's coding-setup requirement is satisfied by the Claude Code Pro subscription already in place — no need for the guide's OpenRouter/free-model fallback path.

**1.4 — gdown over browser download for the dataset.** The Drive folder has two files totaling 10.6GB; gdown handles Drive's large-file confirm-token redirect automatically and runs unattended in the background, where a browser-driven download would tie up an interactive tab and be harder to resume/monitor.

## Decision 2 — Problem scope (event day, 2026-08-22)

**2.1 — Levels unlock sequentially; build for L1 first, but design with L2–L5 in mind.** The platform locks L2 until L1 is submitted, so there's no benefit to skipping ahead — but L4 (spatial grounding via SRT telemetry) and L5 (network reasoning) both consume the same trajectory output L1 produces. Worth keeping the L1 output schema (track ID, class, per-frame bbox + timestamp) generic enough that later levels don't require a rewrite.
