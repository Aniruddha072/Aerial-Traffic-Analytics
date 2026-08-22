# Handoff

For picking this up cold — a fresh session, a teammate, or future you after a break.

## What this is

Solo entry to the FlytBase Visual Intelligence Hackathon (2026-08-22, remote). Build a system that extracts road-user trajectories from raw drone footage and derives traffic insight from them, across 5 sequential levels (650 pts total). See `README.md` for the full problem statement and level table.

## Where things stand

Check `docs/progress.md` for the live checklist — that's the source of truth, not this file. This file is about *how* to pick things back up, not *what's* done.

## Key facts you'll need

- Hackathon ID: `aa2c5b58-2047-44bb-9902-45c05c1530cf`
- Dashboard: https://fbhackathonplatform-production.up.railway.app/participant/p_0-S4Pl (login as Aniruddha More — if the browser shows a different name, it's the wrong session, log out and back in)
- Dataset: `data/Intersection_Merged.MP4`, `data/Multi_Road_Merged.MP4` (gitignored — if missing, re-run gdown against the [Drive folder](https://drive.google.com/drive/folders/1YvfPkzp7xZUJN5VmeCswu2xePYCMe4Xv), see `docs/decisions.md` 1.4)
- `.env` has `GEMINI_API_KEY` (gitignored, not in this repo — recover from the local machine or regenerate at aistudio.google.com)
- Setup runbook: [Visual Intelligence Playbook artifact](https://claude.ai/code/artifact/addc13d5-8e4e-4784-a7b6-b88c2e2da25e)

## Gotchas already hit (don't rediscover these)

- `gemini-2.5-flash` 404s — use `gemini-3.6-flash` (check aistudio.google.com if this drifts again)
- The dashboard's homepage "Participant" link is a fixed shortcut, not session-aware — it can show whoever's session cookie happens to be active in that browser profile, not necessarily you. Verify the name shown before trusting the page.
- Dataset server at `192.168.10.85:8000` is venue-WiFi-only — use the Drive folder instead when remote.

## Submission format (per level)

Code as `.zip` (≤200MB) + a write-up link (Drive/Notion/etc.) + a video demo link, submitted through the dashboard.
