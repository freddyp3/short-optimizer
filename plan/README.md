# Shorts Pre-Publish Copilot: Master Plan

Single source of truth for the project. Phase docs live next to this file (`01-…` to `08-…`).
Anything in `CLAUDE.md` that disagrees with this folder is stale; this folder wins.

## What it is

An AI agent system for Brawl Stars YouTube Shorts (channel @alfie369k). A creator uploads a draft
Short and gets back a **publish plan**:

1. Hook and pacing feedback with timestamps (e.g. "viewers likely leave at 0:07").
2. Predicted performance vs the channel's own baseline (retention / swipe rate / engagedViews, not raw views).
3. Trending hashtags (computed live per run from a watchlist of top Brawl Stars channels, not stored).
4. A sound suggestion (official "Top Songs on Shorts" chart + user-supplied sounds).

It is built on the creator's own channel data and validated with evals against real retention data.

## Goals

- A working, public product: web app, target **500+ real creators**.
- A portfolio piece for FDE / AI engineering roles: agents, tool use, multimodal, evals, MCP, deployment, security.
- Resume shape (only measured numbers go on the real resume):
  1. Used the copilot on ~30 of my own new Shorts and measured growth (use `engagedViews`; Shorts view counting changed 2025-03-31), then launched publicly to 500+ creators.
  2. Multimodal eval (Gemini video analysis vs real retention curves).
  3. Claude tool-use agent with latency numbers and prompt-injection defense.
  MCP server is in scope and goes on the resume.

## Working rules

- Writing complete implementations is fine and expected. Explain key decisions briefly.
- Pace: build end-to-end; do not pause between phases for checkpoints.
- Never commit `.env`, `secrets/`, OAuth client secrets, or tokens. `data/` is gitignored.
- Python 3.11+, venv at `./venv`. Run tests with `./venv/bin/python -m pytest tests/`.
- Views are heavy-tailed (median ~13k, max 10.6M): target retention, swipe rate and relative metrics.

## Phases (build order)

| # | Doc | Deliverable | Gate / milestone |
|---|-----|-------------|------------------|
| 1 | [01-channel-export.md](01-channel-export.md) | Own-channel data in `data/channel/` | code written; **sign-in + first run pending** |
| 2 | [02-hook-test.md](02-hook-test.md) | Feasibility result on 40 Shorts | **Go/no-go gate**, M1 ~2026-10-10 |
| 3 | [03-eval-harness-analyze-video.md](03-eval-harness-analyze-video.md) | Eval dataset, harness, `analyze_video` tool | first eval number |
| 4 | [04-trend-tools.md](04-trend-tools.md) | `trending_hashtags`, `trending_audio` | |
| 5 | [05-orchestrator-agent.md](05-orchestrator-agent.md) | Claude agent producing the publish plan | |
| 6 | [06-persona-panel.md](06-persona-panel.md) | `audience_panel` tool | **ships only if evals show it beats a single model** |
| 7 | [07-mcp-server.md](07-mcp-server.md) | MCP server for channel data + tools | |
| 8 | [08-api-deploy-security-growth.md](08-api-deploy-security-growth.md) | FastAPI, Docker, cloud deploy, injection defense, growth | M2 ~early Nov (web app), M3 security + growth |

Parked, not dropped: thumbnail/title CTR model (see `../BACKLOG.md`).

## Architecture (end state)

```
                    ┌────────────────────────┐
 draft Short ──────▶│  FastAPI  (web app/API) │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐   traces + cost per run
                    │ Orchestrator (Claude,  │──────────────────────────▶ observability
                    │ tool use)              │
                    └─┬──────┬───────┬─────┬─┘
        analyze_video │      │       │     │ audience_panel (gated)
          (Gemini)    │      │       │     │
                      ▼      ▼       ▼     ▼
              trending_hashtags  trending_audio  channel data (baseline)
                      │                              ▲
                      └──────── same tools exposed ──┘
                                via MCP server
 data/channel/  ◀── channel exporter (OAuth, own channel only)
 eval dataset   ◀── past Shorts + real outcomes; eval harness scores every tool/agent change
```

## Repo layout (target)

```
src/channel_export/   Phase 1 (exists)
src/tools/            analyze_video, trending_hashtags, trending_audio, audience_panel
src/agent/            orchestrator
src/evals/            dataset builder, harness, metrics
src/mcp_server/       MCP server
src/api/              FastAPI app
src/obs/              tracing + cost tracking
plan/                 this folder
tests/
data/                 gitignored
```

## Hard constraints

- YouTube Developer Policies: other channels' API data may be stored ≤30 days, no scraping, no aggregating
  creators' data, restrictions on derived metrics. Own-channel data is fine. Hence trend tools compute live per run.
- Quota: 10,000 units/day for most methods; `search.list` reportedly has its own bucket (100 calls/day since June 2026), verify at developers.google.com/youtube/v3/determine_quota_cost.
- Shorts view counting changed 2025-03-31 (replays count): use `engagedViews` across that date.
- No legal automated Short→sound mapping exists, so audio stays thin.
- Keep a held-out eval set; iterating prompts on the same 40 videos overfits.

## Milestones

- **M1 ~2026-10-10:** first eval number (hook test result).
- **M2 ~early Nov:** shipped web app.
- **M3:** MCP server hardened, injection defense, growth to 500+ creators.
