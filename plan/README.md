# Shorts Pre-Publish Copilot: Master Plan

Single source of truth for the project. Phase docs live next to this file (`01-…` to `08-…`).
Anything in `CLAUDE.md` that disagrees with this folder is stale; this folder wins.

## What it is

An AI agent system for Brawl Stars YouTube Shorts (channel @alfie369k). A creator uploads a draft
Short and gets back a **publish plan**:

1. Hook and pacing feedback with timestamps (e.g. "viewers likely leave at 0:07").
2. Predicted retention vs the channel's own baseline (`hold_resid`), with feature-level reasons.
2b. Predicted audience reaction, grounded in real comments on the most similar past Shorts (see "Product definition").
3. Trending hashtags (computed live per run from a watchlist of top Brawl Stars channels, not stored).
4. A sound suggestion (official "Top Songs on Shorts" chart + user-supplied sounds).

It is built on the creator's own channel data and validated with evals against real retention data.

## Product definition (decided 2026-11-03)

The product makes two predictions for a draft Short, and every claim is backed by evidence:

1. **Retention:** the predicted `hold_resid` (does it hold viewers better or worse than this channel's typical Short of
   the same length), shown as "+8 vs your usual", with reasons ("payoff at 9 s: -6").
   - **How:** Gemini acts as a **feature extractor**: it fills a fixed checklist per video (format, hook, payoff
     timing, dead time, ending, audio). Code measures what code can measure (cuts/s, loudness, length). A small
     regularised model trained on the channel's own past Shorts maps features to `hold_resid`. Gemini describes; the
     channel's data decides what matters. Details: [03](03-eval-harness-analyze-video.md), section 0b.
2. **Audience reaction:** how this channel's viewers will react, and why.
   - **How:** **retrieval, then grounded audience agents.**
     1. Retrieve: find the channel's most similar past Shorts (by checklist features) and pull their real top comments,
        comment themes and engagement rates (like/comment/share per engaged view).
     2. React: LLM audience agents (the persona panel) read that evidence plus the draft's analysis and predict the
        reaction ("expect 'part 2' requests and arguments about Sprout"), citing the past videos and comments they
        are based on.
   - Details and eval: [06](06-persona-panel.md), section 0.

Output shape (assembled by the orchestrator, Step 5):

```
Retention:  +8 vs your usual
  why: hook text in first second (+4), clean loop (+3), payoff at 9s (-6, late for you)
Viewer reaction:
  most like: "Guess the brawler" (3 past videos, avg +12)
  real comments on those: "part 2 pls" · "got it in 3 sec" · "sprout was obvious"
  predicted reaction (audience agents): part-2 requests, mild arguments; confidence: low (thin comments)
Hashtags / sound: Step 4
```

Rules that follow from this:
- **Comments, likes and other engagement are labels and retrieval evidence for *past* videos, never inputs about the
  draft being judged.** A draft has no comments yet; using them as features leaks the answer.
- **Eval video comments stay hidden when that video is the one being predicted** (leave-one-out retrieval).
- **The hook test (Step 2) still runs as pre-registered.** Gemini's holistic judgement becomes the baseline row the
  feature model must beat; nothing about the GATE pass mark changes.
- **Paid Gemini key (`GEMINI_API_KEY_PAID`) is for feature extraction only;** the hook test stays on the free key.
- **Comment text is untrusted input** (prompt injection): it is data, never instructions, and is covered by the Step 8
  security tests.

## Goals

- A working, public product: web app, target **500+ real creators**.
- A portfolio piece for FDE / AI engineering roles: agents, tool use, multimodal, evals, MCP, deployment, security.
- Resume shape (only measured numbers go on the real resume):
  1. Used the copilot on ~30 of my own new Shorts and measured growth (use `engagedViews`; Shorts view counting changed 2025-03-31), then launched publicly to 500+ creators.
  2. Multimodal eval (Gemini video analysis vs real retention curves).
  3. Claude tool-use agent with latency numbers and prompt-injection defense.
  MCP server is in scope and goes on the resume.

## Working rules

- The agent writes complete implementations, one step at a time. After each step it explains the step and waits for "next" (full rules in `AGENTS.md` under "How we work").
- Minimise spend: free tiers first, ask before any paid API usage.
- Never commit `.env`, `secrets/`, OAuth client secrets, or tokens. `data/` is gitignored.
- Python 3.11+, venv at `./venv`. Run tests with `./venv/bin/python -m pytest tests/`.
- Views are heavy-tailed (median ~13k, max 10.6M): target retention, swipe rate and relative metrics.

## Phases (build order)

| # | Doc | Deliverable | Gate / milestone |
|---|-----|-------------|------------------|
| 1 | [01-channel-export.md](01-channel-export.md) | Own-channel data in `data/channel/` | code written; **sign-in + first run pending** |
| 2 | [02-hook-test.md](02-hook-test.md) | Feasibility result on 40 Shorts | **Go/no-go gate**, M1 ~2026-10-10 |
| 3 | [03-eval-harness-analyze-video.md](03-eval-harness-analyze-video.md) | Feature extraction (`analyze_video`), retention model (`hold_resid`), eval harness | beats hook-test Gemini baseline on GATE, then HELDOUT number |
| 4 | [04-trend-tools.md](04-trend-tools.md) | `trending_hashtags`, `trending_audio` | |
| 5 | [05-orchestrator-agent.md](05-orchestrator-agent.md) | Claude agent producing the publish plan | |
| 6 | [06-persona-panel.md](06-persona-panel.md) | `similar_videos` retrieval + `audience_panel` grounded agents | retrieval ships; **agents ship only if they beat retrieval-only and a single model** |
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
src/tools/            analyze_video (feature extractor), similar_videos, trending_hashtags, trending_audio, audience_panel
src/models/           retention model (features -> hold_resid), reaction-label builders
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
