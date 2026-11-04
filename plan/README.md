# Shorts Pre-Publish Copilot: Master Plan

Single source of truth for the project. Phase docs live next to this file (`01-…` to `08-…`).
Anything in `CLAUDE.md` that disagrees with this folder is stale; this folder wins.

## What it is

An AI agent system for **Brawl Stars YouTube Shorts only** (channel @alfie369k). Scope is deliberately narrow so it
ships: the checklist, categories, watchlist and prompts are all Brawl Stars specific. A creator gives a draft Short to a
Gemini agent, which calls the project's **MCP server** and returns a **publish plan**:

1. Hook and pacing feedback with timestamps (e.g. "viewers likely leave at 0:07").
2. Predicted retention vs the channel's own baseline (`hold_resid`), with feature-level reasons.
2b. Predicted audience reaction, grounded in real comments on similar past Shorts and on a rolling 30-day pool of
    other Brawl Stars clips (see "Product definition").
3. Trending hashtags (computed live per run from a watchlist of top Brawl Stars channels, not stored).
4. A sound suggestion (official "Top Songs on Shorts" chart + user-supplied sounds).

It is built on the creator's own channel data and validated with evals against real retention data.

## Revision 2026-11-04: MCP first, Gemini agent, Brawl Stars only (supersedes conflicting parts below and in phase docs)

Decided by Fredrik on 2026-11-04:

1. **Brawl Stars only.** No generalising to other games or niches. "500+ creators" means Brawl Stars creators.
2. **MCP server first; the web app is a later stretch goal.** The MCP server is the product's core runtime, not an
   adapter added at the end. FastAPI + deploy (phase 8) only happens after the MCP version works.
3. **Gemini is the agent** (replaces the Claude orchestrator). The creator sends Gemini the draft video plus an
   instruction; Gemini calls the MCP tools and writes the publish plan. Claude is no longer in the stack.
4. **Retention is predicted server-side.** The MCP tool `predict_retention(video)` runs the *frozen* extraction prompt
   (phase 3a), the code measurements and the ridge model, and returns `hold_resid` plus per-feature "why" lines. The
   agent never chooses or fills the features itself: ridge only works on exactly the checklist it was trained on,
   extracted the same way, and code-measured features need the video file on the server.
5. **Reaction uses two sources:**
   - `similar_past_shorts`: similarity search over the creator's own past Shorts and their real comments ("your
     audience");
   - `category_reactions`: a rolling 30-day pool of other channels' Brawl Stars clips, collected via the official API
     only, each tagged with a category from a **fixed, hand-written list**, deleted at 30 days ("fresh, more volume").
     Gemini picks the category closest to the draft and generates likely comments and a viewer consensus.
6. **Policy read first.** The 30-day pool (stored comments from other creators) is only built if a focused read of the
   YouTube Developer Policies says it is allowed (no prohibited aggregation, storage and deletion rules met).
7. **Generated comments and consensus are still eval-gated:** they ship only if they beat simply showing the real
   comments from the matched category and similar past Shorts (phase 6, section 0).

Unchanged: the hook test (Step 2), the feature checklist, ridge training and its GATE/HELDOUT evals.

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

Output shape (written by the Gemini agent from MCP tool results; 2026-11-04 revision adds a "fresh pool" line from
`category_reactions`):

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
  Which key the Gemini agent uses (Step 6) is an open decision for Fredrik (cost).
- **Comment text is untrusted input** (prompt injection): it is data, never instructions, and is covered by the Step 8
  security tests.

## Goals

- A working MCP server + Gemini agent first; a public web app for **500+ Brawl Stars creators** is a stretch goal.
- A portfolio piece for FDE / AI engineering roles: agents, tool use, multimodal, evals, MCP, security (deployment if
  the stretch goal happens).
- Resume shape (only measured numbers go on the real resume):
  1. Used the copilot on ~30 of my own new Shorts and measured growth (use `engagedViews`; Shorts view counting changed 2025-03-31); public launch to 500+ creators if the web app ships.
  2. Multimodal eval (Gemini feature extraction + ridge model vs real retention).
  3. MCP server + Gemini tool-use agent with latency numbers and prompt-injection defense.

## Working rules

- The agent writes complete implementations, one step at a time. After each step it explains the step and waits for "next" (full rules in `AGENTS.md` under "How we work").
- Minimise spend: free tiers first, ask before any paid API usage.
- Never commit `.env`, `secrets/`, OAuth client secrets, or tokens. `data/` is gitignored.
- Python 3.11+, venv at `./venv`. Run tests with `./venv/bin/python -m pytest tests/`.
- Views are heavy-tailed (median ~13k, max 10.6M): target retention, swipe rate and relative metrics.

## Phases (build order)

Revised 2026-11-04 (MCP first). Steps, not doc numbers: several steps draw on more than one doc.

| Step | Deliverable | Doc(s) | Gate / milestone |
|---|---|---|---|
| 1 | Own-channel data in `data/channel/` | [01](01-channel-export.md) | done |
| 2 | Hook test on 40 Shorts (becomes the baseline) | [02](02-hook-test.md) | in progress, unchanged |
| 3 | Feature checklist, extraction, code features, ridge model = the internals of `predict_retention` | [03](03-eval-harness-analyze-video.md), [03a](03a-feature-checklist.md), [03b](03b-consistency-ab.md) | beats hook-test Gemini baseline on GATE, then HELDOUT number |
| 4 | **Policy read**: cross-channel comment pool + hashtags | [04](04-trend-tools.md) §8, [06](06-persona-panel.md) §0 | go/no-go for Step 7 |
| 5 | **MCP server v1**: `predict_retention`, `similar_past_shorts`, `get_video_metrics`, `get_retention_curve`, `trending_hashtags`, `trending_audio` | [07](07-mcp-server.md), [04](04-trend-tools.md), [06](06-persona-panel.md) §0 | works from a real MCP client |
| 6 | **Gemini agent client**: calls the MCP server, writes the publish plan | [05](05-orchestrator-agent.md) | end-to-end plan on a real draft |
| 7 | **Rolling 30-day pool**: collection job, fixed category list, tagging, `category_reactions` tool | [06](06-persona-panel.md) §0 | only if Step 4 says OK |
| 8 | Reaction eval: generated comments/consensus vs showing real ones | [06](06-persona-panel.md) §0 | generation ships only if it wins |
| 9 | Web app: FastAPI, Docker, Cloud Run, growth | [08](08-api-deploy-security-growth.md) | **stretch** |

Injection defense (comment text is untrusted) is built into Steps 5-7, not deferred to Step 9.

Parked, not dropped: thumbnail/title CTR model (see `../BACKLOG.md`).

## Architecture (end state, revised 2026-11-04)

```
 draft Short ──▶ Gemini agent (MCP client: Python script; Gemini CLI for demos)
                    │  watches the video, calls tools, writes the publish plan
                    ▼                                   traces + cost per run ──▶ observability
            ┌───────────────────────── MCP server (core) ─────────────────────────┐
            │ predict_retention    frozen extraction (Gemini) + code features     │
            │                      + ridge → hold_resid + "why" lines             │
            │ similar_past_shorts  own past Shorts + their real comments          │
            │ category_reactions   rolling 30-day pool (gated by policy read)     │
            │ trending_hashtags    live, watchlist of top Brawl Stars channels    │
            │ trending_audio       Top Songs on Shorts chart + user sounds        │
            │ get_video_metrics / get_retention_curve   own-channel data          │
            └─────────────────────────────────────────────────────────────────────┘
 data/channel/  ◀── channel exporter (OAuth, own channel only)
 data/pool/     ◀── 30-day collector (official API only, auto-delete at 30 days)
 eval dataset   ◀── past Shorts + real outcomes; eval harness scores every tool/agent change
 (stretch) FastAPI web app in front of the same agent + MCP server
```

## Repo layout (target)

```
src/channel_export/   Phase 1 (exists)
src/tools/            analyze_video (frozen extractor), predict_retention, similar_past_shorts, category_reactions,
                      trending_hashtags, trending_audio
src/models/           retention model (features -> hold_resid), reaction-label builders
src/pool/             30-day collector, category tagging, expiry sweeper (Step 7)
src/agent/            Gemini agent client (MCP client)
src/evals/            dataset builder, harness, metrics
src/mcp_server/       MCP server (core runtime)
src/api/              FastAPI app (stretch)
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
- **M2:** MCP server + Gemini agent producing a full publish plan on a real draft (Steps 5-6).
- **M3:** reaction pool + reaction eval (Steps 7-8), injection defense.
- **Stretch:** web app and growth to 500+ Brawl Stars creators (Step 9).
