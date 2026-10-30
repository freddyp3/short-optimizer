# Phase 2: Hook Test (feasibility gate)

Status: design, ready to implement. Milestone M1 ~2026-10-10. Depends on Phase 1 output (`data/channel/<timestamp>/`).

**Question this phase answers:** can Gemini, watching only the video file (no title, no stats), tell which of the creator's
Shorts hold viewers and where viewers leave? If not better than chance, `analyze_video` (Phase 3) and the whole
"hook/pacing feedback" pitch has no evidence behind it, and we pivot before building the agent.

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Label metric | `hold_resid`: lifetime `averageViewPercentage` residualised on log(duration) and era, sanity-checked against YouTube's own length-matched `relativeRetentionPerformance` |
| Swipe rate | Not in any API we export. Optional manual secondary label from Studio ("Viewed vs swiped away") for the 40 gate videos only |
| 2025-03-31 view change | Retention and `averageViewPercentage` are ratios, not counts, but still mix definitions for old videos that kept getting views after the change. Rule: eligible pool = published before 2025-03-01 AND <=20% of lifetime `engagedViews` after 2025-03-31 |
| Splits | `DEV` 10/10 (prompt iteration), `GATE` 20/20 (frozen prompt, one confirmatory run), `HELDOUT` rest of the extremes (touched once, at the end of Step 3) |
| Video files | Google Takeout (bulk, officially allowed) or creator's local source files; Studio per-video download as fallback; yt-dlp only as last resort (against YouTube ToS, see section 3) |
| Model | `gemini-3.8-flash` primary; `gemini-3.1-pro-preview` or `gemini-4-argon` as one comparison run (Argon preferred if the account has access: ~same cost as Pro, newer). fps 5 and high media resolution on the gate; fps/resolution ablation on DEV only |
| Output | Pointwise JSON: `p_strong` 0..1, `confidence`, `dropoff_timestamp` (seconds), short reasons |
| Primary scoring | Rank-based group call (top 20 `p_strong` of 40 = "strong") must reach >=26/40; drop-off scored by percentile rank of retention loss at the predicted time with a permutation test |
| Repeats | 3 runs per video at default sampling, scores averaged before scoring (no best-of-3) |
| Budget | ~$5 for the gate on Flash, ~$15 if Pro comparison is included; hard cap $30 for the whole phase |

## 1. Inputs (what Phase 1 gives us)

From the channel exporter (`src/channel_export/`), all under `data/channel/<UTC timestamp>/`:

- `videos.csv`: `video_id`, `title`, `published_at`, `content_type`, `is_short`, `duration_seconds` (+ other flattened Data API columns).
- `video_metrics.csv`: one row per (`video_id`, `creatorContentType`): `engagedViews`, `views`, `estimatedMinutesWatched`,
  `averageViewDuration`, `averageViewPercentage`, likes, comments, shares, subscribers gained/lost. Use rows with `creatorContentType == SHORTS`.
- `retention.csv`: per video, 100 points of `elapsedVideoTimeRatio` with `audienceWatchRatio`, `relativeRetentionPerformance`,
  `startedWatching`, `stoppedWatching`, `totalSegmentImpressions`.
- `video_daily.csv`: daily `engagedViews`/`views` per video (used for the post-2025-03-31 share).

Facts that shape the design (flagged because they are easy to get wrong):

- `audienceWatchRatio` for Shorts can exceed 1 and can rise mid-curve (loops, rewatches). Do not treat the curve as monotone.
- `relativeRetentionPerformance` (0..1) is YouTube's comparison against videos of similar length across YouTube, i.e. already
  length-controlled, but relative to all of YouTube rather than this channel. Good as a sanity check, not as the sole label.
- Low-view videos can have empty retention rows (privacy threshold); they drop out of the eligible pool automatically.
- Swipe rate ("viewed vs swiped away") is a Studio UI number. The nearest API proxy is `engagedViews / views`, but only
  post-2025-03-31 (before it `views` and `engagedViews` were the same definition, so the ratio is ~1). Our pool is mostly
  pre-change, so this proxy is **not usable** for selection. Not used.
- Retention is queried over publish date .. export end date, so curves are lifetime curves.

## 2. Selecting the videos

Script: `src/evals/hook_test/select.py` -> `data/evals/hook_test/selection.json` + `selection.csv`. Deterministic (fixed seed 340), no network.

### 2.1 Eligible pool

Keep a Short if all hold:

1. `content_type == SHORTS`, `duration_seconds` between 8 and 60.
2. `published_at < 2025-03-01` (one-month buffer before the 2025-03-31 view-counting change).
3. Share of lifetime `engagedViews` after 2025-03-31 (from `video_daily.csv`) is <= 20%. Videos with a long tail past the change have curves that mix two
   view definitions; exclude rather than correct.
4. `engagedViews >= 1000` and a complete 100-point retention curve (no gaps). Below this the curve is mostly noise.
5. Video file obtained and its ffprobe duration is within 0.5 s of `duration_seconds` (section 3).

Expect ~150-220 of the 277. Write the exclusion counts per rule into `selection.json` (`pool_funnel`). If the pool is <120, relax rule 3 to 35% and record it.

### 2.2 The label metric

Primary: `hold_score = averageViewPercentage` (lifetime, Shorts rows). It is the fraction of the video the average view watched.

Length confound. Shorter videos mechanically retain better, so raw `averageViewPercentage` would make "short = strong" the winning heuristic. Fix in two layers:

1. **Residualise.** Fit `averageViewPercentage ~ a + b*log(duration_seconds) + c*is_late_era` by OLS (numpy `lstsq`, era = published after 2024-09-01, adjust the cutoff to the median publish date).
   `hold_resid` = residual. Strong/weak are defined on `hold_resid`, not the raw value.
2. **Stratify and verify.** Cut the pool into 3 duration tertiles; take strong/weak candidates within each tertile in equal numbers (rounded). After selection, run a Mann-Whitney U test (implement with numpy rank sums, normal approximation is fine at n=20) of duration strong vs weak: require p > 0.2 and |median difference| <= 2 s. ~~If it fails, re-draw with the next seed (`seed+1`, up to 20 tries) and record the attempt number.~~
**Rerandomization (2026-10-29, defined before running, before any Gemini call):** draw seeds 340..389 (50 draws). Discard draws where DEV, GATE or HELDOUT fails the rules above (duration p, date p, median gap). Among the rest, keep the lowest score = max |standardized mean difference| strong vs weak (difference of means / eligible-pool SD) over {DEV, GATE, HELDOUT} x {duration, publish date}; ties go to the lowest seed. All 50 attempts and scores are stored in `selection.json`.

~~Sanity filter (label quality): keep a strong candidate only if `mean(relativeRetentionPerformance) > 0.5` and a weak candidate only if `< 0.5`.~~
**Dropped (2026-10-29, before any Gemini call).** On the real pool the median `mean_rrp` is 0.65 (the channel beats YouTube's length-matched average on most videos) and `mean_rrp` correlates -0.67 with duration. The filter removed 31/59 weak vs 3/59 strong candidates, left 28 pairs (< 36 minimum), and widened the strong/weak median duration gap to 5-7.5 s, i.e. it reintroduced the length confound. Labels are plain top/bottom quartiles of `hold_resid`; `mean_rrp` stays a secondary label.

**Category (2026-10-29):** the reference class is Brawl Stars only (title contains "brawl", plus 7 creator-confirmed hashtag-only Sep 2025 Shorts listed in `categorize.py`). Clash Royale (3 videos), the Nov-Dec 2023 hyper-casual game uploads and non-gaming uploads are excluded.

Secondary labels, computed and stored but **not** used for selection (analysed after the gate, to see whether predictions track them):
- `hook3s = audienceWatchRatio at elapsedVideoTimeRatio = min(1, 3/duration)`: the "hook" survival; closest available analogue of swipe rate.
- `mean_rrp = mean(relativeRetentionPerformance)`.
- `swipe_manual` (optional): from Studio per-video Engagement tab "Viewed vs swiped away", typed into `data/evals/hook_test/swipe_manual.csv` for the 40 GATE videos only (~15 min). Report the correlation of `p_strong` with it.

### 2.3 Upload-date confound

Retention drifts with algorithm/audience changes and game updates, and visuals reveal the era (new brawlers, UI changes), so Gemini could score "looks 2024-spring" instead of "holds viewers".
- Stratify by publish-date tertile as well as duration: build the 3x3 grid, sample strong and weak equally per column where counts allow (relax to row-only if a cell has <2 candidates).
- Verify with Mann-Whitney on `published_at` strong vs weak (require p > 0.2).
- Include `is_late_era` in the residual model (above).
- Run the **date-only** and **duration-only** baseline classifiers (section 7.3) and report them next to Gemini. Both should sit near 50%; if one scores >=65% the sample is confounded and the result is void until fixed.

### 2.4 Candidate pools and splits

After residualising and the sanity filter, rank by `hold_resid`:
- strong candidates = top 25% of the eligible pool; weak candidates = bottom 25%. The middle 50% is deliberately excluded (labels there are noisy and not what a feasibility test should hinge on). Expect ~40-55 per side.
- Allocate within the stratification grid, then split **disjointly** and persist the split before any Gemini call:

| Split | Strong | Weak | Purpose | Rules |
|---|---|---|---|---|
| `DEV` | 10 | 10 | write/tune prompt v1..vN, pick fps/resolution, debug schema | free to iterate |
| `GATE` | 20 | 20 | the 40 videos of the gate | prompt frozen (git tag `hook-test-prompt-final`, SHA-256 of prompt file in `selection.json`) before first run; **one** confirmatory 3-run batch |
| `HELDOUT` | all remaining strong/weak candidates (aim 20/20, accept >=15/15) | | final unbiased number for Step 3 and the resume | not run at all in Step 2 unless the decision tree says so; run once at the end of Step 3 |

If candidate count is too small for 10/10 + 20/20 + 15/15 (needs ~45 per side), shrink `DEV` to 6/6 first, then `HELDOUT` to 10/10, never `GATE`.
`selection.json` stores per video: `video_id`, `split`, `label` (strong/weak), `hold_score`, `hold_resid`, `duration_seconds`, `published_at`, `mean_rrp`, `hook3s`, `blind_id`.
`blind_id` = `v` + 8 random hex chars from `secrets.token_hex(4)` (seeded), stored only in this file.

Pre-register: commit `selection.json` and `plan/02-hook-test.md` (this file) before running the gate. Do not edit the thresholds after seeing results.

## 3. Obtaining the video files

The APIs we use (Data v3, Analytics v2, Reporting) provide no media download. Options, best first. Allowed/not-allowed notes come from YouTube's help page for downloading your own uploads and general ToS reading; verify current wording before relying on it.

| # | Option | Effort | Status |
|---|---|---|---|
| 1 | **Google Takeout** (takeout.google.com -> YouTube and YouTube Music -> Videos; choose only the channel's uploads) returns the original uploaded files for the whole channel plus metadata | one export (hours to days, many GB, comes as zip chunks) | Officially provided by Google for your own content. Allowed. Preferred for bulk. Files are named by title, so map to `video_id` via the metadata files in the archive, falling back to duration matching |
| 2 | **Creator's local source files** (the exported edits from CapCut/Premiere etc.) | low if folders are organised | Allowed, original quality. Risk: file may differ from what was published (trimmed in-app, music added with the YouTube Shorts sound library, YouTube-side edits). Match by ffprobe duration (+-0.5 s) plus a manual look at 2-3 frames. Flag mismatches and exclude |
| 3 | **YouTube Studio -> Content -> video menu -> Download** (owner, per video) | ~1 min per video, ~100 videos for DEV+GATE+HELDOUT | Allowed; YouTube itself shows the download link, which is what the ToS requires. Returns the processed MP4 (resolution can be 720p or 360p depending on the video; fine for Gemini) |
| 4 | **yt-dlp on own public videos** | scriptable | **Flag:** technically against YouTube's ToS ("don't download unless a download link is provided") even for your own channel; it is not covered by the API Developer Policies because it is not API data, but the ToS risk remains. Also fragile (needs updates). Last resort, user's call, not wired into the repo scripts |
| 5 | Gemini's native YouTube-URL input (public videos only) | none | Not used for the gate: we can't control fps/clipping exactly the same way, can't be sure no metadata leaks into the context, and the product path is uploading **unpublished drafts**, so file upload is the faithful setup. Useful only as a quick sanity check on 2-3 videos |

Recommended path: Takeout (option 1) for everything; Studio download (option 3) for stragglers. Whatever the source, the retention numbers come from the published, processed video, so prefer the closest available file to the published one; record `source` per file.

Script `src/evals/hook_test/fetch_videos.py` is **not** a downloader. It:
1. Scans a user-supplied directory (`--src ~/Takeout/...`) for `*.mp4/*.mov`.
2. Runs `ffprobe -v error -show_entries format=duration -of csv=p=0` (ffmpeg must be installed: `brew install ffmpeg`).
3. Matches each to `video_id` by duration within 0.5 s and (if Takeout metadata/title text is present) by title similarity; writes `data/evals/hook_test/video_map.csv` (`video_id,path,duration_ffprobe,match_method,source`) and lists ambiguous files for manual resolution (`--resolve video_id=path`).

Blinding step (same script or `blind.py`): for every selected video, `ffmpeg -i in -map_metadata -1 -c copy data/evals/hook_test/blind/<blind_id>.mp4` (strips title/encoder/creation-time tags; file name is the blind id). Then extract 3 frames (t=0, mid, end) into a scratch dir and eyeball them for burned-in leakage: channel handle/watermark, "Subscribe" overlays, view counters, end cards that reveal a title. A watermark that appears on **every** video is harmless; one that appears on only some is a leak (crop or exclude).

## 4. Gemini setup

Researched from ai.google.dev (video understanding + pricing pages, fetched 2026-10-01); re-verify names and prices on the day of implementation, these change often.

- **SDK:** `google-genai` (add to `requirements.txt`, pinned). `GEMINI_API_KEY` in `.env`; never log it.
- **Model:** `gemini-3.8-flash` (latest stable Flash; $0.75/M input, $3.75/M output until 2026-12-31, then $1.50/$7.50). Comparison model: `gemini-3.1-pro-preview` ($2/$12 per M, no free tier). Model id is a config constant (`config.py`), recorded in every result row.
- **Mode:** static video processing. The docs also describe an "agentic" mode for newer Flash models; not used (we want a fixed, reproducible view of the video).
- **Frames:** default is 1 fps, ~100 tokens/s at low media resolution, ~300 tokens/s at high resolution, audio ~32 tokens/s. fps is configurable through the video `processing` / `VideoMetadata` settings (docs example: `"fps": 0.5`). Median Short is ~21 s and hooks happen in the first seconds, so use **fps=5, media_resolution=high** on the gate (fast cuts, damage numbers, HUD text). Confirm the exact parameter spelling for the SDK version you pin, since the docs show a `processing` object and older SDKs use `types.VideoMetadata(fps=...)`. Keep audio in (sound design and music drops affect retention). Do not clip.
- **Ablations on DEV only (not on GATE):** fps {1, 3, 5} x resolution {low, high}; pick the cheapest setting within noise of the best (measured by DEV AUC, section 7), then freeze.
- **Upload:** Files API (`client.files.upload`) for every video: they are reused across 3 runs and ablations. Poll until state is `ACTIVE`. Files expire (~48 h), so cache `file.name` + upload time in `data/evals/hook_test/uploads.json` and re-upload when stale. Delete files after the phase (`client.files.delete`).
- **Sampling:** use the model's default temperature (Google recommends leaving Gemini 3.x temperature at 1.0; lowering it can cause looping). Reproducibility comes from repeated runs and aggregation (section 8), not from temperature 0. Set `response_mime_type="application/json"` with a `response_schema`. Keep thinking at the model default; record thinking tokens from `usage_metadata`.
- **Rate limits / retries:** sequential calls with exponential backoff on 429/5xx (3 retries); a schema-invalid response counts as a failed attempt and is retried once at most, then stored as `invalid` (never silently dropped; report the invalid rate).

### 4.1 Cost estimate

Token model: tokens ~= duration_s x fps x tokens_per_frame + 32 x duration_s (audio) + ~1.5k prompt. With the docs' ~300 tokens/s at 1 fps high-res, assume ~300 tokens per frame (conservative):

- 21 s x 5 fps x 300 = ~31.5k video tokens + ~0.7k audio + ~1.5k prompt = ~34k input tokens.
- Output ~0.6k JSON + up to ~3k thinking tokens = ~3.6k.

| Item | Per call | Calls | Total |
|---|---|---|---|
| Flash, GATE (40 videos x 3 runs) | 34k x $0.75/M + 3.6k x $3.75/M = ~$0.039 | 120 | ~$4.7 |
| Flash, DEV iteration (20 videos x ~10 batches) | ~$0.039 | 200 | ~$7.8 |
| Flash, DEV ablations (6 settings x 20 videos x 1 run, mixed fps) | ~$0.025 avg | 120 | ~$3 |
| Pro, GATE comparison (40 x 1 run) | 34k x $2/M + 3.6k x $12/M = ~$0.11 | 40 | ~$4.4 |
| Argon (intro $2/$10), GATE comparison (40 x 1 run) | 34k x $2/M + 3.6k x $10/M = ~$0.10 | 40 | ~$4.2 |
| Argon (post-intro $4/$20), GATE comparison | ~$0.21 | 40 | ~$8.4 |
| Total (with Pro or intro-Argon comparison) | | | ~$20 |

`gemini-4-argon` (introductory $2/M in, $10/M out through 2026-11-30, then $4/$20) is multimodal with video understanding and a 1M context; requires a paid-tier Gemini API key (not on the free tier). If the intro price has lapsed, a full 3-run Argon GATE (~$25) would consume nearly the whole $30 cap; run Argon as a single comparison batch only.

Argon-specific gotchas (verified against ai.google.dev, 2026-10-20):
- Reasoning is controlled by `thinking_level` (`"low" | "medium" | "high"`, default `"medium"`). `thinking_budget` / `"MINIMAL"` return HTTP 400 — different parameter name than 3.x. Thinking tokens bill as output ($10/M intro); measure `usage_metadata` on DEV before any batch. Try `"low"` first — Google recommends it for classification/extraction tasks.
- "Agentic video understanding" is GA and opt-in (default `static`). Keep it OFF for eval runs — the gate needs an identical, reproducible view of every video; adaptive frame inspection would make settings non-comparable across videos.
- 1M context is unified: input + thinking + output share it. Per-call output is capped (64K per SDK hop, "long decode" continues automatically in new SDKs) — fine for our ~4k outputs, just don't be surprised by the cap number in docs.
- Implicit prompt caching only (>=6,144 shared prefix tokens); our prompt is ~1.5k so no savings — don't build the cost model on cached-input pricing.
- Batch API (50% off) is supported; viable for the GATE since results aren't needed interactively.

Hard cap $30; the harness refuses to start a batch whose estimated cost (from the token counter, `client.models.count_tokens` on one sample video per setting) pushes the running total past the cap. Cost is tracked per call (input/output/thinking tokens x price table in `config.py`) and summed into the results JSON; this seeds the per-run cost tracking that Phase 8 needs. Using the Batch API (50% discount) is possible for the final 3-run gate but not needed at this size.

## 5. Prompt and output schema

Files: `src/evals/hook_test/prompts/hook_test_v1.txt` (+ `v2...` per DEV iteration; the final one is hash-frozen), `src/evals/hook_test/schema.py`.

### 5.1 Blinding rules for the model

- The user turn contains only: the video part, and the instruction text. No title, no description, no date, no channel name, no stats, no mention of group sizes or the 50/50 split.
- The instructions say the clip is a vertical YouTube Short from a Brawl Stars channel and that the task is to judge **viewer retention**, not entertainment value.
- Video order across calls is randomised per run (seeded) to avoid position and warm-up effects. Each call is independent (no chat history, no cross-video context).

### 5.2 Prompt skeleton (v1, derived a priori from `docs/domain-notes.md`, then tuned on DEV only)

```
You are analysing a vertical YouTube Short (Brawl Stars gameplay) to predict how well it holds viewers.
Viewers watch in a swipe feed and can leave at any moment. Retention is the share of viewers still
watching at each moment relative to how long the clip is.

Watch the whole video (picture and audio). Then:
1. Judge the first 3 seconds: is there a reason to stay (clear stake, striking visual, payoff promised)?
2. Find moments that hold attention (clutch plays, big damage numbers, ability effects, team wipes,
   game-ending screen, funny/unusual situations) and dead stretches (downtime, repetition, unclear action).
3. Estimate the single timestamp where the largest share of the remaining audience leaves
   (the main drop-off). Give it as seconds from the start, 0 <= t <= video length.
4. Give the probability that this video retains viewers better than a typical Short of the same length
   from this channel (p_strong, 0 to 1). Use the full range; 0.5 means you genuinely cannot tell.
Return only JSON matching the schema. Keep evidence short and tied to timestamps.
```

Rules for prompt iteration: edit only against DEV; every edit bumps the version file and logs DEV metrics in `data/evals/hook_test/dev_log.jsonl` (version, setting, AUC, drop-off rank). No peeking at GATE. Cap at 8 versions, then freeze the best by DEV AUC (ties: fewest tokens).

### 5.3 Response schema (Pydantic -> Gemini `response_schema`)

```python
class Moment(BaseModel):
    t_seconds: float                 # timestamp of the event
    kind: Literal["hook", "peak", "dead_zone", "drop_risk"]
    note: str                        # <= 20 words

class HookTestResult(BaseModel):
    p_strong: float                  # 0..1, P(video retains better than typical same-length Short)
    group: Literal["strong", "weak"] # model's hard call (secondary; primary uses p_strong rank)
    confidence: Literal["low", "medium", "high"]
    dropoff_timestamp_s: float       # main drop-off, seconds from start
    hook_verdict: Literal["strong", "ok", "weak"]
    video_length_s_estimate: float   # sanity check vs ffprobe duration
    moments: list[Moment]            # <= 6
    reasoning: str                   # <= 60 words
```

Post-validation (`schema.py::validate`): clamp-check `0 <= dropoff_timestamp_s <= duration + 0.5` (else mark `invalid`, do not clamp silently), `0 <= p_strong <= 1`, `abs(video_length_s_estimate - duration) <= 3` logged as a quality flag (a big miss means the model mis-saw the video, a useful diagnostic for fps choice).

## 6. Ground truth for the drop-off

Script `src/evals/hook_test/ground_truth.py` (pure functions, unit-tested).

Input: one video's 100 curve points `a_i = audienceWatchRatio` at ratio `r_i = i/100` (i=1..100; use whatever ratios the file has), and `duration D` seconds.

1. Convert to seconds `t_i = r_i * D`. One curve point spans `D/100` s (0.21 s for a 21 s Short).
2. Smooth lightly: centred 3-point moving average (edges: shrink the window). Keep the smoothed curve `s_i`.
3. Window `w = max(1, round(100 / D))` points ~= 1 second. For each start `i`, `loss_i = (s_i - s_{i+w}) / max(s_i, eps)` = relative decline over the next second. Negative means rising (loops/rewatch); keep the sign, do not clip.
4. **Ground-truth drop-off** `t* = t_i` of the maximal `loss_i`. Two versions, both stored:
   - `t*_all`: unrestricted. Often lands in the first second (mass swipe-away at the start), which is real but trivially predictable.
   - `t*_post`: restricted to `t >= 1.5 s` (after the instantaneous swipe-out). This is the "interesting" drop-off and the **primary** target.
5. **Scoring a prediction `p` (seconds):**
   - `rank_score(p)`: percentile rank of `loss` at the window containing `p` among all windows of that video (1.0 = steepest loss, 0 = steepest rise, expectation 0.5 under a uniformly random `p`). **Primary drop-off metric.** It uses the whole curve, so it is robust to curve noise picking a slightly wrong argmax.
   - `err(p) = |p - t*_post| / D`, secondary (normalised absolute error); report also `hit@1s = |p - t*_post| <= 1.0 s`.

Exclusions: videos whose curve has `max(loss) - median(loss) < 0.03` (essentially flat) are flagged `flat_curve` and excluded from drop-off scoring only (still counted in the group score). Report how many.

Caveat to state in the write-up: the curve is an aggregate of everyone who saw the video; a drop at second 7 can come from content, from the loop point, from a title/caption overlay, or from where the Shorts feed pre-loads. It is a noisy but real signal; this is why the bar is "beats random", not "hits exactly".

## 7. Scoring and the gate

Script `src/evals/hook_test/score.py` -> `data/evals/hook_test/results/<run_id>/scores.json` + `report.md`.

### 7.1 Per-video aggregation

Each video gets up to 3 runs. `p_strong_mean = mean(p_strong)`; `dropoff_pred_s` = median of the 3 predicted timestamps (median, not mean, so one outlier run does not move it). A video with 0 valid runs is `invalid`; the gate requires <=2 invalid videos of 40, otherwise fix infrastructure and re-run only those.

### 7.2 Criterion A: group prediction

- **Primary call:** rank the 40 GATE videos by `p_strong_mean`; the top 20 are predicted "strong", the other 20 "weak". Correct = matches the true label. This removes the model's calibration bias (a model that says "strong" for everything would otherwise sit at exactly 20/40 and look uninformative for the wrong reason) and is legitimate because the test set is balanced by construction. State this in the report; the product will use the continuous score vs the channel baseline.
- **Pass:** >= **26/40** correct. One-sided exact binomial under p=0.5: P(X>=26)=0.0403 (P(X>=25)=0.0769, P(X>=27)=0.0192, P(X>=28)=0.0083). Use `math.comb`, no scipy needed.
- **Also report (not gating):** the hard `group` accuracy; AUC of `p_strong_mean` vs label with a 10,000-sample permutation p-value and a bootstrap 95% CI; Spearman between `p_strong_mean` and continuous `hold_resid` (within the 40); Wilson 95% CI on the accuracy (26/40 = 65%, CI [49.5%, 77.9%]; 28/40: [54.6%, 81.9%]).

### 7.3 Baselines that must be beaten (reported next to Gemini; they are the confound check)

1. **Duration-only:** logistic/threshold on `duration_seconds` with leave-one-out cross-validation.
2. **Date-only:** same on `published_at`.
3. **Title-length / metadata:** same on title length and caps ratio (cheap proxy for "creator effort").
4. **Always-strong / coin flip:** 20/40 expected.

If any of 1-3 reaches >=26/40 on the gate set, the selection is confounded; fix the stratification (section 2) and re-draw GATE from candidates before declaring anything. Gemini passing while these also pass is not evidence for Gemini.

### 7.4 Criterion B: drop-off beats random

- **Statistic:** `T_obs = mean over scored videos of rank_score(dropoff_pred_s)` against `t*_post`-defined loss curve (section 6).
- **Null 1 (uniform random timestamp, the stated baseline):** for each video draw `p ~ Uniform(1.5 s, D)` (same support as the restricted target), compute `rank_score`, take the mean over videos; repeat 10,000 times (seeded `numpy.random.Generator`). `p_B = (1 + #{T_null >= T_obs}) / (1 + 10000)`.
- **Null 2 (prediction permutation, the honest one):** keep Gemini's predicted **ratios** `p/D`, shuffle them across videos (10,000 permutations), recompute. This controls for Gemini always predicting "~30% of the way in" or "near the start" regardless of content. B is only credited if it passes both nulls.
- **Constant baselines:** predict `0.3 D`, `0.5 D`, and the channel-median of `t*_post/D`; Gemini's `T_obs` must exceed all three (report the numbers; this is a descriptive check, no test).
- **Pass:** `T_obs >= 0.60` (a modest effect: the predicted second is on average in the top 40% steepest-loss windows) AND `p_B < 0.05` under both nulls. Also report mean `err` and `hit@1s` vs the uniform null.

### 7.5 Combined decision and the multiple-comparison caveat

A and B are two shots at the same question, so "A or B" has a false-positive rate of about `1 - (1-0.040)(1-0.05) = 8.8%` under the null. That is acceptable for a feasibility gate, but it is why the tree below treats "B only" as a conditional go that must replicate on HELDOUT, while "A" alone is a go. Never add a third criterion after seeing results.

### 7.6 Statistical caveats (to be copied into `report.md`)

- n=40 is small. Power of A (>=26/40) if true accuracy is 0.60/0.65/0.70/0.75/0.80: 32% / 57% / 81% / 95% / 99%. A true 65% model fails the gate 43% of the time, so a no-go at 22-25/40 is "no evidence", not "evidence of none".
- Labels are noisy (finite views, retention estimates, algorithmic reach differences), so the accuracy ceiling is below 100%; the extreme-quartile design raises the ceiling but also makes the task easier than the real distribution. The held-out number on the full 25/75 range at Step 3 is the one to quote externally, not this one.
- One channel, one game, one creator's style: result does not generalise to other channels (relevant to the "500 creators" goal; Phase 8 deals with it).
- Videos are not exchangeable in time; stratification reduces but does not remove era confounds.
- 3 runs per video reduce sampling noise from the model but do not add independent videos; n stays 40 for the binomial test. Do not multiply n by the number of runs.
- Picking best of several prompt versions inflates results; this is why DEV tuning is separated from GATE and GATE runs once.
- The gate is a one-sided test at ~0.04-0.05, not a measure of usefulness. Usefulness is Phase 3's job.

## 8. Repeatability

- 3 runs per video per setting, default temperature, each with a different seeded video order; store every raw response (`data/evals/hook_test/raw/<run_id>/<blind_id>__r<k>.json`) with model id, prompt SHA-256, fps, resolution, usage tokens, cost, latency, timestamp.
- Report run-to-run stability: mean pairwise Spearman of `p_strong` across the 3 runs, % of videos whose hard `group` flips between runs, and the std of `dropoff_timestamp_s` per video. If mean pairwise Spearman < 0.5, the signal is mostly noise: report it, and use 5 runs for the held-out evaluation.
- Gate uses the aggregate of all 3 runs; never select the best run.
- `run_id` = UTC timestamp + short git SHA; results are append-only (new run, new directory).

## 9. Files to create

```
src/evals/__init__.py
src/evals/hook_test/__init__.py
src/evals/hook_test/config.py          model ids, price table, fps/resolution, seeds, thresholds (0.60, 26, 0.05), paths
src/evals/hook_test/data.py            load videos.csv / video_metrics.csv / retention.csv / video_daily.csv from latest data/channel/<ts>/
src/evals/hook_test/select.py          pool funnel, residualisation, stratified strong/weak draw, DEV/GATE/HELDOUT split, balance checks
src/evals/hook_test/fetch_videos.py    map local/Takeout files to video_id (ffprobe), blind-copy with metadata stripped
src/evals/hook_test/schema.py          Pydantic models + validate()
src/evals/hook_test/gemini_client.py   Files API upload cache, generate_content wrapper, retry, usage + cost capture
src/evals/hook_test/run.py             CLI: --split dev|gate|heldout --runs 3 --model ... --fps ... --resolution ...; budget guard; writes raw responses
src/evals/hook_test/ground_truth.py    smoothing, loss windows, t*_all/t*_post, rank_score
src/evals/hook_test/stats.py           binomial sf, Wilson CI, AUC, Mann-Whitney, permutation tests, bootstrap (numpy only)
src/evals/hook_test/baselines.py       duration/date/metadata baselines, constant-timestamp baselines
src/evals/hook_test/score.py           aggregation, criteria A/B, decision, writes scores.json + report.md
src/evals/hook_test/prompts/hook_test_v1.txt (v2...)
data/evals/hook_test/                  selection.json, selection.csv, video_map.csv, blind/, uploads.json, dev_log.jsonl, raw/, results/   (gitignored under data/)
tests/test_hook_test_select.py
tests/test_hook_test_ground_truth.py
tests/test_hook_test_stats.py
tests/test_hook_test_score.py
```

Add to `requirements.txt`: `google-genai` (pinned), `pydantic` (if not already pulled in), `python-dotenv` (if not present). `numpy` already comes with pandas; do not add scipy. System dependency: `ffmpeg` (ffprobe). Commands:

```
./venv/bin/python -m src.evals.hook_test.select
./venv/bin/python -m src.evals.hook_test.fetch_videos --src ~/Takeout/YouTube...   # then blind copies
./venv/bin/python -m src.evals.hook_test.run --split dev --runs 1                  # iterate prompts
./venv/bin/python -m src.evals.hook_test.run --split gate --runs 3                 # once, prompt frozen
./venv/bin/python -m src.evals.hook_test.score --run <run_id>
```

## 10. Tests (no network, no API keys; run with `./venv/bin/python -m pytest tests/`)

- `ground_truth`: synthetic curve with a known cliff at 7.0 s of a 21 s video -> `t*_post` within 0.25 s; a monotone linear decay -> flat-curve flag; a curve with a rise (loop) keeps negative `loss`; `rank_score` of the cliff time is 1.0 and of a flat region ~0.5; windows near the end do not index out of range; D=8 and D=60 give sane `w`.
- `stats`: `binom_sf(26, 40) == pytest.approx(0.0403, abs=1e-4)`, `binom_sf(48, 80) ~ 0.0465`; Wilson CI for 26/40 matches (0.495, 0.779); AUC of perfect/random/inverted scores = 1/~0.5/0; permutation p of a perfect prediction is < 0.001 and of random predictions is uniform-ish (run 200 seeds, mean p in [0.4, 0.6]); Mann-Whitney on identical samples gives p ~ 1.
- `select`: on a synthetic 200-video frame with a planted length confound (short videos retain better) the residualised selection has |median duration diff| <= 2 s; DEV/GATE/HELDOUT are disjoint; same seed gives same output; videos published after 2025-03-01 or with >20% post-change engagedViews are excluded; videos with <1000 `engagedViews` or gapped curves are excluded; sanity filter on `relativeRetentionPerformance` is applied.
- `schema`: out-of-range `dropoff_timestamp_s` or `p_strong` is rejected as `invalid`, not clamped; extra/missing fields fail.
- `score`: given a fabricated results folder (40 videos, 3 runs) with 27 correct, decision is `GO`; with 25 correct and drop-off at chance it is `NO_GO` or `AMBIGUOUS` per the tree; invalid videos >2 raises; median/mean aggregation as specified; blinding check test: the request body assembled by `gemini_client` for a video contains no title string and the uploaded filename equals the `blind_id`.
- `gemini_client` (mocked): retry on 429, one retry on schema failure, cost computed from usage metadata; budget guard aborts when the cap would be exceeded.

## 11. Acceptance criteria (the phase is done when)

1. `selection.json` exists and is committed before any GATE call; contains pool funnel, splits (10/10, 20/20, rest), balance-test p-values (duration p>0.2, date p>0.2, |median duration diff|<=2 s), prompt hash, seed.
2. All 40 GATE videos have blind files with stripped metadata, frames spot-checked for leaks, durations matching ffprobe.
3. Final prompt frozen and tagged; DEV log shows the iteration history (<=8 versions).
4. GATE run: 3 runs x 40 videos complete, <=2 invalid videos, raw responses and costs stored; total spend within the $30 cap.
5. `report.md` states: Criterion A count + exact p + Wilson CI, AUC + CI, Criterion B `T_obs` with both null p-values, all baselines, stability numbers, flat-curve exclusions, cost, and the decision per the tree below, with caveats from 7.6.
6. Unit tests above pass.
7. Result recorded in `CLAUDE.md` Progress and `plan/README.md` (milestone M1), including the decision.

## 12. Go / no-go decision tree

Inputs: `A` = correct count of 40 (pass >=26), `B` = drop-off criterion pass (both nulls), `C` = confounds (any baseline 1-3 >=26/40, or balance tests failed).

```
C true?  -> INVALID. Fix stratification, re-draw GATE from unused candidates (HELDOUT is not touched), rerun once.
            (This re-draw is logged and the first result is kept in the report.)
else:
A >= 26                       -> GO.
                                 Proceed to Step 3 with this prompt as v0. Run HELDOUT once at end of Step 3 for the number that goes on the resume.
A in 23..25 (AMBIGUOUS):
    B pass                    -> CONDITIONAL GO. Run HELDOUT once (same frozen prompt, 3 runs, 15-20 per side).
                                 Pool GATE+HELDOUT: pooled accuracy >= 48/80 (p=0.0465) OR B passes again on HELDOUT -> GO. Else NO-GO.
    B fail                    -> run HELDOUT once; pooled accuracy must reach >= 48/80 -> GO, else NO-GO.
A <= 22:
    B pass                    -> CONDITIONAL GO (scoped): group prediction has no evidence, drop-off localisation does.
                                 Replicate B on HELDOUT; if it holds, Step 3 `analyze_video` ships as drop-off/moment analysis ONLY,
                                 with NO performance-prediction claim (the "predicted performance vs baseline" feature is dropped from the plan).
    B fail                    -> NO-GO.
```

Before concluding NO-GO from A in 23..25 or B fail, do exactly one rescue iteration, and only on DEV: try `gemini-3.1-pro-preview`, fps 5 vs 10, a prompt that gives the model the channel's own hold-pattern notes (from `docs/domain-notes.md`) or few-shot DEV examples. Any change must beat DEV baseline by a clear margin (AUC +0.1), then re-run **HELDOUT** (not GATE) as a fresh confirmatory test. Never re-use GATE after seeing its results.

### On NO-GO: pivot options (in priority order)

1. **Retrieve instead of generate (use the data we already own).** Drop "Gemini judges the video". Build baseline-driven features from the channel's own history: duration, hook-timing heuristics extracted by cheap CV/audio features (scene-cut rate in first 3 s via ffmpeg `scdet`, audio loudness onset, on-screen text/number detection), and the channel's retention-by-length curves. The copilot reports "your retention for Shorts of this length typically drops at X" from real data. Still a legitimate multimodal-lite and eval story; claims are descriptive, not predictive.
2. **Narrow the target to what Gemini can see.** If hook3s (first-3-second retention) is predicted but overall retention is not, scope `analyze_video` to hook feedback only. Re-test on HELDOUT with `hook3s` as the label; this needs no new data.
3. **Pairwise instead of pointwise.** Show Gemini two videos of similar length (one strong, one weak) and ask which retains better. Pairwise judging is often more sensitive than absolute scoring; a no-go on pointwise with positive pairwise AUC changes the product to "which of my 2-3 drafts should I post first". Test on DEV pairs, confirm on HELDOUT pairs.
4. **Pivot the product claim, keep the engineering.** The agent/MCP/deploy/security/eval parts of the project are valuable independent of the Gemini prediction accuracy. Reframe as a pre-publish checklist/copilot (trending hashtags + audio + format checks + structured moment annotations without performance claims) and keep the honest negative result as a portfolio artifact ("tested whether a multimodal LLM can predict Shorts retention; here is the evidence and what I changed"). A properly run negative eval is itself a resume-worthy result.
5. **Resurrect the parked idea** (thumbnail/title CTR model, `BACKLOG.md`) only if CTR data turns out to be obtainable; currently it isn't, so this is last.

Record the outcome either way in `CLAUDE.md` Progress and update `plan/README.md` phase table and milestone M1.

## 13. Order of work (target dates)

| Day | Task |
|---|---|
| 10-01..10-02 | Phase 1 export finished; data audit script prints pool funnel and eligible count (decide thresholds if pool is small) |
| 10-02..10-04 | Request Takeout now (it takes hours to days); meanwhile write `select.py`, `ground_truth.py`, `stats.py` + tests |
| 10-04..10-05 | Map and blind the DEV+GATE+HELDOUT files; frame leak check; commit `selection.json` |
| 10-05..10-07 | Gemini client + schema + run.py; DEV prompt iteration and fps/resolution ablation (<$12) |
| 10-08 | Freeze prompt (tag + hash); run GATE once (3 runs) |
| 10-09 | Score, baselines, report, decision (and HELDOUT per tree if ambiguous) |
| 10-10 | M1 recorded in README/CLAUDE.md |

## 14. Open items to verify while implementing

- Exact `google-genai` parameter names for fps, media resolution and the Files API state polling in the pinned SDK version (docs fetched show a `processing` object with `fps`; older SDKs use `VideoMetadata`).
- Token cost per frame at high resolution (use `count_tokens` on one real Short at fps 5 and replace the 300 tokens/frame assumption in section 4.1).
- Whether Takeout includes per-video metadata with the video id in the archive (otherwise rely on duration matching plus manual resolution).
- Whether Studio's "Viewed vs swiped away" can be read per video for the 40 GATE videos (optional secondary label).
- Current Gemini model ids and prices on the day of the run (the Flash promo price ends 2026-12-31).
