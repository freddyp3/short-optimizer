# Phase 6: `audience_panel` (persona panel), gated by an experiment

**Status:** not started. **Depends on:** phase 1 (demographics, geography, comments), phase 3 (eval dataset,
harness, metrics, `analyze_video`). **Ships only if** the experiment in section 4 says it beats a single-model
baseline. If it loses, it is cut and documented as a negative result (section 9). Both outcomes are a
portfolio win; the experiment is the deliverable, the tool is conditional.

## 1. What the tool would do

`audience_panel(video, analysis?) -> PanelReport`: a panel of N simulated viewers, each a persona grounded in
the channel's real audience, reacts to a draft Short. Output:

- per-persona reaction: `would_swipe_at_s` (timestamp or null), `swipe_reason`, `would_like/share/comment`, one-line comment they would leave, 1-5 hook score;
- aggregate: predicted drop-off timestamp(s), predicted strong/weak score (a number in [0,1]), disagreement across personas (a spread, shown as uncertainty), top recurring complaints/praise;
- cost + trace metadata (per-persona tokens, latency).

It is a **tool the orchestrator may call** (phase 5), not a pipeline step. Orchestrator sees the aggregate only.

## 2. Research evidence (honest summary: mixed)

Searched 2026-10-01. I have read abstracts/snippets, not every paper in full; verify before citing on a resume.

- **Persona-prompted agents have modest but real signal.** A study of social-media reaction prediction
  (120K+ agent-persona combinations from 1,511 humans, 27 LLMs) reports ~70.7% overall accuracy, a 13-point spread
  by LLM choice, better on entertainment/lifestyle than news/politics (our domain, gaming, is closer to the
  former), and better on positive than negative reactions. The same line of work reports it does **not** beat a
  conventional supervised text classifier. Sources: [arXiv 2604.19787](https://arxiv.org/pdf/2604.19787),
  [Scientific Reports](https://www.nature.com/articles/s41598-026-66277-8).
- **Homogeneity / variance collapse.** "Silicon sampling" papers find simulated populations are more orderly than
  humans: effective sample size is smaller than the nominal N, between-group variance shrinks, same persona gives
  the same answer repeatedly, and subgroup differences are compressed. Sources:
  [arXiv 2510.11408](https://arxiv.org/pdf/2510.11408),
  [arXiv 2604.06663](https://arxiv.org/html/2604.06663v1) (audience-segmentation approach to restoring
  heterogeneity), [arXiv 2606.30085](https://arxiv.org/html/2606.30085v1).
- **LLM-generated personas carry bias** toward stereotyped/positive/idealised portrayals and the effect leaks into
  downstream simulation results ([NeurIPS 2025 position paper, "LLM Generated Persona is a Promise with a Catch"](https://papers.neurips.cc/paper_files/paper/2025/file/5fc150b75e8b509735564de16bc3527b-Paper-Position_Paper_Track.pdf)).
- **Benchmarks for crowd-level reaction** exist ([ConsumerSimBench, arXiv 2605.17079](https://arxiv.org/html/2605.17079v1)),
  none that I found target video retention curves for Shorts. Nothing here says personas help on *video*
  drop-off prediction; that is the gap this experiment tests.

**Prior, stated before running anything:** I expect a null or small effect on strong-vs-weak classification
(the information is in the video, personas add framing and noise), and a better chance of an effect on
*qualitative* usefulness, which this experiment does not measure (section 8). Expected P(ship) ~ 25%.

## 3. Hypotheses (pre-registered; written before any panel run)

Primary metric = AUC for separating strong from weak Shorts (definition of labels from phase 3). Let
`P` = panel, `B2` = best single-call baseline (below).

- **H1 (primary):** AUC(P) > AUC(B2). Null: no difference.
- **H2 (drop-off):** panel's aggregated drop-off timestamp has lower median absolute error vs the real
  retention-curve drop-off than B2's single predicted timestamp.
- **H3 (diversity matters):** P beats B3 (one call told to imagine 12 diverse viewers). If P ~ B3 the separate
  calls/sampling machinery is not paying for itself and B3 is the thing to ship if anything.
- **H4 (grounding matters):** P-grounded (demographic + comment grounded) beats P-generic (same N, same
  prompts, generic "YouTube Shorts viewer" personas of random ages/countries). If not, the grounding claim in
  the README is unsupported and must not be made.
- **H5 (cost):** if H1 holds, cost multiple (panel $/video / B2 $/video) is within the cap in section 6.

H1 and H2 are co-primary; I use Holm correction across them (alpha 0.05 overall). H3-H5 are secondary and only
reported with CIs.

## 4. The experiment

### 4.1 Eval set and split

Source: the phase-3 eval dataset (past Shorts with real outcomes; ~277 Shorts total, fewer with usable
retention because YouTube withholds low-view rows).

- **Labels:** strong/weak by the phase-3 definition (retention or swipe rate relative to channel baseline,
  top vs bottom group, middle excluded for the classification metric). Continuous target for AUC-free
  rank correlation: the baseline-relative residual.
- **Split (fixed before any run, seeded, saved to `data/evals/panel_split.json`):**
  - `dev` (about 40%): all prompt/persona-recipe iteration happens here.
  - `test` (about 60%, all unseen): **one** frozen run per arm at the end. No prompt edits after.
  - Phase-2's 40 videos are a subset of dev if they were already used for prompt iteration; check that
    phase 3 did not also tune on test. If it did, hold out the newest Shorts by date instead.
- Need **>= 60 labelled test videos** for the decision to mean anything (see power, 4.7). If fewer, the
  verdict can only be "inconclusive, cut for scope".

### 4.2 Arms (what is compared)

All arms use the same video, the same label, the same model family for the final judgement, and the same output
schema (a 0-1 `strong_score`, a `dropoff_s` timestamp, 1-sentence rationale).

| Arm | Description | Why it exists |
|---|---|---|
| **B0** | One Gemini call, video, no personas, no channel info | the dumbest baseline, = the phase-2/3 `analyze_video` prediction |
| **B1** | B0 + a text summary of channel demographics (age x gender, top countries) and top comment themes in the prompt | tests "just tell the model who the audience is" |
| **B2** | One Claude call on the `analyze_video` structured output (text, no video) + the B1 audience summary | **the matched control**: exactly the information the panel personas get, without personas |
| **B3** | Same inputs as B2, one call, "imagine 12 viewers from this distribution, react as each, then give your overall estimate" | the one-call panel; isolates separate sampling from the persona framing |
| **B4** | B0 repeated K times at temperature > 0, scores averaged (K matched to panel cost) | compute-matched baseline: panel might just be self-consistency |
| **P-generic** | Panel with N generic personas (random age/country, no comment themes) | ablates grounding (H4) |
| **P** | Panel with grounded personas (section 5), N personas, R repeats | the treatment |
| **P-video** *(optional)* | As P but each persona sees the raw video via Gemini | tests whether personas need pixels; expensive, run on a 40-video subset only |

**B2 is the baseline for the decision rule** (the strongest honest control). B0 is reported as context. If P beats
B0 but not B2, the value is the extra info, not personas, and the verdict is "don't ship".

### 4.3 What each persona sees

Default (P): the `analyze_video` output (timestamped hook/pacing/moments, phase 3 schema, text only) plus the
draft's title/hashtags. Rationale: one Gemini video call per Short, not N; persona calls are cheap text calls;
persona reaction is then a function of (persona, description), which is what the experiment isolates. Risk:
the panel can only react to what the describer noticed, so the panel is bounded by `analyze_video` quality.
Mitigation: P-video on a subset to measure how much is lost.

Anti-leak rules: the analysis step must not see titles/stats of eval videos (already required by phase 2);
persona prompts must not contain the video's own comments, view counts or any outcome field.

### 4.4 Metrics

On the **test** split, per arm:

1. **Group accuracy**: strong-vs-weak at the threshold chosen on dev (fixed; no test-time tuning), reported as
   accuracy and balanced accuracy. Compare to 50% and to the phase-2 gate (>= 26/40 equivalent, ~65%).
2. **AUC** of `strong_score` vs the strong/weak label (primary). Also Spearman vs the continuous
   baseline-relative residual.
3. **Drop-off error**: for each video, the real drop-off is the timestamp (as a fraction of runtime, converted
   to seconds) of the steepest one-step fall in `audienceWatchRatio` within the retention curve
   (phase-3 defines this precisely; reuse it, do not redefine it here). Report median absolute error in seconds
   and the share within +/-2 s. Compare against two reference predictors: (a) uniform-random timestamp, (b)
   "always the median real drop-off time of the dev set". The model must beat (b) to count, not just (a).
4. **Calibration of disagreement** (secondary): does panel spread correlate with |error|? If not, the panel's
   uncertainty is decoration.
5. **Cost and latency** per video (from `src/obs/` tracing; tokens in/out per call, wall-clock).

### 4.5 Aggregation of panel outputs into a prediction

Defined here, **frozen before test**, but a small search over variants on dev is allowed:

- `strong_score`: mean of persona `hook_score`s weighted by persona weight `w_i` (the persona's sampled
  share), rescaled to [0,1]. Variants tried on dev: unweighted mean, median, fraction of personas who "watch to
  the end".
- `dropoff_s`: for each persona the stated swipe time (null = watched fully). Predicted drop-off = the
  weighted mode of a 1-second histogram of swipe times (ties broken earliest); if > 50% of weight is null,
  predict "no major drop". Variant: swipe-time survival curve S(t) over personas, drop-off = steepest fall of
  S(t). The survival-curve version also gives a **predicted retention curve** that can be compared to the real
  curve (RMSE), a richer metric than a single timestamp.
- Pick the aggregation on dev by the primary metric, then freeze. Report all variants in an appendix to avoid
  hiding the garden of forking paths.

### 4.6 Runs and noise

LLMs are stochastic and panel outputs are noisy, so noise gets measured, not ignored:

- Each arm is run **R = 3** independent times on the test set (temperature as per arm config). Per-video score
  is the mean over runs; run-to-run SD per video is stored too. Report between-run SD of the arm-level AUC.
- Panel size N = 12 default. Dev sweep over N in {4, 8, 12, 24} with the aggregate score vs N plotted: the
  literature's variance-collapse finding predicts a plateau at small N, and a flat curve means N=4 is enough
  (cost cut of 3x).
- Seeds: persona sampling seed per video is derived from `hash(video_id, run_idx)` so reruns are reproducible
  and personas differ across runs. Panel for B-arms is not sampled.

### 4.7 Decision rule (written now, applied once)

Unit of resampling is the **video** (paired comparison; panel and B2 scored on the same videos).
Paired bootstrap, 10,000 resamples of test videos, for `Delta = metric(P) - metric(B2)`.

**Ship `audience_panel` iff ALL of:**

1. **Effect size:** Delta-AUC >= 0.05 (e.g. 0.60 vs 0.55), or drop-off median abs error reduced by >= 1.0 s
   (about 10-15% of a ~21 s Short... confirm against the phase-3 error scale before the run).
2. **Confidence:** the 95% bootstrap CI for Delta excludes 0 on at least one co-primary metric (Holm-adjusted
   across the two), and the point estimate has the right sign on the other.
3. **Not just noise:** Delta exceeds 2x the between-run SD of the arm AUC.
4. **Beats the compute-matched baseline:** P also beats B4 (point estimate positive, 90% CI lower bound > 0).
5. **Cost:** panel cost per video <= **5x** B2 cost **and** <= an absolute budget (target < $0.15/video at
   expected pricing; the product budget is decided in phase 8). If the effect is large (Delta-AUC >= 0.10), the
   multiple may stretch to 10x.
6. **Grounding claim:** the README may only say "grounded in your real audience" if H4 also holds (P beats
   P-generic with CI lower bound > 0). If H1 holds and H4 fails, ship as "reaction simulator", without the claim.

**Power check, honest version:** with ~60 test videos (30 strong / 30 weak), the SE of an AUC is roughly 0.07;
the paired SE of a difference is somewhat smaller but a true Delta of 0.05 is **not reliably detectable**. So:
(a) the decision rule can only confirm a big effect; (b) a CI that includes both 0 and 0.05 is reported as
"inconclusive" and **resolves to cut** (never "ship on a hunch"); (c) before running, compute the minimum
detectable effect from the dev-set paired variance and write it in the results doc. If the dataset is
larger than expected (middle group included with the continuous target, ~150+ videos) use the continuous
metric (Spearman) as primary to gain power.

### 4.8 Compute and cost estimate

Placeholders for price per million tokens must be filled from the pricing pages at run time (do not trust
this table's dollar numbers; they exist to size the order of magnitude).

Assumptions: test set ~80 videos, dev ~60, ~21 s median Short; `analyze_video` output ~1.5k tokens; persona
prompt ~1.2k tokens in (persona card + rubric + analysis) and ~300 out; Gemini video call ~21 s x
~300 tokens/s (high fps, audio) ~ 6-10k tokens in.

| Arm | Calls / video | ~Tokens in / out per video |
|---|---|---|
| B0 | 1 Gemini video | 8k / 1k |
| B1 | 1 Gemini video | 9k / 1k |
| B2 | 1 Claude text (analysis reused) | 3k / 0.5k |
| B3 | 1 Claude text | 3k / 1.5k |
| B4 | K=5 Gemini video | 40k / 5k |
| P (N=12) | 12 Claude text | 12 x (3k / 0.3k) = 36k / 3.6k |
| P-generic | 12 Claude text | same as P |

Notes: `analyze_video` output is computed **once per video** and cached (`data/evals/cache/`), shared by
B2/B3/P. Prompt caching on the shared prefix (rubric + analysis) cuts the P input cost substantially, since only the
persona card differs; measure this, it can make N=12 closer to 3-4x one call than 12x. Sketch of the run
count: (dev + test ~140 videos) x (7 arms) x (R=3) ~ 3k arm-runs, dominated by P (12 calls each ~ 15k panel
calls total) and B4. Expected total, order of magnitude **tens of dollars, not hundreds**; set a hard budget
cap in the runner (`--max-usd`) that aborts the run and logs the spend. Dev sweeps (N, aggregation variants)
add roughly the same again; budget **~$100 total** for the whole phase and decide the exact figure after
a pilot on 10 videos.

## 5. Persona construction

Goal: personas that are **grounded** (distribution matches real audience data) but not **caricatures**
(not "teen boy who loves Brawl Stars and says 'lit'").

### 5.1 Data inputs (from phase 1, `data/channel/<ts>/`)

- `demographics.csv`: viewerPercentage by `ageGroup x gender`, `creatorContentType` split; **use the SHORTS rows**.
- `geography.csv`: by `country` (+ `cities.csv` top 250), Shorts rows.
- `devices.csv`, `subscribed_status.csv` (subscriber vs non-subscriber), `traffic_sources.csv` (how they arrive: Shorts feed vs search etc.).
- `comments.csv`: top-level comments and replies, per video.
- Known limits to state in the output docs: gender reported is male/female/user-specified only; thresholds hide
  small groups; demographics are signed-in viewers only; **no joint distribution** of age x gender x country
  exists in the API (we have age x gender, and country marginally).

### 5.2 Persona sampling from marginals

1. Sample `(age_group, gender)` from the joint `viewerPercentage` table (this joint is real).
2. Sample `country` from the geography marginal, **assuming independence** from age/gender. Flag this in
   code and docs; the independence assumption is wrong in places (e.g. country skew by age) and is the best
   available. Tested as a sensitivity (shuffle country labels, see ablations).
3. Sample `viewer_type` from `subscribed_status` (subscriber vs not) and `entry_path` from Shorts traffic sources (feed, search, sound page, ...).
4. Sample `device` (phone vs tablet/TV) from `devices.csv`.
5. Sample a **taste profile** (below) from the comment-derived distribution, *conditionally independent of
   demographics* (no comment-level demographics exist; commenters' ages are unknown). This is stated as a
   limitation, not hidden.
6. Weights: each persona carries `w = 1/N`; for stratified sampling (better variance than iid at N=12) use
   quota sampling by (age x gender) cell with proportional allocation and iid for the rest, rounding the
   largest-remainder way. Report the realised vs target marginal in the trace.

Stratified sampling is chosen over "pick 12 archetypes" so the panel is a sample of the audience, not a list of
characters the author imagined.

### 5.3 Comment-theme clustering

Offline, once per export (`personas/build_themes.py`), never per run:

1. Load comments; drop channel-owner replies, empties, pure-emoji spam, bot/promo patterns, comments under a
   minimum length, **and the comments of any eval video** when building the eval-time themes
   (see leakage below).
2. Embed comments (a cheap embedding model), cluster (HDBSCAN or k-means, k chosen by silhouette; target 6-12
   clusters), then have Claude name each cluster and write 2-3 *paraphrased* example comments (never verbatim
   usernames or ids; keeps PII out of prompts and keeps content short).
3. Each theme record: `{theme_id, label, share (of comments), typical_voice: short style note,
   what_they_praise, what_they_complain_about, example_paraphrases}`. Typical themes to expect (hypotheses, the
   clustering decides): "wants meta/tier-list takes", "brawler main loyalty", "skill criticism / 'that was
   luck'", "humour/meme replies", "requests for a specific brawler/mode", "asks about settings or build".
4. Human check: the creator spot-reads each cluster label and example (30 min). This is the one place his domain
   intuition is used, and it is logged as `themes_reviewed_by_creator: true`.

Persona taste profile = a theme (sampled by `share`) + a **sampled** mix of 2-3 sampled attributes (play
frequency, trophy-level bucket, favourite mode, patience with intros) from a small hand-written
attribute table. Attribute tables are explicit data files, not LLM-invented, so the persona distribution is
inspectable.

### 5.4 Keeping personas grounded and not stereotyped

- **Structured persona card, short.** JSON: demographics, viewer_type, entry_path, theme, 2-3 sampled
  attributes, `attention_budget` (seconds of patience, sampled lognormal, calibrated to the channel's real
  retention: median fraction watched). The card is data, not a paragraph of backstory; long backstories
  increase caricature and cost.
- **Do not say "you are a 14-year-old boy, act like one".** The prompt tells the model it is *estimating how a
  viewer with these attributes behaves in a Shorts feed*, third person, with the explicit instruction that
  demographics are a weak prior and individual viewers vary; ask for behaviour (swipe time, reason) before
  opinion, and ban stereotype-flavoured language in the output schema (reason is a constrained enum +
  <= 15 words free text).
- **Behavioural calibration, not just demographic.** `attention_budget` and `entry_path` are calibrated so the
  panel's simulated aggregate retention curve (on the dev set average) matches the channel's mean real
  retention curve at the 1-, 3-, 5-, 10-second marks (within +/- 5 points). A panel that cannot reproduce the
  *average* Short cannot discriminate between Shorts; this is a cheap sanity gate before any strong/weak
  eval.
- **Heterogeneity checks (acceptance tests, section 7):** within-panel variance in swipe times on a fixed
  video > a floor; persona-pair agreement not near 1.0; swapping the persona card changes outputs (a
  "persona sensitivity" test: same video, persona A vs B, differences vs repeating A).
- **Demographic-flip test (bias probe):** same video, flip only gender or age bucket; large systematic
  swing in scores from demographics alone is flagged as stereotype leakage and the persona field is removed
  from the card if it carries no behavioural signal on dev (ablation: drop-one-field).
- **No sensitive-attribute inference about real people.** Cards describe aggregate audience cells; no real
  commenter's username/text goes into a prompt.

### 5.5 Leakage rules (these protect the experiment's validity)

- Eval-time themes are built from comments on videos **excluding the eval videos** (and for temporal honesty
  prefer comments from videos published **before** the eval video where the dataset size allows).
  Comments on an eval video are partly an outcome of the video; using them is cheating.
- Demographics are channel-wide Shorts aggregates and so include the eval videos' viewers. Impact is
  small (one video among ~277) but non-zero; state it. Per-video demographics are **not** used in
  persona construction.
- Persona-card sampling never sees labels or retention of the target video. Calibration of
  `attention_budget` uses dev-split average retention only.

## 6. Cost multiple, in the decision rule

Defined in 4.7(5): <= 5x B2 and <= the absolute product budget; up to 10x for a large effect. In production the
tool is optional and off by default for cheap runs; the orchestrator calls it when the user asks "how will
audiences react". Latency target: panel p95 < 15 s via parallel calls (asyncio, N concurrent), concurrency-capped
to avoid rate limits.

## 7. File layout, tests, acceptance criteria

```
src/tools/audience_panel/
  __init__.py            exports audience_panel()
  schema.py              pydantic: PersonaCard, PersonaReaction, PanelReport, ThemeRecord
  audience.py            load demographics/geography/devices/etc from data/channel/<ts>/ into an AudienceModel (marginals)
  sampling.py            stratified sampling of PersonaCards from AudienceModel (seeded)
  themes.py              comment cleaning, embedding, clustering, theme naming (offline build)
  attributes.py          loads data/personas/attributes.yaml (hand-written attribute tables)
  prompts.py             persona-reaction prompt, B-arm prompts (kept together so arms share wording)
  react.py               one persona -> PersonaReaction (async, structured output, retries, cost logging)
  aggregate.py           PersonaReaction[] -> PanelReport (weighted mean, swipe histogram, survival curve)
  tool.py                MCP/agent-facing wrapper, input validation, budget guard
  README.md              what it is, how to rebuild themes, the verdict
src/evals/panel/
  arms.py                B0..B4, P, P-generic, P-video runners with a common interface
  run_panel_eval.py      CLI: --split dev|test --arms ... --runs 3 --max-usd 100
  stats.py               paired bootstrap, Holm, AUC, MDE calc
  decision.py            applies the section 4.7 rule to a results file, prints SHIP/CUT/INCONCLUSIVE
data/personas/           gitignored: themes.json, audience_model.json, attributes.yaml
data/evals/panel_split.json, data/evals/panel_runs/<ts>/   gitignored; results.json, per-video scores, traces
docs/negative-results/ or docs/panel-experiment.md          written either way (section 9)
tests/test_audience_panel/
  test_audience.py       marginals sum to ~1, SHORTS rows used, handles missing/withheld rows
  test_sampling.py       seeded determinism; realised marginals within tolerance of target at N=1000; stratified N=12 quotas
  test_themes.py         cleaning drops owner/spam/eval-video comments; no usernames/ids in theme output (PII test)
  test_leakage.py        eval-time theme build excludes eval-video comments; prompts contain no outcome fields (grep for stats keys)
  test_react.py          fake LLM client: schema validation, retry on bad JSON, null swipe handled
  test_aggregate.py      hand-computed swipe-histogram and survival-curve cases, weights, all-null -> "no drop"
  test_stats.py          bootstrap CI on a synthetic known-effect dataset, Holm
  test_decision.py       table-driven: each clause of the rule can independently flip the verdict
  test_heterogeneity.py  persona-sensitivity and demographic-flip probes (fake LLM + one optional live marked slow)
```

Live-API tests are marked `@pytest.mark.live` and skipped by default. Everything else runs offline with fake
clients, per project convention.

**Acceptance criteria**

- *Build gate (before the experiment):* personas sample reproducibly; realised marginals match audience data
  (max abs error < 2 points at N=1000); retention-calibration check passes on dev (+/- 5 points at 1/3/5/10 s);
  no leakage tests failing; the one-video smoke run costs under budget.
- *Experiment gate:* split file frozen and committed; hypotheses and decision rule committed (this document at
  that git SHA); test run executed once with all arms; `decision.py` output saved with the raw per-video scores.
- *Ship gate:* verdict SHIP per 4.7 **and** `audience_panel` passes tool-contract tests against the
  orchestrator's tool schema (phase 5) and the MCP schema (phase 7).
- *Cut gate:* verdict CUT or INCONCLUSIVE, code moved to `experiments/audience_panel/` (or left behind a
  feature flag, default off), orchestrator unchanged, writeup done.

## 8. What the experiment does not measure

- **Qualitative usefulness.** A panel could *not* predict retention better and still give a creator
  better-phrased feedback. That is a different claim and a different experiment (blind creator preference test
  on ~20 videos; 30 minutes of the creator's time). Not part of the ship gate because a good-sounding
  explanation is exactly what LLMs fake best. If someone wants to ship on usefulness, that has to be a
  separate pre-registered test with the creator as judge, plus the new-Shorts study from the master plan.
- **Out-of-channel generalisation.** Personas are calibrated to @alfie369k's audience; other creators'
  audiences (the 500+ creators goal) would need their own OAuth demographics, which the product may not have.
  So even a SHIP result is a result for this channel, until validated on another channel.

## 9. If it loses

1. Do not ship it; do not soften the rule after seeing the numbers.
2. Write `docs/panel-experiment.md`: hypotheses, arms, split, results table with CIs, cost table, the
   persona-sensitivity and heterogeneity diagnostics, and what the failure suggests (e.g. "persona output
   collapsed: effective N of ~3 at nominal N=12, consistent with the silicon-sampling literature").
3. Keep the framework: `src/evals/panel/` (arms + stats + decision) is reusable for any future "does
   component X beat a single-call baseline" question; the orchestrator and `analyze_video` evals can use it
   (this ties into phase 3 and 5 evals).
4. Resume/portfolio framing (only measured numbers): "Pre-registered a paired-bootstrap A/B of a 12-persona
   audience simulator vs matched single-call and compute-matched baselines on N held-out Shorts; found no
   (or inconclusive) improvement (dAUC = x, 95% CI [a, b]) at k-times the cost, so cut it." A clean negative
   result with honest stats is the interview story: eval-driven scoping, resisting a flashy feature.
5. Update `plan/README.md` and `CLAUDE.md` Phase 6 line: "evaluated, cut, see docs/panel-experiment.md";
   the architecture diagram drops the box; MCP phase unaffected.
6. Optional cheap fallback worth keeping if B1 beat B0: the "audience summary in the prompt" (demographics +
   comment themes) as a plain prompt block in `analyze_video`/orchestrator. That is the actual product
   value if grounding helps but personas do not.

## 10. Dependencies and order of work

- **Phase 1 (hard):** `demographics.csv`, `geography.csv`, `devices.csv`, `subscribed_status.csv`,
  `traffic_sources.csv`, `comments.csv`, and `retention.csv` (for calibration and drop-off truth). Needs the first
  successful signed-in export. If comments are sparse (low-engagement Shorts), theme clustering may yield
  fewer clusters; that is fine, record counts.
- **Phase 3 (hard):** eval dataset with labels + split machinery, `analyze_video` (structured text output used by
  B2/B3/P), harness conventions, retention-curve drop-off definition, tracing/cost helpers (`src/obs/`).
- **Phase 5 (soft):** only needed at ship time (tool contract). Experiment can run before the orchestrator exists.
- **Phase 2 (soft):** if the feasibility gate fails (Gemini cannot predict strong/weak from video at all), the
  panel experiment is **moot** because personas get a description of a video the describer cannot judge;
  skip phase 6 unless the failure was in the prompt rather than the premise.
- **Dependencies (pip):** `anthropic`, `google-genai` (already planned), `pydantic`, `numpy`, `scipy`,
  `scikit-learn` (AUC, clustering), `hdbscan` or sklearn's `HDBSCAN`, an embedding model client; all behind
  the existing `requirements.txt`.

Build order inside phase 6: (1) audience model + sampler + tests; (2) themes build + creator review;
(3) B0-B4 arms (they are cheap and give the baseline numbers even if the panel is never built); (4) panel
react/aggregate + calibration check on dev; (5) dev sweeps (N, aggregation, prompts); (6) freeze and run test
once; (7) decision, writeup, then ship or cut.
