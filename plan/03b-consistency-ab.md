# Phase 3b: Code-layer changes, prompt v3, and the v2-vs-v3 consistency A/B

Status: **done (2026-11-04): v2 wins; results in section 6.** Amends plan/03a section 6 (freezing procedure). Everything here is
decided before any extraction on the training pool, and nothing in it ever looks at `hold_resid`.

## 0. Why

Three DEV runs (v0-v2) were never a clean repeat (each changed the wording), but v1 vs v2 (only the `dead_time_s`
wording changed) showed where Gemini is unstable: which payoffs it lists and how big it rates them (`payoff_count`
4/10), audio fields (6-8/10), and one perception miss (`vcdb916f3`, the audio gag) that moved 8 fields. A likely root
cause is that the free-text `timeline` breaks the video down differently on every run. Prompt v3 tests a fixed
breakdown. Ten DEV videos cannot tell two versions apart (9/10 vs 10/10 is chance), so the comparison runs on 30
training-pool videos, and the winning version's runs double as the section 6.2 consistency check.

Amendment to 03a section 6.1 (logged): v3 is a third prompt iteration, beyond the cap of two. Reason: a structural
change to the scratchpad, chosen from run-to-run agreement only, with no outcome data seen. **v3 is the last prompt
change.** Whatever fails after the A/B is handled by code substitutes or dropped, not by a fourth version.

## 1. Code layer (applies to both v2 and v3 outputs, no prompt change)

All of it lives in `src/tools/analyze_video/derive.py` (Gemini output + code features in, model features out) and
`measure.py`. Gemini's raw answers are always kept next to the derived values.

| # | Change | Rule |
|---|---|---|
| 1.1 | Clear payoff count | `payoff_count` (core) = payoffs with weight ≥ 2. `payoff_count_all` (full) = all payoffs |
| 1.2 | Tail, two versions | `tail_all` = duration − last payoff; `tail_clear` = duration − last weight ≥ 2 payoff (falls back to the last payoff of any weight, then to duration). The A/B picks one for the core (section 4.3) |
| 1.3 | `ending` from the tail | Gemini's outro_or_black / result_screen_hold / cliffhanger are kept; otherwise ≤ 1.0 s cut_on_payoff, ≤ 3.0 s short_tail, else long_tail, from the selected tail version. Already built (`resolve_ending`) |
| 1.4 | Opening (option A) | `opening_frame` stays Gemini's; the override `resolve_opening_frame` is removed. New code features from the first 0.5 s, decoded at 10 fps: `open_first_frame_black` (frame 0 mean luma below the G5 threshold), `open_motion` (median mean-abs frame difference / 255), `open_luma` (mean luma / 255). They replace `open_black` / `open_still` in `code_features.csv`; existing columns stay byte-identical |
| 1.5 | Payoff snapping, two versions | Save scene-cut timestamps per blinded file (`data/features/cuts/<blind_id>.json`, same ffmpeg threshold as G2; see 1.11). `t_s_snapped` = nearest cut within 0.5 s for kinds punchline, reveal, spectacle; else unchanged. Every timing feature is computed raw and snapped; the A/B picks one (4.3) |
| 1.6 | Main payoff | Unchanged: the weight-3 payoff (v2 and v3 both keep "exactly one weight 3"); `main_payoff_frac` = t / duration |
| 1.7 | `arc` | Unchanged: Gemini's answer (10/10 stable; deriving it from weights would add payoff-list noise) |
| 1.8 | Audio timing | Already decided: G8 `t_loudness_peak_s` replaces `t_audio_change_s` as a feature (03a E3) |
| 1.9 | Pinning | Checked 2026-11-04 (ai.google.dev model page and "What's new in Gemini 3.8 Flash"): `gemini-3.8-flash` is the **stable** id (no dated variant listed), default thinking level `medium` (levels low/medium/high; minimal errors). The extractor sets `thinking_level="medium"` explicitly (the hook test keeps its pre-registered config untouched) and records `resp.model_version` in every raw record. Pricing note: introductory $0.75/$3.75 per M tokens until 2026-12-31, $1.50/$7.50 from 2027-01-01, so the full extraction should run before then |
| 1.11 | Duration | Every per-call duration (prompt, windows, tail, fractions) is the ffprobe duration of the **blinded file Gemini sees**, not the export's rounded `duration_seconds` (e.g. `v952137ad`: export 12, file 11.3 s). Scene cuts are measured on the same blinded file (`data/features/cuts/<blind_id>.json`) |
| 1.10 | Load limit | Every ffmpeg or Gemini job runs with at most 2 workers |

## 2. Prompt v3 (only the timeline changes)

v3 = v2 word for word, except `timeline`:

- Code renders the window list into the prompt: `0.0-1.0, 1.0-2.0, …`, the last window ending at the ffprobe
  duration (N = ceil(duration) windows). **No scene-cut hints** (they would bias Gemini towards cuts and make its
  features echo `cuts_per_s`).
- `timeline` becomes a list with exactly one entry per window, in order: `{start_s, end_s, seen, heard}`.
  `seen` = what is on screen and what happens; `heard` = speech (quote key lines), music, sound effects.
  If nothing new happens, write `no change`. Describe, never evaluate.
- Payoffs and every other field are filled from the timeline, as in v2; payoff `t_s` keeps 0.1 s precision.
- Validator: entry count = N and each `start_s` / `end_s` within 0.05 s of the given window, otherwise the record is
  invalid (one automatic retry, as now). No `max_length` on the timeline array: the API rejects `maxItems` = 61 there with 400 INVALID_ARGUMENT (found on
  the first v3 call, 2026-11-04, isolated with text-only probe calls; no v3 record existed yet). The validator's exact
  window-count check enforces the length instead.

## 3. The A/B experiment

- **Videos:** 30 from the `train` split of `src/models/pool.py` (never DEV, GATE or HELDOUT), stratified into duration
  thirds (10 each), sampled with seed 340. Selection written to `data/features/gemini/consistency_selection.csv`.
- **Duplicates (amendment 2026-11-04, before any call on these videos):** the first draw contained the same upload
  twice ("TOP 5 MELODIE SAD PINS", `v2a54836c` / `vda204d7d`). The whole eligible pool (all four splits) is scanned
  for duplicates by frame hashes, and duplicates are removed by a fixed, outcome-blind rule: within a group keep the
  earliest-published upload; a train video that duplicates a DEV/GATE/HELDOUT video is dropped from train (the
  evaluation sets are frozen and never edited; duplicates between them are only reported). Exclusions live in
  `data/features/duplicates_exclude.json` and `src.models.pool.train_rows` applies them. The 30 are then re-drawn
  with the same seeds on the de-duplicated train pool. The same exclusions apply to model training later (a
  duplicate across CV folds or in the H1 neighbour set would leak).
- **Blinding:** the training-pool videos have no blinded copies yet (only the 110 hook-test videos do). Create them
  with the hook test's blinding (metadata stripped, random ids) before any call; the contact-sheet leakage check is
  repeated on these 30.
- **Runs:** v2 run 1, v2 run 2, v3 run 1, v3 run 2. Same model, settings and uploads. 120 calls.
- **Cost:** v2 about $0.032 a video (measured), v3 estimated $0.035-0.045 (longer output). About **$4-5** total; hard cap
  **$6** (abort before exceeding it).
- **Gemini-only:** nothing here uses outcomes. `hold_resid` is not loaded by any script in this phase.

## 4. Scoring (pre-registered)

### 4.1 Agreement statistics (per version, run 1 vs run 2, over the 30 videos)
- Enums and bools: Cohen's κ; ordinal (`arc_ord`, `loop_seam`): weighted κ (linear).
- Counts and seconds: ICC(2,1) (absolute agreement), plus median |Δ| for timestamps.
- 95 % bootstrap CIs over videos (2,000 resamples, seed 340) for every statistic.
- Computed on the **derived features the model uses**, plus every raw field for the full model.

### 4.2 Core features scored
`t_premise_clear_s`, `payoff_count` (clear), `main_payoff_frac`, `tail` (selected version), `dead_time_s`, `arc_ord`,
`loop_seam`. (`cuts_per_s` is code, always identical.) Thresholds from 03a 6.2: κ ≥ 0.6, ICC ≥ 0.7, timestamps median
|Δt| ≤ 1.0 s.

**Correction (2026-11-04, after the runs):** the committed scoring script (`ab_score.py`, commit fa305e9) scores the
main payoff as the timestamp `t_main_payoff_s` with the timestamp rule, not `main_payoff_frac` with ICC as listed
above. The script is what ran; section 6 reports the decision both ways (it is the same).

### 4.3 Decisions, in this order
1. **Validity guard for v3** (checked first; v3 loses if any fails): invalid rate after retry ≤ 10 %; mean clear payoff
   count within ±25 % of v2's; a read of 5 videos' v3 timelines (one per duration band + 2 longest) finds no systematic
   filler (`no change` on windows where something visibly happens) or invented audio.
   **Laziness check** (10 long-band videos, v3 both runs): compare the last third of the windows with the first third.
   v3 is "lazy" if the `no change` share in the last third is ≥ 20 percentage points above the first third, or if
   more than 1 in 5 of the payoffs v2 found in the last third of the video (t ≥ 2/3 of the duration, either v2 run)
   have no v3 payoff (either v3 run) within 1.0 s. (Made precise 2026-11-04, before any A/B call: "matching event"
   = a v3 payoff; a free-text match against the timeline would almost always succeed with fixed windows.) Lazy triggers the fallback in 4.4 instead of disqualifying v3.
2. **v2 vs v3:** the version with more core features passing their threshold wins. Tie: higher mean of the core
   agreement statistics (κ and ICC values; timestamps scored as the share of videos with |Δt| ≤ 1.0 s), if it leads by
   ≥ 0.05. Otherwise **v2** (incumbent, simpler).
3. **Within the winner:** snapped vs raw timing, and `tail_all` vs `tail_clear`: keep the version with the higher ICC on
   `tail` and the higher share of `main_payoff` |Δt| ≤ 1.0 s; ties go to raw / `tail_all`.
4. **Consistency gate (03a 6.2) on the winner's two runs:** keep, substitute (C6 → G4, D2 → G6) or drop each feature by
   the thresholds. Prevalence rule (6.2b) after that.

### 4.4 Fallback if v3 is lazy (pre-registered, not a new iteration)
Only for videos longer than 30 s: cut the video into two halves overlapping by 3 s (frame-exact re-encode), get the
v3 timeline for each half (windows numbered on the full-video clock), join them in code (the overlap's windows are
taken from the first half), then fill every other field with one **text-only** call (joined timeline + v3 field
definitions, no video). Cost about 1.1-1.2× of v3 for those videos. Tested with two runs on the 10 long-band videos
only (about $1, inside the $6 cap); then scored with 4.1-4.3 on the 30 videos (short and middle bands keep the plain
v3 runs). Videos of 30 s or less always use plain v3.

The scoring script is written by the lead before any run and committed first, so the rules cannot drift after seeing
results.

## 5. Steps (one at a time, each reviewed before the next)

1. This plan approved.
2. Pinning check (1.9): documented default thinking level and any dated model id.
3. Code layer (section 1) + tests; `code_features.csv` columns 1.4 and cut files 1.5, ≤ 2 workers.
4. Prompt v3 + schema + validator (section 2) + tests; one dry-run render of the prompt for a 12 s and a 60 s video.
5. Blinded copies + selection of the 30 (section 3), with the leakage check.
6. Runner: `--split train` (selection file only), `--runs 2`, `--prompt v2|v3`; still refuses gate / heldout.
7. Scoring script (section 4) written and committed before any training-pool call.
8. Dry run (cost estimate), then the four runs.
9. Report: the 4.3 decisions with CIs, then freeze (03a 6.3) and the full extraction (about $8, asked first).

## 6. Results (2026-11-04)

Runs: v2 `20261104T224114Z_fa305e9` ($1.90), v3 `20261104T231214Z_fa305e9` ($2.27; resumed after the maxItems fix,
commit 0d6b26d). 120/120 records valid, `model_version` `gemini-3.8-flash` throughout. Report:
`data/features/gemini/ab_report/20261104T224114Z_fa305e9__20261104T231214Z_fa305e9.{md,json}`.

**v3 guard and laziness:** passed. Invalid rate 0 %; mean clear payoffs 1.32 vs 1.32; manual read of v01452c0c,
vbbbcc779, vc320954c, vda204d7d, vdae4c4e0 (run 1) found careful entries to the last window, `no change` only on
genuinely static stretches, no invented audio (read = yes). `no change` share first third 0.0 %, last third 0.6 %;
25/25 late v2 payoffs matched by v3.

**Core agreement (run 1 vs run 2, 30 videos; 95 % bootstrap CIs in the report):**

| Feature | v2 | v3 |
|---|---|---|
| `t_premise_clear_s` (share ≤ 1 s) | 0.90 pass | 0.87 pass |
| `payoff_count`, clear (ICC) | 0.29 fail | 0.55 fail |
| `t_main_payoff_s` (share ≤ 1 s) | 0.93 pass | 0.97 pass |
| `tail_all` (ICC) | 0.99 pass | 1.00 pass |
| `dead_time_s` (ICC) | 0.77 pass (CI 0.15-0.99) | 0.46 fail |
| `arc_ord` (weighted κ) | 0.78 pass | 0.73 pass |
| `loop_seam` (weighted κ) | 0.53 fail | 0.59 fail |

**Decision: v2.** As committed (main payoff by timestamp): v2 5 passes vs v3 4. As the 4.2 text read
(`main_payoff_frac` by ICC): 4 vs 4, mean lead of v3 = 0.0499 < 0.05, so the incumbent v2 wins. v2's
`main_payoff_frac` ICC (0.56) is driven by one video (`v2a54836c`, main payoff 20.0 s vs 2.2 s); without it, 0.99.
The A/B found no meaningful difference between the versions; v2 is the simpler and 17 % cheaper one.

**Variants (4.3 step 3, v2):** snapping changes nothing (share 0.93 raw and snapped) → raw. `tail_all` ICC 0.99 vs
`tail_clear` 0.72 → `tail_all`.

**Gate (4.3 step 4, v2):**
- Core keep: `t_premise_clear_s`, `t_main_payoff_s` / `main_payoff_frac`, `tail_all`, `dead_time_s` (fragile: wide CI),
  `arc_ord`. Substitute: `loop_seam` → G6 `first_last_dhash_dist` (also in the `× is_short` interaction).
- `payoff_count` (clear, ICC 0.29) fails. **Amendment (2026-11-04, creator's call, outcome-blind):** the core uses
  `payoff_count_all` instead (ICC 0.91 in v2, 0.94 in v3), i.e. back to the original 03a C1 definition. Chosen by
  run-to-run agreement only, like the tail variant; `hold_resid` was not looked at. The clear count stays computed for
  reporting. The weight 1 vs 2 call is what is unstable; the number of payoffs Gemini lists is not.
- Full model drops: `rewatch_trigger` (κ 0.36), `premise_legible_nonplayer` (0.32), `info_new_per_5s` (0.45),
  `audio_visual_sync` (0.55), `beat_before_main_payoff` (0.57), `twist` (0.60, just under). All other full fields pass.
- Next: prevalence rule (03a 6.2b) on the training pool after the full extraction, then freeze (03a 6.3).
