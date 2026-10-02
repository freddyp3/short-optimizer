# Phase 1: Channel Export (runbook + design record)

Part of the [master plan](README.md). Code: `src/channel_export/`. Module README: `src/channel_export/README.md`
(Google Cloud setup steps live there; this doc does not repeat them).

Verification legend: **[verified]** = read in code or checked on disk on 2026-10-01. **[unverified]** = could not be
checked without a live run against the YouTube APIs.

## 1. Goal

Pull every piece of own-channel data the YouTube APIs expose for @alfie369k into `data/channel/`, so that:

- Phase 2 (hook test) can pick 20 strongest / 20 weakest Shorts by retention or swipe rate and score drop-off timestamps
  against real retention curves.
- Phase 3 (eval harness) can build the dataset of past Shorts + real outcomes, relative to the channel baseline.
- Phase 7 (MCP server) can serve channel data as tools.

It is a one-channel, read-only export. Output is raw JSON (audit trail) plus flat CSVs plus a manifest.

## 2. Current state (as of 2026-10-01)

| Item | State |
|---|---|
| Code | Written. 22 tests pass (`./venv/bin/python -m pytest tests/`) **[verified]**. Never run against live APIs **[unverified]**. |
| OAuth client | `secrets/client_secret.json` exists, type `installed` (Desktop app), project `short-optimizer`, redirect `http://localhost` **[verified]**. |
| Second client file | `secrets/client_secret_313669972049-…apps.googleusercontent.com.json` also exists, same project and client-id prefix but different bytes **[verified]**. `client_secret.json` wins (explicit default path is checked first, discovery is only a fallback). Whether the two differ because of a rotated secret is **[unverified]**; if sign-in gives `invalid_client`, try the other file via `--client-secret`. |
| Token | None: `secrets/token.json` does not exist **[verified]**. |
| `data/` | Empty **[verified]**. |
| Git hygiene | `data/`, `secrets/`, `client_secret*.json`, `token*.json`, `.env` are all in `.gitignore` **[verified]**. |
| Google Cloud side | Three APIs enabled, consent screen set to External/Testing, channel-owning account added as test user: **[unverified]** (cannot be checked from the repo). Confirm before step 1. |
| Dependencies | `requirements.txt` has google-api-python-client, google-auth, google-auth-oauthlib, pandas **[verified]**. |

Channel facts from `CLAUDE.md`: 277 Shorts, mostly Mar 2024 to Mar 2025. Whether the channel also has long-form uploads
is **[unverified]** (the export will tell you via `content_type`).

## 3. Runbook

All commands from the repo root. Order matters: the Reporting API step is time-sensitive.

### Step 0: preflight (2 min)

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m pytest tests/            # expect 22 passed
ls secrets/client_secret.json                  # must exist
```

In Google Cloud Console, confirm: YouTube Data API v3, YouTube Analytics API, YouTube Reporting API are enabled; the
consent screen has your account under Test users.

### Step 1: create Reporting jobs (do first, same day as sign-in)

```bash
./venv/bin/python -m src.channel_export --create-reporting-jobs --reporting-only
```

Why first: a new Reporting job only backfills about 30 days of history and generates files from creation onward, so every
day of delay is a day of bulk-report data lost forever **[module docstring claim; Google behavior unverified live]**.
This is also the first browser sign-in, so it doubles as the OAuth test.

Sign-in checklist (browser opens, `prompt=consent`, so a refresh token is always issued):

1. Pick the Google account that is a test user.
2. Brand Account: on the identity chooser pick the **channel** (alfie369k), not the personal profile.
3. "Google hasn't verified this app": Continue.
4. **Tick every permission checkbox** (youtube.readonly and yt-analytics.readonly; both are required).
5. Token is written to `secrets/token.json` (chmod 600).

Expected result: log lines `Created reporting job <id> for <type>` for each of the 9 types, and a
`data/channel/<UTC ts>/manifest.json` with `reporting_jobs.created` populated. Types the channel is not offered land in
`reporting_jobs.unavailable` (not an error). Re-running is idempotent: existing jobs go to `reporting_jobs.existing`.

The 9 report types **[verified, reporting.py]**: `channel_reach_basic_a1`, `channel_reach_combined_a1`,
`channel_basic_a3`, `channel_combined_a3`, `channel_traffic_source_a3`, `channel_device_os_a3`,
`channel_playback_location_a3`, `channel_demographics_a1`, `channel_sharing_service_a2`.

Note: `--reporting-only` still creates a timestamped run directory with a (mostly empty) manifest and the table CSVs that
exist (core CSVs are skipped in reporting-only mode).

### Step 2: smoke test (3 newest uploads, no comments)

```bash
./venv/bin/python -m src.channel_export --max-videos 3 --skip-comments
```

Check the new `data/channel/<ts>/manifest.json`:

- `error_count` is 0 or only explainable errors (see section 6).
- `channel.title` is the channel, not your personal profile.
- `videos.exported` is 3; `retention.csv` has 3 x 100 rows (or fewer if a video is below the privacy threshold).
- `analytics_reports.video_lifetime.ok` is 1.

Caveat: `video_lifetime` and the channel-level reports always cover the whole channel, regardless of `--max-videos`.
Only per-video reports (daily, retention, traffic) are limited to the 3 videos.

### Step 3: full export

```bash
./venv/bin/python -m src.channel_export
```

Whole channel history from the channel's `publishedAt` date, all uploads, comments included. Progress logs every 25
videos. Rough call count: Analytics about `30 + 2 x videos + batched traffic + one detail query per (video, referrer
type)`; for 277 videos that is several hundred requests. Data API: `3 + videos/50 + comment pages` units, so tens to low
hundreds of units out of 10,000/day **[estimate from README formulas, not measured]**. Runtime: not measured
**[unverified]**; the per-video loops are sequential, expect minutes, not seconds.

Use `--since YYYY-MM-DD` (e.g. `--since 2024-03-01`) to shrink the range if needed. Do not use `--monetary` unless
revenue is wanted (adds a scope and a second consent, needs YPP, and nothing in phases 2 to 8 consumes it).

### Step 4: download Reporting files (about 48h after step 1)

```bash
./venv/bin/python -m src.channel_export --download-reports --reporting-only
```

Files land in the persistent archive `data/channel/reporting/<report type>/<start>_<end>_<reportId>.csv`. Already-archived
files are skipped, so it is safe to run as often as you like. First files typically appear about 24 to 48h after job
creation **[Google behavior, unverified here]**.

### Step 5: refresh cadence

| What | When | Why |
|---|---|---|
| `--download-reports --reporting-only` | every 2 to 3 weeks, at minimum before 30 days pass | Files expire after 30-60 days; the archive is the only long-term copy. |
| Full export | before each eval/hook-test run that needs fresh retention, and at least every 30 days | Developer Policies: stored Data API metadata must be refreshed or deleted within 30 days. Analytics lags 48-72h. |
| Re-sign-in | every 7 days if the app is in Testing mode | Refresh tokens expire after 7 days; the tool detects `RefreshError` and reopens the browser. Run an interactive command, not a cron job. |

Each full export creates a new `data/channel/<ts>/` folder (nothing is overwritten). Phases 2 and 3 should point at a
specific run folder (or a `latest` symlink, see section 8).

## 4. What lands on disk

```
data/channel/
  <UTC ts>/                  one folder per run (e.g. 20261001T201500Z)
    raw/                     untouched JSON of every API response; errors as *.error.json
    manifest.json
    channel.csv  videos.csv  video_metrics.csv  video_daily.csv  retention.csv
    traffic_sources.csv  traffic_source_details.csv  demographics.csv  geography.csv
    devices.csv  subscribed_status.csv  sharing_services.csv  playback_locations.csv
    channel_traffic_sources.csv  channel_daily.csv  cities.csv  comments.csv
    (video_revenue.csv  revenue_daily.csv  ad_types.csv   only with --monetary)
  reporting/<report type>/<start>_<end>_<reportId>.csv    persistent archive, shared across runs
```

CORE tables (`channel, videos, video_metrics, video_daily, retention, traffic_sources, demographics, geography, devices`)
are always written, possibly empty (header-less if empty, since an empty DataFrame has no columns to write; treat a 0-row
file as "no data"). Extra tables appear only if at least one report returned rows or columns.

Column names below are from `analytics.py`, `flatten.py`, `data_api.py` **[verified from code]**. Any column the API fails
to return is simply absent.

### 4.1 Analytics metric groups (reused below)

- `BASIC_METRICS` = `engagedViews, views, estimatedMinutesWatched, averageViewDuration, averageViewPercentage, redViews,
  estimatedRedMinutesWatched, likes, dislikes, comments, shares, subscribersGained, subscribersLost,
  videosAddedToPlaylists, videosRemovedFromPlaylists`
- `INSIGHT_METRICS` = `engagedViews, views, estimatedMinutesWatched`
- `RETENTION_METRICS` = `audienceWatchRatio, relativeRetentionPerformance, startedWatching, stoppedWatching,
  totalSegmentImpressions`

Types: INTEGER columns become pandas nullable `Int64`; FLOAT/CURRENCY columns become `float64`.

### 4.2 Tables

| File | Grain | Columns |
|---|---|---|
| `channel.csv` | 1 row | Flattened channel resource, dotted names: `id`, `snippet.title`, `snippet.customUrl`, `snippet.publishedAt`, `statistics.subscriberCount/viewCount/videoCount`, `contentDetails.relatedPlaylists.uploads`, `brandingSettings.*`, `status.*`, `topicDetails.*`, `localizations.*`. Lists/dicts are JSON-encoded in the cell. |
| `videos.csv` | 1 per upload | Front columns: `video_id`, `title`, `published_at`, `content_type`, `is_short`, `duration_seconds`; then every flattened part column: `id`, `snippet.*` (title, description, tags, categoryId, publishedAt...), `contentDetails.duration/definition/...`, `statistics.viewCount/likeCount/commentCount`, `status.*`, `topicDetails.*`, `recordingDetails.*`, `fileDetails.*`, `processingDetails.*`, `suggestions.*`, `localizations.*`, `player.*`, `paidProductPlacementDetails.*`, `brandPartner.*`. Parts the API rejects are dropped and listed in `manifest.videos.dropped_video_parts`. `content_type` is Analytics `creatorContentType`; `is_short` is boolean, empty when Analytics had no row for the video (e.g. zero views in range). `duration_seconds` is parsed from ISO-8601. |
| `video_metrics.csv` | per video x `creatorContentType` | `video_id`, `creatorContentType`, then BASIC_METRICS. Lifetime totals over the export date range. A video normally has one row; if several, `content_type` in `videos.csv` takes the one with most views. |
| `video_daily.csv` | per video x day | `video_id`, `day`, BASIC_METRICS. Starts at publish date minus 1 day. |
| `retention.csv` | per video x 100 points | `video_id`, `elapsedVideoTimeRatio` (0.01 to 1.00 in 100 steps), RETENTION_METRICS. This is the Phase 2/3 ground truth. `audienceWatchRatio` is the fraction of viewers present at that point (can exceed 1 for replays); `relativeRetentionPerformance` is vs YouTube videos of similar length; `startedWatching`/`stoppedWatching` are the swipe-away signal. |
| `traffic_sources.csv` | per video x `insightTrafficSourceType` | `video_id`, `insightTrafficSourceType` (SHORTS, YT_SEARCH, HASHTAGS, SOUND_PAGE, VIDEO_REMIXES, ...), INSIGHT_METRICS. Batched query with per-video fallback. |
| `traffic_source_details.csv` | per (scope) x referrer | `report` (`channel_traffic_detail_<TYPE>` or `video_traffic_detail`), optional `video_id`, `insightTrafficSourceType`, `insightTrafficSourceDetail`, INSIGHT_METRICS. Top 25 by views. Per-video detail only for types in `TRAFFIC_DETAIL_TYPES` with `views > 0`. |
| `demographics.csv` | channel-wide | `report` (`demographics` or `demographics_by_content_type`), `ageGroup`, `gender`, `viewerPercentage`, and `creatorContentType` (`ALL` for the unsplit report). |
| `geography.csv` | channel-wide | `report`, `country`, BASIC_METRICS, `creatorContentType`. |
| `devices.csv` | channel-wide | `report`, `deviceType`, `operatingSystem`, INSIGHT_METRICS, `creatorContentType`. |
| `subscribed_status.csv` | channel-wide | `report`, `subscribedStatus`, view metrics + likes/dislikes/shares/playlist adds-removes, `creatorContentType`. |
| `sharing_services.csv` | channel-wide | `report`, `sharingService`, `shares`, `creatorContentType`. |
| `playback_locations.csv` | channel-wide | `report`, `insightPlaybackLocationType`, INSIGHT_METRICS, `creatorContentType`. |
| `channel_traffic_sources.csv` | channel-wide | `report`, `insightTrafficSourceType`, INSIGHT_METRICS, `creatorContentType`; plus `video_id` = null for detail-filtered rows. |
| `channel_daily.csv` | channel-wide x day | `report`, `day`, BASIC_METRICS, `creatorContentType`. |
| `cities.csv` | top 250 | `report`, `city`, `country`, view metrics; data from 2022-01-01 only. |
| `comments.csv` | per comment/reply | `comment_id, video_id, thread_id, parent_id, is_reply, author_display_name, author_channel_id, text_display, text_original, like_count, published_at, updated_at, viewer_rating, can_rate, total_reply_count, is_public, can_reply`. Top-level and replies. Videos with comments disabled are skipped silently. |

The channel-level tables carry a `report` label column; the unsplit report has `creatorContentType = ALL`
(filled in by `combine()`), the split one has real values (`SHORTS`, `VIDEO_ON_DEMAND`, ...).

### 4.3 manifest.json keys **[verified, export.py]**

`tool, version, started_at, finished_at, elapsed_seconds, output_dir, options, channel{id,title,customUrl,publishedAt},
date_range{start,end}, videos{uploads_found,exported,dropped_video_parts}, reporting_jobs, reporting_downloads, steps[],
analytics_reports{<name>:{ok,failed,rows}}, csv_row_counts{<file>:n}, error_count, errors[], quota{...}, raw_files_written`.
Each error carries the step name and, for Analytics, the exact query params.

### 4.4 Reporting files

`data/channel/reporting/<report type>/<start>_<end>_<reportId>.csv` (gzip is auto-decompressed). Column schemas are
Google's, not ours; read the header of each file. Of most interest: `channel_reach_*` (impressions and impressions CTR; the
only source of thumbnail CTR, which feeds the parked thumbnail/title idea in `BACKLOG.md`, not phases 2 to 8). Exact
columns **[unverified until files exist]**.

## 5. Data-quality checks (run after the full export)

Run against the newest run folder. These are the checks, not yet scripts; a small `scripts/check_export.py` is a good
Phase 1 add-on if desired (see section 7, optional).

```python
import pandas as pd, json, pathlib
run = sorted(pathlib.Path("data/channel").glob("20*"))[-1]
m = json.load(open(run / "manifest.json"))
v = pd.read_csv(run / "videos.csv"); r = pd.read_csv(run / "retention.csv")
vm = pd.read_csv(run / "video_metrics.csv"); tr = pd.read_csv(run / "traffic_sources.csv")
```

| # | Check | How | Pass condition |
|---|---|---|---|
| 1 | Right channel | `m["channel"]["title"]`, `customUrl` | @alfie369k |
| 2 | Manifest errors | `m["error_count"]`, `m["errors"]` | 0, or only entries explained in section 6 |
| 3 | Row counts vs 277 Shorts | `v.is_short.sum()`, `len(v)`, `m["videos"]` | `uploads_found == exported`; `is_short == True` count about 277 (exact match not guaranteed: private/unlisted/deleted videos or zero-view videos with empty `is_short` change it). Investigate any gap above ~5. |
| 4 | Shorts vs long-form | `v.content_type.value_counts(dropna=False)` and `v.duration_seconds` | Shorts are `SHORTS`; cross-check that no `SHORTS` has `duration_seconds` > 180 and no `VIDEO_ON_DEMAND` has < 60 (Shorts max length was raised to 3 min in 2024). Rows with NaN `content_type` need a duration-based fallback (see section 8). |
| 5 | Retention present per Shorts video | `r.groupby("video_id").size()`; compare set of `v[v.is_short].video_id` to `r.video_id.unique()` | Each present video has exactly 100 rows. List videos with 0 rows: expected for very-low-view videos (privacy threshold), but they must not be in the Phase 2 pool. Target: retention for >= 90% of Shorts; Phase 2 only needs 40. |
| 6 | Retention sanity | `r.audienceWatchRatio` | Mostly in [0, ~2]; value at ratio 0.01 near 1.0 for most videos; no all-NaN curves. |
| 7 | `engagedViews` availability | `vm.engagedViews.notna().mean()`, `vm.groupby("creatorContentType")` | Column exists and is populated for Shorts rows. If the column is absent or all-NaN, the API rejected it (check `manifest.errors` for `video_lifetime`), and the cross-2025-03-31 comparison rule in the master plan cannot be followed. Also compare `views` vs `engagedViews` on post-2025-03-31 Shorts (`views >= engagedViews` expected). |
| 8 | Data API vs Analytics views | `v["statistics.viewCount"]` vs `vm.views` (SHORTS) | Same order of magnitude; Analytics lags 48-72h. |
| 9 | Date coverage | `m["date_range"]`, `video_daily.day.min()` | Starts at channel `publishedAt`; ends within 3 days of today. |
| 10 | Traffic sources | `tr.insightTrafficSourceType.value_counts()` | `SHORTS` (swipe feed) dominant for Shorts; no mass failures in `analytics_reports.traffic_sources.failed`. |
| 11 | Comments | `m["csv_row_counts"]["comments.csv"]`, `m["errors"]` for step `comments` | Non-zero; no quota errors. |
| 12 | Reporting jobs | `m["reporting_jobs"]` (step 1 run) | 9 created/existing or documented in `unavailable`. |
| 13 | Raw store | `m["raw_files_written"]` > 0 | Raw JSON retained for re-flattening without new quota. |

Also decide and write down: the **definition of the Phase 2 ranking metric** (e.g. `averageViewPercentage`, or swipe rate
approximated as `stoppedWatching` in the first retention segments relative to `startedWatching`). Which of these is
populated for the sample is only knowable from check 5/6. Swipe rate is not a named API metric **[verified: not in
`analytics.py`]**; it must be derived.

## 6. Known failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `No YouTube channel is linked to the account you signed in with…` | Signed in with the personal profile instead of the Brand Account channel | `rm secrets/token.json`, rerun, pick the **channel** on the identity chooser. |
| 403 `insufficientPermissions` / "Request had insufficient authentication scopes" on some or all reports | A permission checkbox was left unticked, or token lacks a scope | `rm secrets/token.json`, rerun, tick every box. `get_credentials` also re-runs sign-in automatically when the cached token lacks required scopes. |
| 403 on `--monetary` reports | Channel not in YPP, or monetary scope not granted | Skip `--monetary` (not needed). |
| `Access blocked: app has not completed verification` / `access_denied` | Account not on the consent screen's Test users list | Add it under Google Auth platform > Audience > Test users. |
| `redirect_uri_mismatch` | OAuth client is a Web application, not Desktop | Create a Desktop app client (the tool warns when it sees `web`). |
| `invalid_client` / `unauthorized_client` | Wrong/rotated client secret file; two client JSONs exist in `secrets/` | Try the other file with `--client-secret PATH`; keep only one. |
| Log says "Could not refresh the saved token… 'Testing'" | 7-day refresh token expiry in Testing mode | Expected: browser sign-in reopens, click through. Do not schedule unattended runs while in Testing. Moving to "In production" removes the limit but may require app verification **[unverified for these scopes]**. |
| `API has not been used in project … or it is disabled` (403 `accessNotConfigured`) | One of the three APIs not enabled | Enable it in the Cloud Console Library; rerun. |
| `quotaExceeded` / `dailyLimitExceeded` | Daily project quota used up (Data API 10,000/day default; Analytics has its own quota) | The exporter detects it, stops that API family and records it; rerun after the daily reset (midnight Pacific). Raw responses already saved are not re-fetched automatically, so rerun is a full run. Use `--skip-comments` or `--since` to cut cost. |
| Rate-limit 403/429/5xx | Transient | Auto-retried (up to `max_retries=5`, statuses 429/500/502/503/504 and rate-limit 403 reasons). |
| First 5 videos of a per-video report all fail | Bad query/scope | Fail-fast guard skips the rest of that family and logs it in `manifest.errors`; fix cause, rerun. |
| Empty retention/traffic-detail for a video | Privacy threshold on small audiences | Expected; exclude from eval pool. |
| `videos.list` drops parts (`dropped_video_parts`) | Owner-only/unsupported parts rejected | Non-fatal; remaining parts are exported. |
| No Reporting files after 48h | Jobs not created, wrong channel, or report type unavailable | Check `reporting_jobs` in the step-1 manifest; rerun `--create-reporting-jobs --reporting-only`; confirm `jobs.list` output in `raw/reporting/`. |
| Reporting files missing for dates >30-60 days ago | Never generated (backfill ~30 days) or expired before download | Unrecoverable; this is why step 1 goes first. |

## 7. Acceptance criteria (definition of done)

Phase 1 is done when all hold:

1. `secrets/token.json` exists and a rerun succeeds without a code change.
2. A full export (no `--max-videos`) completes; `manifest.error_count == 0` or every error is classified in section 6 as
   non-blocking and noted in the run log.
3. `videos.csv` covers all uploads (`uploads_found == exported`) with about 277 Shorts identified by `is_short`/`content_type`;
   every Shorts/long-form discrepancy is explained.
4. `retention.csv` has a 100-point curve for >= 90% of Shorts, and at minimum for 40+ Shorts with enough views to rank
   (Phase 2 needs 20 strongest / 20 weakest plus a held-out set).
5. `video_metrics.csv` has `engagedViews` populated for Shorts.
6. `comments.csv` non-empty.
7. Nine Reporting jobs exist (or `unavailable` is documented), and at least one `--download-reports` run has archived files
   (can complete ~48h after the rest; does not block Phase 2).
8. No secrets or data tracked by git (`git status` shows nothing under `data/` or `secrets/`).
9. Tests still pass: `./venv/bin/python -m pytest tests/` (22).
10. This doc's "[unverified]" marks updated with real observations (runtime, call counts, dropped parts, report columns).

Optional, small: `scripts/check_export.py` implementing the section 5 checks and printing a pass/fail table. Not required
for done; the snippet suffices.

## 8. Handoff outputs

Consumed by **Phase 2 (hook test)** and **Phase 3 (eval harness)**. Recommended conventions to establish at the end of
Phase 1 (these are not implemented yet; do them by hand or in a loader function in Phase 3):

- **Pointer to latest run:** `data/channel/latest -> <UTC ts>` symlink (not created by the exporter **[verified]**), so
  downstream code never hardcodes a timestamp. Alternatively pick `sorted(glob("data/channel/20*"))[-1]`.
- **Shorts universe:** `videos.csv` where `is_short == True` (fallback for NaN `content_type`: `duration_seconds <= 180`,
  flagged as inferred). Key everywhere is `video_id`.
- **Outcome columns for ranking:**
  - retention curve: `retention.csv` (`audienceWatchRatio` by `elapsedVideoTimeRatio`; convert ratio to seconds with
    `duration_seconds` to get drop-off timestamps),
  - swipe-ish signal: `startedWatching` / `stoppedWatching` (derived),
  - summary: `video_metrics.csv` `averageViewPercentage`, `averageViewDuration`, `engagedViews`.
- **Baseline for "relative to channel":** channel medians over Shorts of the above, computed from `video_metrics.csv`
  (use `engagedViews`, not `views`, for anything spanning 2025-03-31; also consider restricting baselines to
  comparable publish periods).
- **Video files for Gemini:** the export does **not** download video files **[verified: no download code]**. Phase 2 needs
  the 40 Shorts as actual video files from the creator's own copies or YouTube Studio downloads; `videos.csv` supplies
  `video_id`, `title`, `duration_seconds` to match them. Titles/stats must be hidden from Gemini in Phase 2.
- **Comments and demographics:** `comments.csv`, `demographics.csv` (`creatorContentType == SHORTS`) feed Phase 6
  (persona panel, gated) and any later prompt grounding; Phase 7 (MCP) exposes the tables read-only.
- **Reporting archive:** `data/channel/reporting/` for any later reach/CTR work (parked per `BACKLOG.md`).
- **Held-out set:** decide the split at handoff time (e.g. by hashed `video_id`) and record it in a committed file, since
  `data/` is gitignored. Not yet done.

## 9. Open items to confirm during the first live run

- Exact runtime and call counts for the full export.
- Whether `engagedViews` is accepted by every report it is requested on (the analytics module claims all combinations were
  checked against docs on 2026-09-25; no live confirmation).
- Whether `creatorContentType` as a dimension on the Top-videos report returns the expected rows (code comments say it
  is documented only as a dimension).
- Which `unavailable` Reporting types apply to this channel, and the real column headers of the reach reports.
- Whether the two client-secret JSON files differ meaningfully.
