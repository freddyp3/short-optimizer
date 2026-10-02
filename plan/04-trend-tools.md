# 04: Trend Tools (`trending_hashtags`, `trending_audio`)

Phase 4 of the master plan ([README.md](README.md)). Depends on: nothing from phases 2/3 except the shared
Gemini client (for the optional audio style match). Feeds: [05-orchestrator-agent.md](05-orchestrator-agent.md),
[07-mcp-server.md](07-mcp-server.md).

Research date: 2026-10-01. Anything I could not confirm is collected in "Unverified / open questions" at the end
and marked **[UNVERIFIED]** inline.

---

## 0. Summary of decisions

| Topic | Decision |
|---|---|
| Recent-upload source | `playlistItems.list` on each watchlist channel's uploads playlist, then `videos.list` (batched 50 ids). **Never `search.list`.** |
| Auth | Public data, API key only (`YOUTUBE_API_KEY` in `.env`). No OAuth for other channels. |
| Storage | Nothing from other channels is written to disk by default. In-memory TTL cache (6 h). Optional disk cache is off by default, hard TTL 24 h. Both are far under the 30-day limit. |
| "Trending" | Recency-weighted hashtag frequency across the watchlist, with a per-channel cap and a min-distinct-channels floor. Plus a coarse `rising / steady / fading` momentum label. |
| Per-channel relative performance | **Off by default** (`perf_weighted` mode). It is a custom ratio over other creators' channels, which is the exact thing the derived-metrics policy restricts. |
| Output | Only normalized hashtag tokens and counts. No channel names, video ids, titles, or per-channel numbers leave the tool. |
| Audio chart | No official API or export found. Ingest by manual CSV/JSON drop. No scraping. |
| Audio matching | There is no legal automated Short -> sound mapping. Tool returns chart rows + user-supplied sounds, optionally ranked by a Gemini audio-style comparison against the draft. It never claims a sound is "trending on Shorts like yours". |

---

## 1. Policy research (what I fetched)

Sources fetched 2026-10-01 (summaries came through a fetch tool that paraphrases/extracts; re-read the live pages
before launch, see open questions).

### 1.1 Quota (developers.google.com/youtube/v3/determine_quota_cost)

- "Projects that enable the YouTube Data API have a default quota allocation of 100 `search.list` calls, 100
  `videos.insert` calls, and 10,000 units per day combined for all other endpoints."
- "The `search.list` and `videos.insert` methods have their own quota buckets. Each of these methods has a default
  daily limit of 100 per day."
- `channels.list`, `playlistItems.list`, `videos.list`: 1 unit per call each.
- Daily quota resets at midnight Pacific Time.
- **Confirmed:** the 100/day `search.list` bucket exists in the current docs. **[UNVERIFIED]:** the "since June 2026"
  date (the page did not state when it changed). It does not matter for this design because we avoid `search.list`.
- Quota is per Google Cloud project, so the channel exporter, this tool, and every public user share the same
  10,000 units/day. Design for that (shared cache, section 2.6).

### 1.2 Developer Policies (developers.google.com/youtube/terms/developer-policies)

As returned by the fetch:

- **III.E.4 (storage):** "API Clients may temporarily store limited amounts of Non-Authorized Data for as long as is
  necessary for the purposes of the API Client but not longer than 30 calendar days." Other channels' public data
  fetched with an API key is Non-Authorized Data. Own-channel OAuth data is Authorized Data with its own rules
  (not this doc's concern).
- **III.E.2 (aggregation):** "Do not aggregate API Data or otherwise use API Data or YouTube API Services to gain
  insights into YouTube's usage, revenue, or any other aspects of YouTube's business."
- **III.E.6 (scraping):** API clients "must not ... scrape YouTube Applications or Google Applications, or obtain
  scraped YouTube data or content."
- Compliance guide (developer-policies-guide), "don't" list: aggregating/compiling authorized API data unless it is
  viewable only to the channel owner; "Gamify channel performance by ranking or tracking views between different
  channels, or generally stoking creator rivalries"; "Calculate and assign custom 'scores' to channels based on
  independently calculated averages or ratios."
- **Derived metrics policy** (developers.google.com/youtube/terms/derived-metrics-policy, introduced 2026-05-04,
  updated 2026-06-01): developers are generally prohibited from creating metrics that replace or modify API data;
  custom channel scores, ratios, leaderboards, etc. are allowed only for audited developers with an approved
  analytics use case (via the quota extension form). Approved clients may store statistical metrics up to 36 months;
  titles, comment text etc. stay on the 30-day refresh/deletion rule.

### 1.3 What this means for the design (the risk map)

| Action | Risk | Design response |
|---|---|---|
| Fetch other channels' recent uploads via public API | Fine (this is the intended API use) | Official endpoints only, API key |
| Store them | OK <=30 days, but unnecessary | In-memory TTL cache, 6 h. No default disk persistence |
| Count hashtags across channels | Grey. Closest rule is III.E.2 "aggregate API Data ... to gain insights into YouTube's usage". Our purpose is a creator-side hashtag suggestion, not insight into YouTube's business, and nothing is stored or exposed as a dataset | Ephemeral; output tokens + counts only; shown only to the requesting user; documented purpose. **[UNVERIFIED]** whether Google would read it this way |
| Score/rank/compare channels (e.g. "this channel overperforms by 3x") | Highest risk. Matches the "custom scores ... ratios" and "ranking between channels" prohibitions | Not in default output. `perf_weighted` mode is behind a flag, default `false`, and must not ship publicly without the derived-metrics application |
| Expose channel names / titles / video ids | Increases the "tracking other creators" flavor and prompt-injection surface | Not returned. Minimum 3 distinct channels per reported tag so no single creator's data is identifiable |
| Scrape charts or hashtag pages | Prohibited (III.E.6) | Not done. See audio section |

Before the public launch (M2/M3), submit the quota extension / compliance audit form describing this use case. Quota
10k/day is enough for us (section 2.7) but the audit is how a public product with 500+ users is normally
cleared. **[UNVERIFIED]** whether an audit is mandatory at our usage level.

---

## 2. `trending_hashtags`

### 2.1 Purpose and contract

Return the hashtags currently being used (recency-weighted) across a watchlist of top Brawl Stars channels, so the
orchestrator can suggest 3 to 5 tags for the draft. Computed per run from live API data; nothing retained beyond the
TTL cache.

Python signature:

```python
def trending_hashtags(
    window_days: int = 14,
    max_tags: int = 20,
    topic_hints: list[str] | None = None,   # e.g. brawler names from analyze_video; only used to flag relevance
    force_refresh: bool = False,
) -> dict: ...                               # schema in 2.8; never raises, errors returned in-band
```

Claude tool definition (what the orchestrator sees):

```json
{
  "name": "trending_hashtags",
  "description": "Get hashtags currently trending across a watchlist of top Brawl Stars channels, recency-weighted. Returns normalized hashtag tokens with usage counts and momentum. Use to choose 3-5 hashtags for a draft Short. Does not return channel or video details. May return status 'degraded' or 'unavailable'; in that case say trends were unavailable rather than guessing.",
  "input_schema": {
    "type": "object",
    "properties": {
      "window_days": {"type": "integer", "minimum": 3, "maximum": 30, "default": 14},
      "max_tags": {"type": "integer", "minimum": 5, "maximum": 40, "default": 20},
      "topic_hints": {"type": "array", "items": {"type": "string"}, "maxItems": 10,
                      "description": "Brawler/mode names from the draft, used only to mark relevant tags"}
    }
  }
}
```

(`force_refresh` is not exposed to the model; only the API/CLI can set it.)

### 2.2 Watchlist config

File: `config/watchlist.toml` (committed; it contains only public channel ids the user chose, not API data).
TOML because `tomllib` is stdlib in 3.11+, no new dependency.

```toml
# Top Brawl Stars channels to sample for hashtag usage. Public ids only.
[settings]
min_channels_ok = 5          # below this, status = "degraded"
per_channel_tag_cap = 3      # a channel counts at most this many videos per tag
min_distinct_channels = 3    # a tag must appear on >= this many channels to be reported
half_life_days = 5.0
max_pages_per_channel = 2    # playlistItems pages (50 each)

[[channel]]
id = "UCxxxxxxxxxxxxxxxxxxxxxx"   # UC... channel id (preferred)
label = "example-1"               # local note only, never returned by the tool

[[channel]]
handle = "@somehandle"            # resolved once via channels.list?forHandle (1 unit), cached in memory
label = "example-2"
```

Rules: 15 to 30 channels is the target. Validate on load with a small dataclass (`WatchlistEntry`, `Settings`);
reject duplicates, ids not matching `^UC[\w-]{22}$`, and handles without `@`. Picking the watchlist is a human task:
choose channels by hand from the Brawl Stars niche (do not use API or scraping to rank "top" channels).

### 2.3 Fetch pipeline (cheap path, no `search.list`)

For each watchlist channel (concurrency via a small thread pool, 5 workers, to keep wall time ~2 to 4 s):

1. **Uploads playlist id.** Derive from channel id: replace the leading `UC` with `UU`. This is a long-standing
   convention, costs 0 units. **[UNVERIFIED]** in current official docs (the documented way is
   `channels.list?part=contentDetails` -> `relatedPlaylists.uploads`, 1 unit per up-to-50 ids). Implementation:
   use the convention first; if `playlistItems.list` returns 404 for a channel, fall back to `channels.list` for the
   failed ids (batched, 1 unit) and use the returned id.
2. **`playlistItems.list`** (`part=snippet,contentDetails`, `playlistId=UU...`, `maxResults=50`). 1 unit per page.
   Newest first. Keep items with `contentDetails.videoPublishedAt >= now - window_days`. Fetch the next page only if
   the last item of the current page is still inside the window, up to `max_pages_per_channel`.
3. **`videos.list`** (`part=snippet,contentDetails,statistics` only if `perf_weighted` is on, else
   `part=snippet,contentDetails`), ids batched 50 per call. 1 unit per call. `snippet.description` is needed for
   hashtags (playlistItems descriptions can be truncated **[UNVERIFIED]**); `contentDetails.duration` is needed to
   keep Shorts only.
4. **Shorts filter.** Keep videos with duration <= 180 s (Shorts max length is 3 minutes since Oct 2024,
   **[UNVERIFIED]** against current help text). The API has no `isShort` flag. Optional probe, not relied on: the
   `UUSH` + id[2:] playlist is widely reported to list only Shorts **[UNVERIFIED, undocumented]**; do not build on it.
5. Discard everything but `(channel_idx, published_at, hashtags)` immediately after extraction. Raw titles and
   descriptions are never put in the cache.

Reading tags from `snippet.tags`: ignore. Those are hidden keywords, not hashtags, and they are not visible to
viewers.

### 2.4 Hashtag extraction (`extract.py`)

Input: title and description of each Short. Output: set of normalized tags per video (a set, so a tag counts once
per video).

- Regex over `title + "\n" + description`: `(?<![\w#])#([^\W\d_][\w]{1,39})` with Unicode flag (`re.UNICODE`);
  must start with a letter, 2 to 40 chars. This excludes `#1`, `#123`, and `##`.
- Normalize: `unicodedata.normalize("NFKC", s).casefold()`; strip trailing underscores.
- Drop pure noise tags (config list in code, editable): `shorts`, `short`, `youtubeshorts`, `fyp`, `viral`,
  `trending`, `subscribe`, `foryou`, `ytshorts`. Keep `brawlstars` but mark it `baseline: true` in output and exclude
  it from the ranked list (it is on nearly everything, so it carries no information).
- Cap at 15 tags per video (spam guard).
- Security: because descriptions are attacker-controlled text, **only the regex-matched, length-capped token** ever
  leaves this function. No raw text is forwarded to any LLM. Output schema validation (2.8) re-checks every tag against
  `^[^\W\d_][\w]{1,39}$` before returning.

### 2.5 What "trending" means

Three candidate definitions considered:

| Definition | Verdict |
|---|---|
| Overall counts in the window | Simple, but dominated by evergreen tags (`#brawlstars`-adjacent) and by one prolific channel spamming a tag. Used as an input, not the answer. |
| Recency-weighted frequency | **Chosen.** Rewards tags used a lot lately; self-corrects for stale tags. |
| Relative performance within each channel (video views / that channel's median) | Best signal of "tag associated with overperforming content", but it is a custom ratio over other creators' channels. Policy risk (section 1.3), and also noisy (tags correlate with brawler/meta, not causally with performance). **Off by default.** |

Chosen score, for tag `t`:

```
w(v)      = 0.5 ** (age_days(v) / half_life_days)            # half_life_days = 5 by default
score(t)  = sum over channels c of  sum over the top `per_channel_tag_cap` videos v of c containing t of  w(v)
```

Reported only if `n_channels(t) >= min_distinct_channels` (default 3). The per-channel cap stops one creator
dominating; the channel floor is a breadth requirement and also prevents any single creator's data being
identifiable.

Momentum label (per tag, not per channel):

```
rate_recent = weighted uses in the last 3 days / 3
rate_prior  = weighted uses in days 3..window / (window - 3)
momentum    = "rising"  if rate_recent >= 1.5 * rate_prior
              "fading"  if rate_recent <= 0.5 * rate_prior
              "steady"  otherwise
```

Because the counts come from a small watchlist and ~14 days, report `n_videos` and `n_channels` so the model can see
how thin the evidence is. Tags with `n_videos < 5` get `low_evidence: true`.

`perf_weighted` mode (disabled; spec only so it can be enabled after the derived-metrics application): adds
`videos.statistics.viewCount` to `part`, computes per-video `ratio = views / median(views of that channel's videos
in the window)` and replaces `w(v)` with `w(v) * min(ratio, 3)`. Views are heavy-tailed (see master plan), so clip
the ratio. Output adds `perf_lift` per tag only. Gate: `settings.perf_weighted = true` AND env
`ENABLE_DERIVED_METRICS=1`; startup refuses otherwise.

### 2.6 Policy-safe handling and caching

- **Ephemeral by default.** The whole pipeline runs in memory. No file under `data/` is written for other channels.
- **Shared in-process TTL cache** (`cache.py`): key = `(watchlist_hash, window_days)`, value = the computed result
  plus `fetched_at`. TTL 6 h. Purpose: with a public app, 500 creators must not each cost ~50 units (section 2.7).
  Cache holds only the final per-tag aggregate plus the tiny per-video tuples needed to recompute for other
  `window_days` values (published_at + tag set + channel idx), no titles, descriptions, ids, or stats.
- **No disk cache by default.** If one is ever needed (e.g. surviving restarts on Cloud Run): file in
  `data/cache/trends.json`, hard TTL 24 h, delete on read if older, written only with the same stripped tuples. The
  30-day ceiling is ~30x away.
- Cache is process-local; on a multi-instance deploy each instance fetches independently (still <= ~250 units/day
  per instance).
- Purpose limitation: use the output only inside the publish plan shown to the requesting creator. It must not be
  exposed as a browsable "trends dataset" endpoint in the FastAPI app or MCP server beyond the tool's own schema
  (the MCP tool returns the same capped output).
- Attribution/disclosure: the app's privacy/ToS page should state that it uses YouTube API Services and link
  Google's privacy policy and YouTube ToS (standard requirement **[UNVERIFIED]** in exact wording; check the ToS page
  at launch).

### 2.7 Quota cost per run

Let N = number of watchlist channels (target 20), P = pages per channel actually needed (1 to 2).

```
playlistItems.list: N * P units
videos.list:        ceil(total_recent_videos / 50)  ~ N * P   (each channel's page is already <= 50 ids)
channels.list:      0 normally (UU derivation); +1 per 50 ids only on 404 fallback; +1 per handle resolve (cached)
total (cache miss)  ~ 2NP units  = 40 to 80 units for N=20
```

Batching `videos.list` across channels reduces calls only marginally (50 ids per call). With a 6 h cache TTL the
absolute worst case is 4 refreshes/day = ~320 units/day regardless of user count, about 3% of the 10,000 shared
quota. (`search.list` approach for comparison: 1 call per channel-query, hard-capped at 100/day, and returns ids
only, so it would also need `videos.list`. Rejected.)

The tool returns `usage.quota_units` so the trace (`src/obs/`) can record it.

### 2.8 Output schema

```json
{
  "tool": "trending_hashtags",
  "status": "ok",                       // "ok" | "degraded" | "unavailable"
  "generated_at": "2026-10-01T18:04:11Z",
  "from_cache": false,
  "cache_age_seconds": 0,
  "window_days": 14,
  "coverage": {"channels_ok": 19, "channels_failed": 1, "shorts_considered": 212},
  "tags": [
    {
      "tag": "colette",
      "score": 11.4,
      "n_videos": 17,
      "n_channels": 8,
      "momentum": "rising",              // rising | steady | fading
      "low_evidence": false,
      "relevant_to_hints": true          // only if topic_hints given: case-insensitive substring match
    }
  ],
  "baseline_tags": ["brawlstars"],
  "warnings": [],                        // e.g. "3 channels returned 404", "served stale cache (quota exceeded)"
  "usage": {"quota_units": 44, "api_calls": 42, "latency_ms": 2310},
  "notice": "Computed live from public YouTube API data; not stored. Counts reflect a small hand-picked watchlist."
}
```

Not in the output, on purpose: channel ids/names/labels, video ids, titles, descriptions, per-channel stats, any
rank of channels.

`unavailable` shape: `{"tool", "status": "unavailable", "error": {"code": "...", "message": "..."}, "tags": [], "usage": {...}}`.

### 2.9 Error handling

| Condition | Behavior |
|---|---|
| `YOUTUBE_API_KEY` missing | `unavailable`, code `config_missing`, no network call |
| HTTP 403 `quotaExceeded` / `rateLimitExceeded` | Stop all further calls this run. If a cache entry exists with age <= 24 h serve it with `status:"degraded"`, warning "stale"; else `unavailable` code `quota_exceeded`. Do not retry until next PT midnight (store a process-level `quota_blocked_until`). |
| 403 other (e.g. `forbidden`, key restrictions) | `unavailable`, code `auth_error`, message kept short, no key echoed |
| 404 on a playlist | Fallback via `channels.list` once; if still failing, skip channel, add to `channels_failed` |
| 5xx / network timeout | 2 retries, exponential backoff 0.5 s then 1.5 s, per-request timeout 10 s; then skip channel |
| `channels_ok < min_channels_ok` | `degraded`, tags still returned but `warnings` explains thin coverage |
| Zero tags pass the floors | `ok`, empty `tags`, warning "no tag met min_distinct_channels" |
| Whole-run wall clock > 20 s | Return what is complete with `degraded` |
| Any unexpected exception | Caught at the tool boundary; `unavailable`, code `internal_error`; log stack trace, never raise to the orchestrator |

### 2.10 Optional own-channel validation (not required)

Hashtag usefulness for Shorts reach is itself unproven. A cheap check uses only the creator's own data
(Authorized Data, allowed): across the 277 own Shorts, compare tag usage against relative engagedViews. This does
not validate the *trending* signal, only whether tags matter at all. Put in 03's eval harness as a side analysis, not
a gate for this tool.

---

## 3. `trending_audio`

### 3.1 What exists officially (and what does not)

- **YouTube Charts**, "Top Songs on Shorts": help text says "Top Songs on Shorts recognizes song performance across
  YouTube Shorts. Songs are ranked daily, based on the total number of Shorts created that feature the song."
  (support.google.com/youtube/answer/9014376). Daily and weekly charts; a top-50 list per market. Viewable at
  charts.youtube.com.
- **Coverage:** the help page says the chart appears in the 61 countries where YouTube Charts operate; one third
  party listing says "22 markets" for the Shorts chart. **[UNVERIFIED]** which is current.
- **API / export:** I found none. The help page contains no download or export option, and no terms for reusing chart
  data. charts.youtube.com is a JS app, so I could not read it with a plain fetch. The YouTube Data API v3 has no
  chart or audio-track endpoint, and no endpoint maps a Short to the sound it uses. **[UNVERIFIED]** absence of an
  undocumented/partner feed (YouTube Music partner/Content ID reports exist for rights holders, not for us).
- **Third-party scrapers** (Apify actors etc.) sell exactly this data. Using them would mean obtaining scraped YouTube
  data, which III.E.6 prohibits. Not used.
- The chart is **music only**, ranked by number of Shorts made with the song. Many Brawl Stars Shorts use game audio,
  voiceover, or meme sounds that never appear on it. The suggestion must be honest about that.
- No legal automated Short -> sound mapping, so we cannot compute "which sound did top Brawl Stars Shorts use".

### 3.2 Ingestion options (honest ranking)

| Option | What | Legality / effort | Verdict |
|---|---|---|---|
| A. Manual chart drop | User opens the official chart weekly, pastes the top N rows into `data/audio/chart.csv` | Human reading a public page; no automated access. Reuse terms for chart data **[UNVERIFIED]** (keep to rank/title/artist, personal use, small). 5 min/week | **Ship** |
| B. User-supplied sounds per run | Draft request includes `candidate_sounds: ["Song - Artist", ...]` the creator is considering | Their own input | **Ship** |
| C. Gemini audio analysis of the draft | Describe the draft's audio and judge style fit against each candidate | Own draft video; model knowledge of songs can be wrong | **Ship as optional**, flagged |
| D. Scraped/3rd-party feed | Apify etc. | Prohibited (III.E.6) | **Rejected** |
| E. Short -> sound mapping from other channels | No legal path | n/a | **Not possible** |

### 3.3 Data files

`data/audio/chart.csv` (gitignored, user-maintained):

```csv
rank,title,artist,market,chart_week,bpm,genre,mood,notes
1,Example Song,Example Artist,US,2026-09-28,128,electronic,hype,
```

Required: `rank,title,artist,chart_week` (ISO date of the week start). Optional: `market,bpm,genre,mood,notes`
(user-added metadata, which helps style matching because the model cannot hear the track). JSON accepted too
(`chart.json`: list of the same objects). Loader validates, caps at 100 rows, strips control characters, and truncates
each string field to 120 chars (these files are user-controlled but still go into a prompt, so treat as untrusted
text: pass as quoted data, never as instructions).

Staleness: weekly chart. `age_days = today - chart_week`. `>10` days -> warning "chart is stale"; `>35` -> status
`degraded` and the chart section is omitted. No chart file -> `chart` empty plus a warning; the tool still works with
user sounds.

### 3.4 Signature, tool definition, and behavior

```python
def trending_audio(
    video_path: str | None = None,              # draft; required only for style match
    candidate_sounds: list[dict] | None = None, # [{"title": str, "artist": str|None, "bpm": int|None, "genre": str|None, "mood": str|None}]
    market: str | None = None,                  # filter chart rows if the CSV has several markets
    top_n: int = 10,
    style_match: bool = False,                  # opt-in; costs one Gemini call
) -> dict: ...
```

```json
{
  "name": "trending_audio",
  "description": "Get sound candidates for a draft Short: rows from the official 'Top Songs on Shorts' chart (user-maintained weekly file) plus any sounds the creator supplied, optionally ranked by how well each fits the draft's audio style. The chart lists popular music across all of Shorts, not what top Brawl Stars Shorts use. Never claim a sound is proven to help this video. If the chart is stale or missing, say so.",
  "input_schema": {
    "type": "object",
    "properties": {
      "candidate_sounds": {"type": "array", "maxItems": 10, "items": {"type": "object",
          "properties": {"title": {"type": "string"}, "artist": {"type": "string"},
                         "bpm": {"type": "integer"}, "genre": {"type": "string"}, "mood": {"type": "string"}},
          "required": ["title"]}},
      "market": {"type": "string"},
      "top_n": {"type": "integer", "minimum": 3, "maximum": 25, "default": 10},
      "style_match": {"type": "boolean", "default": false}
    }
  }
}
```

(`video_path` is injected by the orchestrator/API from the run context, not chosen by the model, so the model cannot
point the tool at arbitrary files.)

Flow:

1. Load chart (3.3), filter by `market`, take `top_n`.
2. Merge with `candidate_sounds`, tagging each `source: "chart" | "user"` (user duplicates of chart rows are merged,
   `source: "both"`).
3. If `style_match` and `video_path`: one Gemini call (reuse the client from `analyze_video`, phase 3) on the draft
   with a fixed JSON-schema response:
   - `draft_audio`: `{has_voiceover, has_game_audio, music_present, est_tempo: "slow|mid|fast", energy: "low|mid|high", mood: [..], speech_density: "none|low|high", notes}`.
   - `fits`: per candidate `{key, fit: "good|ok|poor|unknown", reason (<=140 chars)}`.
   The prompt tells the model: judge only from the supplied metadata and the draft's audio; answer `unknown` for any
   track it does not recognize; do not invent tempo or genre; do not claim popularity. Candidates are passed as quoted
   JSON data.
4. Return sorted: `good` > `ok` > `unknown` > `poor`, ties by chart rank, user-supplied first among ties.
5. Without `style_match`, order is user sounds first, then chart rank; `fit` is `null`.

Limits of the style match, stated plainly: Gemini does not hear the candidate song (no audio is supplied for it;
copyrighted), so `fit` rests on the user-provided bpm/genre/mood and the model's prior knowledge of the track
**[UNVERIFIED reliability; add to evals, see 5.3]**. It is a suggestion aid, not a prediction.

### 3.5 Output schema

```json
{
  "tool": "trending_audio",
  "status": "ok",                         // "ok" | "degraded" | "unavailable"
  "chart": {"name": "Top Songs on Shorts", "market": "US", "chart_week": "2026-09-28", "age_days": 3, "stale": false},
  "draft_audio": null,                    // filled only when style_match ran
  "suggestions": [
    {"title": "Example Song", "artist": "Example Artist", "source": "chart", "chart_rank": 4,
     "bpm": 128, "genre": "electronic", "mood": "hype",
     "fit": "good", "fit_reason": "Fast, high-energy gameplay montage with no voiceover matches the tempo."}
  ],
  "warnings": ["Chart ranks music made in Shorts overall, not Brawl Stars content."],
  "usage": {"gemini_calls": 1, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0},
  "notice": "Chart rows were entered manually from the official YouTube Charts page. No automated Short-to-sound mapping exists."
}
```

Errors: missing chart file -> `status:"degraded"`, suggestions only from user sounds, warning. Gemini error or
schema-invalid response -> retry once, then return suggestions with `fit: null` and warning "style match failed".
No video / no candidates / no chart -> `unavailable`, code `no_inputs`.

The orchestrator should phrase the output as "popular right now; check the sound fits your clip", and must not state
that a sound will increase views.

---

## 4. File layout

```
config/
  watchlist.toml                      # committed, public channel ids only
src/tools/
  __init__.py                         # re-exports tool functions + TOOL_SPECS for the orchestrator/MCP
  _base.py                            # shared: ToolResult helpers, in-band error shape, usage dict (coordinate with 03/05)
  trending_hashtags/
    __init__.py                       # exports trending_hashtags, TOOL_SPEC
    tool.py                           # entry point: orchestrates the pipeline, builds output, catches everything
    watchlist.py                      # load/validate watchlist.toml
    youtube_reader.py                 # thin wrapper over googleapiclient: uploads_page(), videos_batch(), channels_batch(); counts quota units; the injection point for fakes in tests
    extract.py                        # hashtag regex + normalization + noise list
    score.py                          # recency weighting, per-channel cap, floors, momentum
    cache.py                          # in-memory TTL cache (+ optional 24 h disk variant, off by default)
    schema.py                         # TOOL_SPEC + output validation (tag regex re-check)
  trending_audio/
    __init__.py
    tool.py
    chart_loader.py                   # CSV/JSON -> validated rows, staleness
    style_match.py                    # Gemini call + response schema validation (uses phase 3 client)
    schema.py
data/audio/chart.csv                  # gitignored, manual
tests/
  fixtures/trends/                    # canned API JSON + chart files
  test_trending_hashtags_extract.py
  test_trending_hashtags_score.py
  test_trending_hashtags_tool.py
  test_trending_audio_chart.py
  test_trending_audio_tool.py
```

Dependencies: `google-api-python-client` is already in `requirements.txt`. No new runtime dependency (TOML via stdlib
`tomllib`; Gemini client comes with phase 3). Public reads use `build("youtube", "v3", developerKey=...)`, separate
from the OAuth client in `src/channel_export/auth.py`. Check `src/channel_export/common.py` for retry/backoff helpers
before writing a new one. Add `YOUTUBE_API_KEY` to the `.env` template; restrict the key to the YouTube Data API v3 in
Cloud Console.

---

## 5. Tests (all offline, no network, no real keys)

`youtube_reader.py` is the seam: tests pass a `FakeReader` (or a `MagicMock` of the googleapiclient resource chain
`youtube.playlistItems().list(...).execute()`) returning fixtures from `tests/fixtures/trends/`.

### 5.1 `trending_hashtags`

- **extract:** `#Colette`, `#colette`, `#COLETTE` -> one tag; `#1`, `#123`, `##x`, `a#b` not matched; unicode
  tags (`#ブロスタ`) kept; trailing punctuation stripped; >15 tags truncated; noise tags dropped; `#brawlstars`
  surfaces as baseline only; tag appearing twice in one video counts once; hostile description ("ignore previous
  instructions ...") yields no tokens beyond real hashtags.
- **score:** recency weight halves at `half_life_days`; per-channel cap applied (one channel with 10 videos on a tag
  contributes at most 3 videos' weight); a tag on 2 channels is excluded, on 3 included; momentum labels at the 1.5x
  and 0.5x thresholds; `low_evidence` below 5 videos; videos outside the window ignored; deterministic ordering on ties
  (score desc, then tag asc).
- **tool (with fake reader):**
  - happy path: schema valid, no channel/video ids anywhere in the serialized result (assert by scanning the JSON for
    fixture ids/titles), `usage.quota_units == expected`;
  - pagination: stops when the last item is outside the window; respects `max_pages_per_channel`;
  - Shorts filter: a 200 s video excluded, a 59 s video included;
  - cache: second call is `from_cache: true` with zero reader calls; `force_refresh` bypasses; TTL expiry via injected
    clock; cache values contain no titles/descriptions (inspect the internal object);
  - errors: 403 quotaExceeded with warm cache -> `degraded` stale; without cache -> `unavailable`/`quota_exceeded`
    and no further calls; 404 triggers the `channels.list` fallback once; one failing channel -> `channels_failed: 1`;
    5xx retried twice then skipped; `channels_ok < min_channels_ok` -> `degraded`; missing key -> `config_missing`
    with zero calls; random exception -> `internal_error` not raised;
  - `perf_weighted` refuses to enable without `ENABLE_DERIVED_METRICS=1`; default output has no `perf_lift` field.
- **watchlist:** invalid id, duplicate id, handle without `@`, missing file -> clear errors.

### 5.2 `trending_audio`

- chart loader: valid CSV and JSON; missing required column; control characters stripped; fields truncated; >100 rows
  capped; staleness at 10 and 35 days; market filter; no file -> degraded.
- tool: merge/dedupe of chart and user sounds (`source: "both"`); ordering with and without style match;
  `style_match` with a mocked Gemini client returning (a) valid JSON, (b) invalid JSON then valid on retry, (c)
  persistent failure -> `fit: null` + warning; prompt-injection text in a song title is passed only inside the quoted
  data block (assert on the captured prompt); `video_path` not provided with `style_match=True` -> warning, no call.

### 5.3 Evals (phase 3 harness, later)

Hashtag and audio tools have no ground truth we may store. What can be measured: (1) tool reliability (success rate,
p50/p95 latency, quota per run) from traces; (2) style-match sanity on 10 to 15 own old Shorts where the true sound
is known: does `fit` rank the real sound above random chart songs more often than chance? That checks the Gemini
claim in 3.4 and is the one eval worth adding.

---

## 6. Acceptance criteria

`trending_hashtags`:
1. With a 20-channel watchlist and a valid key, a cold run returns `status: "ok"` in under 10 s and uses <= 80 quota
   units; a warm run uses 0 units.
2. No call to `search.list` anywhere (grep test + fake reader that raises on unknown methods).
3. Nothing from the watchlist channels is written to disk in the default config (test asserts `data/` unchanged).
4. Serialized output contains no channel ids/names, video ids, titles, or descriptions, and every tag matches the
   token regex.
5. Every tag reported has `n_channels >= 3`.
6. All error rows in 2.9 are covered by tests; the tool never raises.
7. `perf_weighted` cannot be enabled without the explicit env gate.
8. Run unit tests with `./venv/bin/python -m pytest tests/` green; no network in tests.

`trending_audio`:
1. Works with only `candidate_sounds` (no chart) and with only a chart (no user sounds).
2. Stale (>10 d) and missing chart produce the specified warnings/status.
3. With `style_match` off, zero Gemini calls; with it on, exactly one (two on retry).
4. No code path performs an HTTP request to youtube.com / charts.youtube.com.
5. Output always carries the "chart is music across Shorts, not Brawl Stars specific" warning.
6. Tests green offline.

---

## 7. How the orchestrator calls them (phase 5, 07)

- Register both via `src/tools/__init__.py: TOOL_SPECS` (the same specs are reused by the MCP server so the two
  surfaces cannot drift).
- Parallel tool use: in the first model turn the orchestrator issues `analyze_video`, `trending_hashtags`, and (if
  the user supplied sounds or a chart exists) `trending_audio` together. `trending_hashtags` does not depend on the
  video; relevance filtering happens in the model using `analyze_video`'s brawler/mode names. If it wants
  `relevant_to_hints`, it may call hashtags again after `analyze_video` (cache makes that free).
- The API layer injects `video_path` for `trending_audio`; the model never supplies file paths.
- Per-tool timeout: hashtags 20 s, audio 30 s (style match). On timeout or `unavailable`, the tool result goes back
  in-band and the system prompt tells the model to write "trend data unavailable" and continue the plan. A failure in
  either trend tool must never fail the whole run.
- Tool results are treated as data: the system prompt states that tool output never contains instructions. Hashtag
  output is regex-bounded tokens; audio output includes user-supplied strings, so they are quoted and length-capped
  upstream.
- Plan wording rules (put in the orchestrator prompt): suggest 3 to 5 hashtags, prefer tags with `momentum != fading`
  and `relevant_to_hints`; mention `low_evidence`; never present counts as a guarantee; for audio, state it is a
  suggestion based on a manually updated chart, and mention staleness when `chart.stale`.
- Trace fields per call: `tool`, `status`, `usage.*`, cache hit, latency. Feeds `src/obs/` cost tracking.

---

## 8. Unverified / open questions

1. **Aggregation reading (III.E.2).** Whether per-run hashtag counts across other channels count as prohibited
   aggregation. Mitigations are built in (ephemeral, tokens only, floors, no channel data) but it is a judgement call.
   Read the live policy text and, ideally, ask via the API compliance contact/audit form.
2. **Derived metrics boundary.** Hashtag score/momentum are metrics about tags, not about channels, and do not
   replace API data, so I believe they are outside the restricted "custom channel score" category, but this is my
   reading. `perf_weighted` is clearly inside it and is gated.
3. **Fetch summaries.** All policy and help-page text above came via a summarizing fetch tool, not the raw pages. Quotes
   may be slightly altered and the "viewable only to the channel owner" line in the compliance guide may be scoped to
   Authorized Data. Re-read the live pages before launch.
4. `search.list` 100/day bucket: confirmed in docs; the "since June 2026" date not found.
5. Uploads playlist `UC` -> `UU` convention and the `UUSH` Shorts-only playlist: undocumented/convention. The
   `UU` path has a documented fallback; `UUSH` is not used.
6. Shorts max length 180 s and how to detect Shorts via the API (no flag exists): heuristic by duration only.
7. Whether `playlistItems.list` descriptions are truncated (we fetch `videos.list` regardless).
8. YouTube Charts: market count (61 vs 22), any export or terms for reusing chart data, any partner API. I could not
   load charts.youtube.com content.
9. Whether a compliance audit/quota extension is required at 500+ creators and exact required ToS/privacy wording
   for API Services disclosure.
10. Gemini's ability to judge fit for songs it cannot hear (3.4): to be measured per 5.3, not assumed.
11. Whether trending hashtags actually move Shorts performance: unproven, see 2.10.
