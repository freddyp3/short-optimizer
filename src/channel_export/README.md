# channel_export

Exports everything the YouTube APIs expose for **your own channel**: channel and video metadata,
comments, YouTube Analytics reports (per video and channel-wide) and YouTube Reporting API bulk
files (including thumbnail impressions/CTR). It signs in with OAuth as the channel owner. The
`YOUTUBE_KEY` API key in `.env` is not used.

## One-time Google Cloud setup

1. **Create a project.** Open <https://console.cloud.google.com/>, click the project picker at
   the top, then **New project**. Name it (for example `short-optimizer`), click **Create**, and
   make sure it is selected.
2. **Enable the three APIs.** Go to **APIs & Services → Library**. Search for each of these,
   open it and click **Enable**:
   - **YouTube Data API v3**
   - **YouTube Analytics API**
   - **YouTube Reporting API**
3. **Configure the OAuth consent screen.** Go to **Menu → Google Auth platform → Branding**. If it
   says *Google Auth platform not configured yet*, click **Get started**, then:
   - App information: an app name (e.g. `short-optimizer`) and your email as user support email → **Next**
   - Audience: **External** → **Next**
   - Contact information: your email → **Next**
   - Agree to the Google API Services User Data Policy → **Continue** → **Create**
4. **Add yourself as a test user.** Go to **Google Auth platform → Audience → Test users → Add users**,
   enter the Google account that owns or manages the channel, and click **Save**. Leave the
   publishing status as **Testing**.
   - In Testing mode, Google issues refresh tokens that **expire after 7 days**. When that happens
     the tool detects the failed refresh and opens the browser sign-in again, so you only have to
     click through consent once a week.
   - Moving the app to *In production* removes the 7-day limit, but Google may then require app
     verification for these scopes. For a personal tool, staying in Testing is simpler.
5. **(Optional) Declare scopes.** Under **Data Access → Add or remove scopes**, you can add
   `.../auth/youtube.readonly`, `.../auth/yt-analytics.readonly` and, if you're in the YouTube
   Partner Program, `.../auth/yt-analytics-monetary.readonly`. In Testing mode the tool requests
   them at sign-in anyway.
6. **Create the OAuth client.** Go to **Google Auth platform → Clients → Create client**. Pick
   **Application type → Desktop app**, give it a name and click **Create**. Click **Download JSON**
   in the dialog that appears (or use the download icon next to the client in the list) and save it as:

   ```
   secrets/client_secret.json      # relative to the repo root
   ```

   If you keep Google's default filename (`client_secret_<id>.apps.googleusercontent.com.json`)
   in `secrets/`, the tool uses it automatically, as long as it is the only **Desktop app** client
   in that folder. Web-application clients are ignored. If there are several Desktop clients,
   rename the one you want to `client_secret.json`. `secrets/`, `client_secret*.json` and
   `token*.json` are gitignored. To keep the file somewhere else, set
   `YT_CLIENT_SECRET=/path/to/file.json` or pass `--client-secret PATH`.

## Signing in

The first run opens a browser window:

- Pick the Google account you added as a test user.
- **Brand Account channels:** the account chooser then asks which identity to use. Pick the
  **channel** (e.g. *alfie369k*), not your personal profile. If you pick the wrong one, the export
  stops with "No YouTube channel is linked…". Delete `secrets/token.json` and run again.
- You'll see *Google hasn't verified this app*. That's expected for your own Testing-mode app:
  click **Continue**.
- **Tick every permission checkbox.** If you leave one unticked, the calls that need it fail with 403.

The token is cached in `secrets/token.json` and refreshed automatically. Use `--no-browser` to print
the sign-in URL instead of opening a browser.

## Running it

Run all commands from the repo root:

```bash
./venv/bin/pip install -r requirements.txt

# 1) Do this first (time-sensitive): create the Reporting API jobs. New jobs only backfill ~30 days,
#    and they are the only source of thumbnail impressions / impressions CTR (reach reports).
./venv/bin/python -m src.channel_export --create-reporting-jobs --reporting-only

# 2) Quick smoke test: 3 newest uploads, no comments
./venv/bin/python -m src.channel_export --max-videos 3 --skip-comments

# 3) Full export (whole channel history, all videos, comments)
./venv/bin/python -m src.channel_export

# 4) About 48h after step 1, and then at least every few weeks (files expire after 30-60 days):
./venv/bin/python -m src.channel_export --download-reports --reporting-only
```

Other flags: `--since YYYY-MM-DD` (Analytics start date; default is the channel creation date),
`--out DIR` (default `data/channel`), `--monetary` (revenue metrics, YPP channels only; adds the
monetary scope, so you'll be asked to consent again), `-v` (debug logging). You can combine
`--create-reporting-jobs` / `--download-reports` with a normal export by leaving out
`--reporting-only`.

## Output

Each run writes `data/channel/<UTC timestamp>/`:

| File | Contents |
|---|---|
| `raw/` | Untouched JSON of every API response (errors saved as `*.error.json`) |
| `channel.csv` | Channel resource, flattened (`snippet.title`, `statistics.subscriberCount`, …) |
| `videos.csv` | One row per upload: all readable parts + `content_type` (Analytics `creatorContentType`), `is_short`, `duration_seconds` |
| `video_metrics.csv` | Lifetime Analytics per video and content type: views, engagedViews, watch time, avg view duration/%, likes, dislikes, comments, shares, subscribers gained/lost, playlist adds/removes |
| `video_daily.csv` | Daily time series per video (same metrics) |
| `retention.csv` | Audience retention per video: 100 points of `elapsedVideoTimeRatio` with audienceWatchRatio, relativeRetentionPerformance, started/stoppedWatching, totalSegmentImpressions |
| `traffic_sources.csv` | Per video × `insightTrafficSourceType` (SHORTS, YT_SEARCH, HASHTAGS, SOUND_PAGE, VIDEO_REMIXES, …) |
| `traffic_source_details.csv` | Top-25 referrers (search terms, hashtags, sounds, related videos…) channel-wide and per video |
| `demographics.csv`, `geography.csv`, `devices.csv`, `subscribed_status.csv`, `sharing_services.csv`, `playback_locations.csv`, `channel_traffic_sources.csv`, `channel_daily.csv` | Channel-wide, once for all content (`creatorContentType = ALL`) and once split by content type (SHORTS, VIDEO_ON_DEMAND, …). The `report` column names the source query. |
| `cities.csv` | Top 250 cities (data from 2022 on) |
| `comments.csv` | Every top-level comment and reply |
| `video_revenue.csv`, `revenue_daily.csv`, `ad_types.csv` | Only with `--monetary` |
| `manifest.json` | What ran, per-report ok/failed/row counts, CSV row counts, every error with its exact query, Data API quota units used, elapsed time |

Reporting API files go to a **persistent** archive, `data/channel/reporting/<report type>/<start>_<end>_<reportId>.csv`,
so repeated `--download-reports` runs only fetch new files. Report types: `channel_reach_basic_a1`,
`channel_reach_combined_a1` (thumbnail impressions + CTR), `channel_basic_a3`, `channel_combined_a3`,
`channel_traffic_source_a3`, `channel_device_os_a3`, `channel_playback_location_a3`,
`channel_demographics_a1`, `channel_sharing_service_a2`.

## Things to know about the numbers

- **`views` vs `engagedViews`.** For Shorts, `views` counts every start and replay. `engagedViews`
  is the older definition (playback past the first frame) and the one YouTube uses for monetization
  and most core analytics. Both are exported everywhere the API allows.
- **Latency.** Analytics data runs 48–72h behind, and the API truncates the end date to the last
  complete day. `videos.csv` statistics (Data API) are close to real time.
- **Shorts split.** The API accepts `creatorContentType` only as a dimension, not as a filter, so
  Shorts-only numbers are the rows with `creatorContentType == SHORTS`.
- **Small numbers disappear.** YouTube withholds rows that fall below privacy thresholds, so
  low-view videos can have empty retention, demographics or traffic-detail data.
- **Failures don't stop the export.** Every report runs on its own. A 400 or 403 is logged with its
  query in `manifest.json` and the export carries on.

## Quota

- **Data API v3:** 10,000 units/day by default. Every call used here costs 1 unit, so a run is
  about `3 + videos/50 + (comment pages per video) + reply pages`. The manifest reports the exact
  count. `search.list` isn't used; since June 2026 it has its own bucket of 100 calls/day.
- **Analytics API:** a separate quota, 1 unit per request. A full run makes about
  `30 + 2 × videos + a few batched traffic queries + one per (video, referrer type) detail query`.
- **Reporting API:** a few calls per run, plus one download per report file.

Per the YouTube API Services Developer Policies, stored Data API metadata must be refreshed or
deleted after 30 days. Re-running the export refreshes it.
