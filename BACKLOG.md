# Backlog

Ideas parked for later: not dropped, just not in the current scope.

## Thumbnail / title CTR model (parked 2026-09-25)

**Why parked:** For Shorts, thumbnails are mostly unseen (the swipe feed reportedly accounts for ~99.9% of Shorts views and doesn't show thumbnails), and CTR for other channels isn't available from any official API, public dataset, or vendor we found. YouTube Developer Policies also restrict aggregating creators' data, derived metrics, and scraped data.

**What exists if we revisit:**
- YouTube Reporting API "reach reports" (added 2026-01-15): `channel_reach_basic_a1` / `channel_reach_combined_a1` give `video_thumbnail_impressions` and `video_thumbnail_impressions_ctr` per video and traffic source, **for your own/authorized channels only**. New jobs backfill just ~30 days, so start a job early if you ever want the history.
- Custom Shorts thumbnails (July 2026) appear in search, channel pages, subscriptions and playlists, but not the feed.
- YouTube's built-in Test & Compare A/B testing doesn't support Shorts.
- Public datasets with no real CTR: YTClickbait21K (arXiv 2606.14780), `vargr/yt_thumbnail_dataset`, Upworthy Research Archive (headline A/B tests, not YouTube).
- Policies: https://developers.google.com/youtube/terms/developer-policies (§III.E.2 aggregation, §III.E.4.h derived metrics, §III.E.6 scraping)
