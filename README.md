# Shorts Pre-Publish Copilot

Give it a draft YouTube Short and it returns a publish plan: timestamped hook and pacing feedback,
predicted performance against the channel's own baseline, and hashtag and sound suggestions.

Built for Brawl Stars Shorts on my own channel ([@alfie369k](https://www.youtube.com/@alfie369k/shorts),
277 Shorts) and evaluated against the channel's real retention data.

**Status:** work in progress. The channel exporter (`src/channel_export/`) is built; the eval
harness and video analysis come next.

## How it works (planned)

- **Channel export:** OAuth pull of the channel's own videos, analytics, retention curves, traffic
  sources and comments (YouTube Data, Analytics and Reporting APIs).
- **Video analysis:** Gemini watches the Short (video + audio) and returns timestamped hook,
  pacing and drop-off analysis.
- **Orchestrator:** a Claude tool-use agent that combines the analysis with channel data and trend
  tools into a publish plan.
- **Evals:** every prediction is scored against real outcomes on held-out Shorts, against simple
  baselines.

Results will be published here once the evals run.
