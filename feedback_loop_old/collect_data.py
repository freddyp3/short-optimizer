"""script i used to build the training csvs.

pulls recent brawl stars shorts from the public youtube data api, saves the
thumbnail + the caption, and works out engagement as likes/views from the
public stats. public data only, dont need oauth for this.
"""
import os
import csv
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("YOUTUBE_KEY")


def search_shorts(q="brawl stars shorts", max_results=25):
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "key": KEY,
            "part": "snippet",
            "type": "video",
            "videoDuration": "short",
            "order": "viewCount",
            "relevanceLanguage": "en",
            "q": q,
            "maxResults": max_results,
        },
    ).json()
    return [v["id"]["videoId"] for v in r["items"]]


def stats(video_ids):
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "key": KEY,
            "part": "statistics,snippet",
            "id": ",".join(video_ids),
        },
    ).json()
    return r["items"]


def main():
    ids = search_shorts()
    rows = []
    for v in stats(ids):
        views = int(v["statistics"].get("viewCount", 0))
        likes = int(v["statistics"].get("likeCount", 0))
        if views < 1000:
            continue  # too small, engagement number would be noise
        caption = v["snippet"]["title"]
        thumb = "https://i.ytimg.com/vi/%s/hqdefault.jpg" % v["id"]
        rows.append([v["id"], caption, thumb, likes / views])

    with open("data/collected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "caption", "thumb_url", "engagement_rate"])
        w.writerows(rows)
    print("saved", len(rows), "rows")


if __name__ == "__main__":
    main()
