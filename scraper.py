import requests
from dotenv import load_dotenv
import os

load_dotenv()

my_yt_key = os.getenv("YOUTUBE_KEY")

params = {
    "key" : my_yt_key,
    "part" : "snippet",
    "type" : "video",
    "maxResults" : 2,
    "order" : "date",
    "q" : "brawl stars shorts",
    "videoDuration" : "short",
    "relevanceLanguage": "en"
}

response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)

print(response.text)
