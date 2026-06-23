import requests
import pandas as pd
import feedparser
from datetime import datetime
import time
import re

query = "flood OR earthquake OR cyclone OR landslide OR fire OR collapse OR rescue OR trapped"

posts_data = []

# ===================================
# 1️⃣ REDDIT JSON COLLECTION
# ===================================
print("Fetching Reddit data...")

headers = {
    "User-Agent": "DisasterNLPProject/1.0"
}

subreddits = ["news", "worldnews", "india", "environment"]

for subreddit in subreddits:
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "limit": 50,
        "sort": "new",
        "restrict_sr": 1
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        for post in data["data"]["children"]:
            post_data = post["data"]

            # Convert timestamp properly
            date_value = datetime.utcfromtimestamp(
                post_data.get("created_utc", 0)
            ).strftime("%Y-%m-%d %H:%M:%S")

            posts_data.append({
                "source": "reddit",
                "text": post_data.get("title", "") + " " + post_data.get("selftext", ""),
                "date": date_value
            })
    else:
        print(f"Failed to fetch from r/{subreddit}")

    time.sleep(2)

print("Reddit data collected.")


# ===================================
# 2️⃣ NEWS RSS COLLECTION
# ===================================
print("Fetching News RSS data...")

rss_urls = [
    "https://news.google.com/rss/search?q=disaster",
    "https://news.google.com/rss/search?q=flood",
    "https://news.google.com/rss/search?q=earthquake"
]

for rss_url in rss_urls:
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:

        # Clean HTML tags
        clean_text = re.sub('<.*?>', '', entry.title)

        posts_data.append({
            "source": "news",
            "text": clean_text,
            "date": entry.published if "published" in entry else ""
        })

print("News data collected.")

# ===================================
# SAVE TO CSV
# ===================================
df = pd.DataFrame(posts_data)

df.drop_duplicates(subset=["text"], inplace=True)

df.to_csv("disaster_raw_data.csv", index=False)

print(f"Total records collected: {len(df)}")
print("Step 1 Completed Successfully!")