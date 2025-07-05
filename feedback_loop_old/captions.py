import os
import anthropic
import xgboost as xgb

client = anthropic.Anthropic(api_key=os.environ["CLAUDE_KEY"])

PROMPT = """write a caption for a brawl stars youtube short.

the video is about: {desc}

rules:
- under 100 characters
- punchy, a bit of hype is fine
- emojis ok but dont overdo it
- no hashtag spam
- {feedback}

reply with ONLY the caption text, nothing else"""

model = xgb.XGBRegressor()
model.load_model("caption_model.json")

ACCEPT_SCORE = 0.06


def caption_features(text):
    """same idea as the thumbnail features but for text"""
    return [
        len(text),
        text.count("!"),
        sum(1 for c in text if ord(c) > 10000),  # emoji-ish
        int("?" in text),
        int(any(w in text.lower() for w in ["free", "insane", "crazy", "new", "win"])),
    ]


def write_caption(desc, feedback=""):
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(desc=desc, feedback=feedback)}],
    )
    return r.content[0].text.strip()


def make_caption(desc, tries=5):
    """same acceptance loop as thumbnails, claude writes, xgboost judges"""
    best = None
    best_score = -1
    feedback = ""
    for i in range(tries):
        cap = write_caption(desc, feedback)
        s = model.predict([caption_features(cap)])[0]
        print("try", i + 1, "score", s, "->", cap)
        if s > best_score:
            best = cap
            best_score = s
        if s >= ACCEPT_SCORE:
            print("accepted:", cap)
            return cap
        feedback = "the last caption was predicted to flop, go more hype. "
    print("gave up, using best:", best)
    return best


if __name__ == "__main__":
    print(make_caption("a 1v3 clutch in knockout"))
