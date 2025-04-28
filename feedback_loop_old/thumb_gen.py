import os
import google.generativeai as genai
from PIL import Image
from frames import get_frames, save_frames

genai.configure(api_key=os.environ["GEMINI_KEY"])

PICK_PROMPT = """these are frames from a brawl stars youtube short, numbered 0 to {n}.
which one would work best as the thumbnail? i want people to click on it.
reply with ONLY the number."""


def pick_frame(video_path, n=8):
    """let gemini look at the frames and choose one"""
    frames = get_frames(video_path, n)
    paths = save_frames(frames)
    model = genai.GenerativeModel("gemini-2.0-flash")
    imgs = [Image.open(p) for p in paths]
    resp = model.generate_content([PICK_PROMPT.format(n=n - 1)] + imgs)
    try:
        choice = int(resp.text.strip())
        return paths[choice]
    except Exception as e:
        print("gemini said something weird:", resp.text)
        print(e)
        return paths[0]


if __name__ == "__main__":
    print(pick_frame("test.mp4"))
