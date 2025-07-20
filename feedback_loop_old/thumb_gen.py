import os
import google.generativeai as genai
from PIL import Image
from frames import get_frames, save_frames

genai.configure(api_key=os.environ["GEMINI_KEY"])

PICK_PROMPT = """these are frames from a brawl stars youtube short, numbered 0 to {n}.
which one would work best as the thumbnail? i want people to click on it.
reply with ONLY the number."""

GEN_PROMPT = """make a youtube shorts thumbnail for a brawl stars video.
it should look exciting and colourful and make people want to click. {extra}"""


def pick_frame(video_path, n=8):
    """let gemini look at the frames and choose one"""
    frames = get_frames(video_path, n)
    paths = save_frames(frames)
    model = genai.GenerativeModel("gemini-2.0-flash")
    imgs = [Image.open(p) for p in paths]
    resp = model.generate_content([PICK_PROMPT.format(n=n - 1)] + imgs)
    try:
        choice = int(resp.text.strip())
        if choice < 0 or choice >= len(paths):
            choice = 0
        return paths[choice]
    except Exception as e:
        print("gemini said something weird:", resp.text)
        print(e)
        return paths[0]


def gen_thumbnail(extra="", out="thumb_gen.png"):
    """or just have it generate one from nothing"""
    model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
    resp = model.generate_content(
        GEN_PROMPT.format(extra=extra),
        generation_config={"response_modalities": ["IMAGE", "TEXT"]},
    )
    for part in resp.parts:
        if getattr(part, "inline_data", None):
            with open(out, "wb") as f:
                f.write(part.inline_data.data)
            return out
    print("no image came back :(")
    return None


if __name__ == "__main__":
    print(gen_thumbnail())
