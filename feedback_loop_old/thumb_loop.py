import xgboost as xgb
from thumb_gen import pick_frame, gen_thumbnail
from features import get_features

model = xgb.XGBRegressor()
model.load_model("thumb_model.json")

# was accepting literally the first attempt because i compared against
# best_score which starts at 0 lol. now theres an actual threshold
ACCEPT_SCORE = 0.05


def score(path):
    return model.predict([get_features(path)])[0]


def make_thumbnail(video_path, tries=5):
    """keep going until the model thinks the thumbnail is good enough"""
    best = None
    best_score = -1
    feedback = ""
    for i in range(tries):
        cand = gen_thumbnail(feedback, out="cand_%d.png" % i)
        if cand is None:
            cand = pick_frame(video_path)
        s = score(cand)
        print("try", i + 1, "->", cand, "score", s)
        if s > best_score:
            best = cand
            best_score = s
        if s >= ACCEPT_SCORE:
            print("accepted:", cand)
            return cand
        feedback = (
            "the last attempt was predicted to get low engagement. "
            "make it brighter and more exciting, bigger text, more contrast."
        )
    print("gave up, using", best)
    return best


if __name__ == "__main__":
    make_thumbnail("test.mp4")
