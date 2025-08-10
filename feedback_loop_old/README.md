# feedback loop thing

idea: gemini makes a thumbnail (or picks the best frame out of the video),
then a xgboost model scores it on predicted engagement. if the score is
too low it goes back and tries again. same thing for captions but claude
writes them.

- `frames.py` pulls candidate frames out of a video
- `thumb_gen.py` = gemini picks a frame or generates a thumbnail
- `features.py` + `train_thumb_model.py` = the xgboost judge
- `thumb_loop.py` = generate -> score -> retry until accepted
- `captions.py` + `train_caption_model.py` = same loop, claude writes

## UPDATE AUG

parking this for now. the models are basically guessing at this data size,
so the "acceptance criteria" doesnt actually mean anything yet. keeping the
code around because the loop idea is still good, it just needs real
engagement data behind it.
