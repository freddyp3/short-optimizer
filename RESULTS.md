# Results

What was tested, how, and what held up. Every number here comes from videos the system had not seen when the
prediction was made.

## Setup

- **Data:** 235 Brawl Stars Shorts from one channel, each with its real retention curve from YouTube Analytics,
  and 12,852 viewer comments.
- **Splits:** 145 videos for building, then two held-out sets (40 and 50 videos). Each held-out set was opened
  once, after the method and the pass rule had been written down. Code refused to load them before that.
- **Test design:** the held-out sets use an extreme-groups design (the channel's strongest and weakest videos by
  retention), which gives the most statistical power from a small number of videos. A further 105 everyday
  videos were scored out-of-sample during development.
- **Metric:** rank correlation between the prediction and the real outcome, after removing the effect of video
  length (longer Shorts always retain a smaller share). 0 means no skill, 1 means a perfect ranking.
- **Checks:** permutation tests with correction for the number of things tried; every held-out score was
  recomputed by an independent script and matched to four decimals.

## Retention predictions

| Prediction | Source | Held-out set 1 | Held-out set 2 |
|---|---|---|---|
| Where viewers leave (0.50 = chance) | Gemini reading a second-by-second description | 0.65 | 0.66 |
| How well the middle holds (25-75 %) | Gemini | 0.56 | 0.63 |
| How well the ending holds (75-95 %) | Random forest on structure features | 0.48 | 0.60 |
| How well the opening holds (0-25 %) | Random forest | not tested | 0.56 |

All pass their pre-registered test on set 2 (p ≤ 0.0005 after correction).

## Working with a small dataset

The whole system was built on 145 videos from one channel. That is the main constraint, and most of the design
follows from it.

- **Small models, few features.** With about 150 videos, roughly ten parameters can be estimated before
  predictions stop generalising (sample-size criteria of Riley et al., *Statistics in Medicine*, 2019). The
  forest uses nine features; the feature checklist was frozen before any model was trained.
- **More rows from the same videos.** Modelling each of the 100 points of every retention curve turns 145 videos
  into 14,500 training rows. In cross-validation this model ranks the moments within a video at 0.43, against 0.28
  for a baseline that knows only position and length.
- **Splitting the target.** One overall retention score could not be predicted at all. Splitting the curve into
  opening, middle and ending gave three targets that could, because different things drive each part.
- **No tuning on test data.** Every choice was made by nested cross-validation on the 145 videos. Significance
  was checked with permutation tests corrected for the number of models tried (19 in one round).
  The two held-out sets were locked in code until the method was frozen, and each was scored once.
- **Failures are kept.** When a model failed, the target was changed and the failure recorded, instead of
  searching for a setting that happened to pass.

The result is four predictions that pass a pre-registered test on videos the system had never seen, from a
dataset a few hundred times smaller than those used in published work on the same problem.

## Compared with published work

The closest public benchmark is *Delving Deep into Engagement Prediction of Short Videos* (Li et al., ECCV 2024,
[arXiv 2410.00289](https://arxiv.org/abs/2410.00289)): 90,000 Snapchat videos, 72,000 of them for training, and
a deep multimodal network over video, audio and text.

| | Li et al. (ECCV 2024) | This project |
|---|---|---|
| Training videos | 72,000 | 145 |
| Model | deep multimodal network | LLM description + random forest + ridge |
| Target | watch percentage normalised for length | retention normalised for length, in three parts |
| Interpretability | black-box predictions | every prediction comes with a reason a creator can act on |
| Data requirements | massive dataset required | works with a few hundred videos from one channel |

That is comparable performance with about 500 times less training data, no video
model trained, and every prediction comes with a reason a creator can act on. On the extreme-groups held-out
sets the same predictions reach 0.56-0.63.

Two other results line up with the literature:

- **Asking an LLM directly does not work.** My first attempt scored 10 of 20. *Large Language Models Are Natural Video Popularity
  Predictors* (Kayal et al., ACL Findings 2025) reports the same pattern: a direct LLM judgement is only a little
  above chance, and performance comes from converting the video to text first.
- **Reference examples help less than expected here.** In that paper the largest gain came from showing the LLM
  similar known videos. On this channel, similar examples did not improve the retention predictions;
  random examples did as well. With one channel, every past Short is already a fairly close example.

## What did not work

- **One overall retention score.** No model beat predicting the average (19 model variants, two rounds).
- **A full predicted retention curve.** Joining the pieces into one curve beats "length only" by 2-13 %
  depending on the set. The tool therefore reports direction per segment, not a number-for-number curve.
- **Likes, comment count, share rate.** Not predictable from the video, or too few events to measure.
- **Showing the LLM similar videos to improve retention predictions.** No gain; random examples did as well.
  Similar Shorts are kept as evidence for the audience section.
- **A stronger LLM, sending raw video instead of the description, hints from the models.** No gain.

## Limits

- Predictions are relative to one channel's audience and say "above / typical / below", not percentages.
- The feature extraction was checked for run-to-run consistency but never against hand labels.
- The audience-reaction text (expected comments, current community talk) is assembled from real past comments and
  a live web search. It is not evaluated.
