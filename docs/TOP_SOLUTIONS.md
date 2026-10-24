# What the top teams did, and why we finished lower

Read 2026-10-24 from the published write-ups: 2nd place (private 0.960), 5th (0.958), 7th (0.957, single model, no
external training data) and team Northstar (671st, 0.942: same public base as ours). Our result: private 0.943,
rank 345 / 6,425.

## The common recipe of the top solutions

| | 2nd | 5th | 7th |
|---|---|---|---|
| Training labels | report LLM (GPT-6.1 Sol, six-way answers) + **vision LLM (GPT-6 Astra) reading every slice** with the report; teacher OOF mixed in | report labels calibrated on gold58 + report reads by a second frontier LLM + vision LLM on 10 selected slices for effusion / synovitis / Baker's; teacher OOF mixed in | three GPT-6 Astra reads (structured report + key slices, image-only, verification), per-hospital synovitis correction, half teacher OOF |
| Label quality on gold58 | 0.899 (reports) -> **0.956** | calibrated + image reads | report-only 0.89-0.91 -> **0.952** |
| Models | six own 2.5D MIL models (ConvNeXt-T / EffNetV2-S), knee-centred crops, two finding specialists | own EffNetV2-M / ConvNeXt families with bone-segmentation crops, region specialists | one own 2.5D label-query network, four backbones |
| Best single model, private | 0.957 | 0.954 | 0.958 |
| Blend | hand-set rank weights, never searched | weighted averages, per-finding components | rank mean |
| Compute / money | ~100 rented GPU-hours (~$100) plus LLM API calls | not stated | API calls, Kaggle GPUs |

Common points:
1. **Labels were the lever.** Every top team rebuilt the training labels with multimodal LLMs that read the images
   (not only the reports), following the host's definitions, and checked each label version on gold58. The public
   report-label sets that the public models (and so our base) were trained on sit at ~0.90-0.93 on gold58; the top
   labels reached 0.95+.
2. **Own models on those labels**, with knee-centred / anatomy-guided crops and diverse views, each reaching
   0.954-0.958 private alone. The blend added little on top (fixed weights; tuned weights lost on held-out data).
3. **Public and private agreed for them** (2nd place: rank correlation 0.977 between boards for its 0.955+ public
   submissions), because their gains came from labels and models, not from tuning on the public LB.
4. External data played a minor role: 7th used none; 2nd used OAI MOAKS only for a PF-OA cartilage head inside a
   specialist (the same data we used, but as an auxiliary specialist instead of a replacement of a base member).

## Why we finished at 0.943

| Cause | Evidence |
|---|---|
| **We never rebuilt the labels.** Our base models (public Raptor / R6) were trained on public report labels (~0.926 on gold58 in our audit); we diagnosed this on Oct 20 ("labels are the bottleneck") but too late to act. | top labels 0.95+ on gold58; Northstar inside our own line: better labels +0.016 private |
| **No 0.95-class model of our own.** Our own image models reached 0.90-0.93; we added one component (OAI fine-tune) to a public ensemble. | Northstar: "a 0.95-class model of your own is the lever"; their own-model blends (22 tries) added <= +0.001, as ours did |
| **A base tuned to the public LB.** Our public 0.959 sat in a crowd of ~250 teams running the same base; its per-finding weights were chosen on the public LB and dropped ~0.016 on private, more than the leaders' ~0.011. | our post-mortem (EXPERIMENT_LOG); Northstar 0.958 -> 0.942 with the same base |
| **Late start (last 10 days of a two-month competition).** Not enough time to build labels and own models. | top teams had the full two months; 2nd place notes 100+ submissions |
| Budget | 2nd place spent ~$100 on GPUs plus vision-LLM API calls over 4,400 exams; we spent ~$50 on GPUs and nothing on labels |

What we did right and the top teams also did: gold58 as the label yardstick, controlled comparisons, fail-loud
inference with fallbacks, crediting public work.

## Lessons carried to the next competitions

1. **Find the data / label bottleneck in week 1 and spend there first** (money and time). In RSNA it was image-aware
   labels; in Name That LLM it is in-style training text.
2. **Own a strong model.** Public pipelines raise the floor; a model of your own at the top tier is the lever.
3. **Fixed, pre-declared blend weights**; tuned weights did not hold for the winners either.
4. **Check that public and private would agree**: gains from data and models transfer, gains from LB tuning do not.
