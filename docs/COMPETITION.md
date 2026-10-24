# RSNA Knee Abnormality Detection — Competition Summary

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection (Overview, Data, Rules, Leaderboard — read 2026-10-09)

## Task
Multi-label, per-study probability prediction for 12 findings on knee MRI:
`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

## Metric
Mean ROC AUC over the 12 targets (macro AUC). Only ranking matters, so calibration is irrelevant.

## Submission
- `submission.csv` with header:
  `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`
- Code competition: submit via Notebook
  - CPU or GPU notebook <= 9 hours
  - Internet disabled
  - Freely and publicly available external data / pretrained models allowed (attach as Kaggle Datasets)
- Max 5 submissions per day, 2 final selections
- Test: ~1300 studies; public LB ~30%, private ~70%

## Data
- `train.csv`: StudyInstanceUID, **Report** (free-text radiology report, multilingual), 12 binary labels
  - **Only a small subset (58 studies) is expert-labeled.** Labels for the rest are expected to be derived from reports.
- `train_series.csv`: StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive (0/1), Fat_Suppression (0/1), Anatomical_Plane (Sagittal/Coronal/Axial)
- `train_series/<Study>/<Series>/<SOP>.dcm` — one slice per file; ~20–45 slices per series (median 30), long tail up to a few hundred
- `test.csv`, `test_series.csv`, `test_series/` — 3 example studies; replaced by the real test set at scoring time. **No Report at test time.**
- DICOM: varying orientation/resolution/intensity; mixed transfer syntaxes (Explicit/Implicit VR LE, JPEG Lossless, JPEG 2000) → offline decoders needed (pylibjpeg / gdcm). 86 metadata tags retained.
- Prevalence may differ between train / public / private.

## Timeline (23:59 UTC)
- 2026-10-15 — Entry deadline + team merger deadline (rules accepted ✅)
- **2026-10-22 — Final submission deadline**
- 2026-11-05 — Winners' code / video / write-up deadline

## Prizes ($77,000)
- Main LB: 1st $9k, 2nd $7k, 3rd $6.5k, 4th $6k, 5th $5.5k, 6th–10th $5k
- Efficiency track: $7k / $6k / $5k
  - `Efficiency = (maxAUC - AUC)/(maxAUC - Benchmark) + RuntimeSeconds/32400` (lower is better)
  - Must be one of the selected final submissions and beat the benchmark

## Key rules
- Max team size 5; one Kaggle account per person
- No private sharing of code/data outside the team; public sharing only on Kaggle forums/notebooks
- Open-source code used must be OSI-licensed without commercial restrictions
- No hand-labeling of test data
- Data: MIRA license; no redistribution outside the competition
- Winner license: CC-BY-NC 4.0; deliver training + inference code, weights (public Kaggle dataset), environment spec, method description, short video, forum link, public model release

## Leaderboard snapshot (2026-10-09, public)
- 1st 0.964, top-10 ≈ 0.962+, 50th ≈ 0.957
- 5,594 teams
