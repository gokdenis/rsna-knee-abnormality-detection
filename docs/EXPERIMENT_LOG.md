# Experiment Log and Lessons

Running record of every experiment: what was tried, the measured result, and what we learned.
Metrics: **gold** = macro AUC of out-of-fold predictions on the 58 expert-labeled studies
(noisy: differences below ~0.02 are not reliable); **LB** = public leaderboard (~30% of test).

## Leaderboard submissions

| Date (UTC) | Submission | Public LB |
|---|---|---|
| 2026-10-11 | v1: ConvNeXt-T 224, report labels, 5-fold | 0.927 |
| 2026-10-11 | v1 (0.5) + PL1 ConvNeXt + PL1 EffNetV2-S + Raptor v9 | 0.932 |
| 2026-10-11 | + PL2 ConvNeXt (80 slices), Raptor weight 1.5 | 0.932 |
| 2026-10-12 | PL1/PL2 students + 320 px + ConvNeXt-S (1.5) + Raptor (2) | 0.931 |
| 2026-10-12 | Public OAI-free stack (0.943) x0.7 + ours (ConvNeXt-S, 320 px, PL2) x0.3 | 0.940 |
| 2026-10-12 | Public OAI checkpoint (0.949) x0.7 + ours x0.3 | 0.946 |
| 2026-10-12 | OAI 0.949 (0.6) + OAI-free stack (0.4), x0.75 + ours x0.25 | 0.946 |
| 2026-10-13 | OAI 0.949 (0.6) + OAI-free stack (0.4), no ours | 0.949 |
| 2026-10-14 | Copy of heliosli's public R7 + Raptor notebook (reported 0.957) | 0.956 |
| 2026-10-14 | Same + nartaa's 224 px checkpoint at 35% of the Raptor slot (varshithakolupuri) | 0.956 (rank 417 / 5954) |
| 2026-10-15 | heliosli 0.9576 recipe (R6 bundle fe055406, private copy, flip TTA, Raptor 384 + 224) | **0.957 (rank 302 / 6045; best so far, ranked above the per-finding OAI variant's 0.957 at rank 370)** |

Lesson: our models **hurt** strong blends (-0.003 against both the 0.943 stack and the 0.949 OAI model),
while the public 0.954 notebook gains +0.005 from its own ConvNeXt-S at the same weight. Ours are too weak
and too correlated to help; leave them out of final blends unless a new model proves itself on the LB.

State on 2026-10-13: the leaderboard top is 0.966 and ~0.963 is top 20; public notebooks reach 0.956-0.957
(heliosli's R7 ConvNeXt-S + nartaa's OAI Raptor CoAtNet, 50/50 per-finding rank blend; all top public
notebooks share the same three datasets). Our 0.946 is far behind, so the public 0.957 recipe becomes the
base, and any gain must come from information the public recipe lacks (OAI MOAKS training, phase B).
Displayed scores are truncated to 3 decimals; ties are ranked by the unrounded score.
On 2026-10-14: 905 teams score >= 0.956, 186 >= 0.957, 55 >= 0.960, 20 >= 0.963, 6 >= 0.965 (5954 teams).
Dropping our models from the OAI + stack blend gave +0.003 (0.946 -> 0.949), confirming the lesson.

## Labels

| Label set | Gold macro AUC | Notes |
|---|---|---|
| Regex extraction (forum) | 0.81 | |
| Single public LLM sets | 0.835-0.904 | 8 sets; no statistically clear winner (forum + our check) |
| Mean of 5 public sets + Synovitis fill | 0.893 | `steven_v4, jev_hybrid, consensus, yunus, pilkwang` |
| 0.5 report + 0.5 OOF (PL1) | 0.923 | biggest single gain we found |
| PL2 (two students) / PL3 (five runs) | 0.926 / 0.926 | saturated |

Lessons:
- "Not mentioned" is not "negative": Synovitis is silent in ~84% of reports but ~34% of those are positive. Filling silent Synovitis from Effusion (`0.22 + 0.41 * effusion`) moved that column 0.68 -> 0.79.
- Report labels and model predictions make different errors; a 50/50 mix beats both by ~0.03 on gold.
- Repeating the pseudo-label loop stops helping after one round (0.923 -> 0.926 -> 0.926).
- Gold evaluation of label mixes is honest only if gold rows are not overwritten before scoring (we hit this bug once: a fake 1.000).

## Models (gold = OOF on the 58 expert studies)

| Run | Data | Labels | Gold | Notes |
|---|---|---|---|---|
| ConvNeXt-T 224 (v1) | 64 slices, 224 px | report blend | 0.892 | 12 epochs, ~2.75 h on 2xT4 |
| EffNetV2-S 224 | v1 | PL1 | 0.897 | ~6 h: depthwise convs are slow on T4, avoid |
| ConvNeXt-T 224 | v1 | PL1 | 0.900 | |
| ConvNeXt-T 224 | v2: 80 slices, 2-98% span | PL2 | 0.898 | denser sampling did not help us |
| ResNet50 224 | v2 | PL2 | 0.881 | weaker; lr 3e-4 may be too high |
| ConvNeXt-T 320 | v3: 80 slices, 320 px JPEG | PL2 | 0.901 | resolution alone did not help |
| ConvNeXt-S 224 | v2 | PL2 | **0.908** | best of ours; capacity helps a little |
| Raptor v9 (public, as is) | Raptor 80 x 336 | their labels | 0.913 | LB 0.932 alone |
| Raptor v9 fine-tuned, lr 2e-5 | Raptor JPEG cache | PL2 | 0.912 (ep 2) | no gain; ~64 min/epoch on T4 |
| Raptor v9 fine-tuned, lr 5e-5 | Raptor JPEG cache | PL2 | 0.907 | higher lr hurts |

Ensembles of our models reach only ~0.902-0.905 on gold; model correlations are 0.93-0.97.

Lessons:
- With these labels, single models plateau around 0.90-0.91 gold (~0.93 LB). The student never beats its 0.926 teacher, so the bottleneck is the information in the labels, not resolution, slice density or fold count.
- Gold of ~0.91 corresponds to roughly 0.93 LB in our runs (Raptor: 0.913 gold -> 0.932 LB).
- Adding our models to Raptor did not move the LB (0.932 -> 0.932/0.931): they are too correlated with it and individually weaker.
- Validation AUC against pseudo-labels (0.94-0.95) is inflated because the targets contain the teacher's predictions. Use gold or LB to compare runs.

## Phase B preparation (2026-10-14)

- `knee.coat96` reproduces the 0.949 preprocessing byte for byte (12/12 competition studies identical).
- Competition coat96 cache (96 x 384, JPEG q95): two CPU shards, 2204 + 2203 studies, 0 failures,
  9.5 + 9.5 GB, ~29 min each, ~92 of 96 slices filled on average.
- The release checkpoint loads strictly into `coat96.CoatMIL`; its stored gold AUC is 0.919 (SWA of
  epochs 14, 11, 12), label order identical to ours.
- Full pipeline (fake OAI DICOM tree -> OAI cache -> 2-epoch masked fine-tune -> SWA) runs locally.

## OAI imaging on the pod (2026-10-14)

- Pod: RunPod secure RTX A6000 (48 GB, 18 vCPU, ~940 Mbit/s), $0.59/h + 500 GB volume. Community stock
  for 4090/L40S was empty. Two RunPod accounts existed; the credit was on the browser account, so the CLI
  key had to be replaced (API error "balance too low" disguised as "no instances available" at first).
- NDA command-line login needs the NDA account password (set via "Update Password" on the NDA profile),
  not the Login.gov password.
- Package layout: one `.tar.gz` per series (`00m/<release>/<ID>/<date>/<barcode>.tar.gz`) plus a
  thumbnail jpg; `image03.txt` in the package maps each file to its `image_description`
  (e.g. SAG_IW_TSE_RIGHT). V00: 93,091 series, 1.05 TB in total.
- Needed series of MOAKS-labelled knees (SAG IW TSE FS, SAG DESS, COR IW TSE, COR MPR, AX MPR):
  V00 2,399 knees / 137 GB, V01 2,594 / 144 GB, V03 1,742 / 103 GB, V06 1,253 / 78 GB.
  Unpacked size is ~2x, so visits are processed one at a time and raw images deleted after caching.
- Download speed ~30-60 MB/s with 32 threads. Large Kaggle outputs break mid-download; `curl -C -` on the
  signed output URLs resumes reliably.
- Training speed on the A6000 (coat96, bf16, 16 windows/study): 5.4 studies/s without gradient
  checkpointing (batch 4 uses 32 GB), 4.2 with it.

## coat96 fine-tunes with OAI MOAKS (phase B)

Start: nartaa's 0.949 checkpoint (gold58 0.9226 with mirror TTA). Competition targets = 0.5 PL3 + 0.5 the
release model's own predictions (self-distillation, mean correlation with PL3 0.88). OAI targets = MOAKS
(masked where not scored). Validation: gold58 + a 5% subject holdout of OAI knees (MOAKS >= 0.5 positive).
lr 2e-5 / 2e-4 (backbone / head), batch 4 x 16 windows, bf16, mirror + random 320 crop, ~27 min/epoch.

| Run | OAI data | Epoch | gold58 | OAI holdout | Notes |
|---|---|---|---|---|---|
| release | - | - | 0.9226 | 0.9273 | starting point |
| A | V00 + V03 (3,939 knees) | 0 | 0.9169 | 0.9376 | |
| A | | 1 | 0.9153 | 0.9357 | |
| A | | 2 | 0.9210 | 0.9414 | |
| A | | 3 | 0.9221 | 0.9449 | effusion 0.88 -> 0.95, synovitis 0.91 -> 0.95, ACL 0.97 -> 0.99 |
| A SWA(2,3) | | | | | uploaded as gkdnzz31/knee-coat96-oai |

| B | V00 + V03 + V06 (5,143 knees), seed 7 | 3 | 0.9217 | 0.9406 (from 0.9120 on its own holdout) | SWA(2,3) saved, not submitted |
| C | all four visits, PL3 mix | - | - | - | stopped after the run A LB result (same confound) |
| D | all four visits (7,588 knees), competition targets = release predictions only | 0 | 0.9216 | 0.9301 (from 0.9161) | |
| D | | 1 | 0.9200 | 0.9276 | |
| D | | 2 | 0.9244 | 0.9369 | first run above the release on gold58 |
| D | | 3 | 0.9241 | 0.9379 | ACL 0.90 -> 0.94, Baker's 0.92 -> 0.98, med. meniscus 0.91 -> 0.95 |
| D SWA(2,3) | | | | | LB: A96 = 0.5 -> 0.956; A96 = 1.0 (replaces the Raptor slot) -> 0.956 (run A: 0.955) |
| E | as D, seed 23 | 1 | 0.9274 | 0.9258 (from 0.9102) | |
| E | | 3 | 0.9271 | 0.9321 | best gold58 so far; SWA(2,3) uploaded |
| D+E | ensemble of two arms | | | | LB: A96 = 0.5 -> 0.956; A96 = 1.0 -> 0.956 (rank 668 / 6043) |
| D+E per finding | A96 = 1.0 for ACL, Effusion, Baker's, Medial Meniscus, Synovitis; 0.25 elsewhere | | | | **LB 0.957 (rank 370 / 6045)**, first 0.957; old faulty targets, no JPEG round-trip |

Reading: with clean (self-distilled) competition targets the OAI fine-tune is at least as good as the
release model on the LB (0.956 when it fully replaces it, vs 0.955 for run A), but the OAI gain does not
show at three decimals. Lesson: run D's GPU sat idle ~9 h overnight because an ssh call hung during the
upload; the pod is now stopped between jobs.

LB tests of run A inside the 0.956 recipe:
- A96 = 0.5 (our model 25% of the final blend): 0.956 (rank 580 / 5995, no visible change).
- A96 = 1.0 (our model replaces the Raptor slot): **0.955**, i.e. worse than the release model there.

Reading: the OAI holdout gain (+0.018) did not transfer to the competition test. Confound: half of the
competition targets came from PL3, whose OOF half is our own weak models (~0.90 gold). Run D removes it
(pure self-distillation on competition studies), so any LB change there is due to OAI.
Lesson: gold58 (and an OAI holdout) did not predict the LB direction; only the LB can judge.
Lesson: the GPU sat idle for 1.5 h after run B because the watcher stopped after epoch 1; always chain
the next job or watch until the run ends.

### Open question: did our stage run on the hidden test?

Our stage is skipped when less than 40 min of the 9 h budget is left, and Kaggle does not show hidden-test
logs. A skipped stage leaves the base submission unchanged, which would also explain the flat 0.956 scores.
Run A replacing the Raptor slot scored 0.955 (not 0.956), so at least that single-arm stage ran.
Test: `knee-h9576-de50r` raises an error when the stage does not complete; a notebook error on the LB
means it was being skipped. **Result: 0.957, no error -> the stage runs on the hidden test within 9 h.**
Correction: an earlier note said it became our best submission; that was read from the leaderboard's
LastSubmissionDate column, which is the team's latest submission, not the best one. Not valid evidence.

## Label relabelling test (2026-10-15)

Blind in-session LLM labelling of the 58 gold reports for the four weakest report-derived findings
(gold AUC; positives 11 / 21 / 27 / 35):

| Source | Lateral OA | PF OA | Synovitis | Effusion | Mean |
|---|---|---|---|---|---|
| jev_hybrid | 0.855 | 0.902 | 0.785 | 0.877 | 0.855 |
| steven_v4 | 0.833 | 0.902 | 0.790 | 0.877 | 0.850 |
| ours (blind) | 0.825 | 0.902 | 0.787 | 0.837 | 0.838 |
| ours + jev (rank mean) | 0.843 | 0.912 | 0.786 | 0.853 | 0.849 |

Conclusion: careful relabelling matches the best public sets but does not beat them; the ceiling is the
reports (gold labels were read from the images; many reports never mention synovitis). Dropped.

## Independent review: verified findings

| Finding | Verified how | Status |
|---|---|---|
| MOAKS +0.5 / -0.5 are within-grade change codes, not grades; the mapper read 0.5 as full-thickness grade 5 -> OA = 1 | data dictionary 1.6.1 (p.5); raw V03: 479 x "0.5", 9 x "-0.5"; V06: 256 x "0.5" | **confirmed, fixed** |
| Meniscus code 7 (progressive partial maceration) missing from tear codes | data dictionary meniscus table; 88 cells in V01/V03/V06 | **confirmed, fixed** |
| `collapse()` picked one read; repeated reads disagree | groupby check | **changed to the mean over reads** |
| Cache slices are JPEG q95, the injected inference reads raw DICOM pixels | code | **confirmed; our stages now apply the same cv2 q95 round-trip** |
| OAI holdout seed = training seed (D and E scored on different knees) | code | **fixed: `holdout_seed` = 2026 for all runs** |
| No OAI-free control with the same steps | design | **added `epoch_size` for matched-steps controls** |
| Builder hard-coded 384 | code | already fixed (224 stage) |
| Stage may be skipped on the hidden test | design | diagnostic `knee-h9576-de50r` pending |

Effect of the label fixes on `data/oai/oai_labels.csv` (old kept as `oai_labels_v1.csv`): V03 change codes
488/488 resolved from earlier visits, V06 121 resolved and 135 masked. Cells changed: Medial OA 373
(101 false positives removed, 21 masked), Lateral OA 279, PF OA 346, Medial Meniscus 210, Lateral
Meniscus 94. All OAI runs so far (A-E) trained on the faulty targets and without the JPEG round-trip at
inference, so they did not test the OAI idea cleanly.

Second review round (same day): the first resolver filled V03 codes from V00 across the area-only V01
reading (74 conflicts with V01) and used other reading projects (32 V06 cells). Rewritten: same reading
project only, V01 area <= 1 resolves, otherwise UNRESOLVED; V05 used as history. V03: 26 + 78 resolved,
384 masked; V06: 85 + 34 resolved, 137 masked. Final changes vs the original table: Medial / Lateral / PF OA
369 / 278 / 350 cells (77 / 17 / 121 masked). Tests: tests/test_oai_labels.py (7 pass).

## Controlled comparison: release vs G0 vs G1 (2026-10-15/16)

Same start (release 0.949 checkpoint, sha256 7e5315da...), seed 11, batch 4, 4 epochs, 11,962 studies per
epoch (G0 resamples competition studies to match G1's steps), competition targets = release predictions
(self-distillation), fixed OAI holdout (holdout_seed 2026, 375 knees, targets in (0.4, 0.6) excluded),
corrected MOAKS targets (md5 f73b3d38...). G1 adds 7,613 OAI knee-visits with oai_weight 0.5.
Pod: secure RTX A6000, 10 vCPU, ~4.85 studies/s.

| Run | gold58 by epoch (start, 0-3) | OAI holdout by epoch | Final gold58 | Final OAI |
|---|---|---|---|---|
| release | 0.9226 | 0.9207 | 0.9226 | 0.9207 |
| G0 (no OAI) | 0.9181, 0.9259, 0.9234, 0.9223 | 0.9022, 0.9130, 0.9136, 0.9134 | 0.9223 | 0.9134 |
| G1 (+ corrected OAI) | 0.9194, 0.9177, 0.9241, 0.9247 | 0.9242, 0.9317, 0.9300, 0.9303 | **0.9247** | **0.9303** |

Per class, OAI holdout (G1 vs G0): ACL 0.968 / 0.970, Medial Meniscus 0.953 / 0.936, Lateral Meniscus
0.925 / 0.900, Medial OA 0.949 / 0.924, Lateral OA 0.874 / 0.791, PF OA 0.883 / 0.872, Effusion 0.987 /
0.972, Synovitis 0.959 / 0.922, Baker's 0.875 / 0.934. Gold58 (G1 vs G0): MCL 0.973 / 0.946, Synovitis
0.854 / 0.838, Fracture 0.921 / 0.949; others within +-0.01.

Reading: without OAI the model loses its OAI knowledge (holdout 0.9207 -> 0.9134); with corrected OAI it
gains (+0.0096 over release, +0.017 over G0). Gold58 difference (+0.0024) is within its noise. Both SWA
checkpoints are in gkdnzz31/knee-coat96-oai-g.

LB (0.9576 base, same per-finding blend: full Raptor half for ACL / Effusion / Baker's / Medial Meniscus /
Synovitis, 25% elsewhere; JPEG round-trip, success mask, required stage): G1 0.957, G0 0.957, D+E 0.957 (with
and without JPEG round-trip). Rank 279 / 6068 (0.957 block: 627 teams, positions ~240-627). The public LB
cannot separate these variants (SD ~0.003); the decision rests on the controlled validation (G1 > G0) and
the hidden-test diagnostic.

## New public base 0.958 (2026-10-16)

heliosli's "blend mine in" notebook (public 0.958) differs from the 0.9576 notebook only in `W_RAPTOR_C`
(Raptor-half weight per finding): Synovitis 0.80, Lateral OA 0.20, PF OA 0.60, Baker's 0.35, others 0.50.
`knee-h958` = our 0.9576 copy (private R6 copy) with these weights; `knee-h958-g1pl` = h958 + G1 with the
G1 notebook's A96 weights (cell diff vs `knee-h9576-g1pl`: the `W_RAPTOR_C` line only): LB **0.958**,
rank 143 / 6092 (was 279). The 0.958 block holds 379 teams (ranks ~117-495); above it 0.959 (40 teams) up to
0.968. Inside the block the order follows the unrounded score.

Seed runs on the pod (G1 recipe, same manifest hashes as G0/G1): G2 seed 23, G3 seed 37, then H1 = the same
recipe on nartaa's 224 checkpoint.

| Run | seed | gold58 ep2 / ep3 | OAI holdout ep2 / ep3 |
|---|---|---|---|
| G1 | 11 | 0.9241 / 0.9247 | 0.9300 / 0.9303 |
| G2 | 23 | 0.9227 / 0.9274 | 0.9357 / 0.9371 |

G2 reproduces the G1 gain over release (0.9225 / 0.9207 before fine-tuning).

SWA checkpoints scored with `scripts/eval_coat96.py` (same gold58 and OAI holdout):

| Checkpoint | gold58 | OAI holdout |
|---|---|---|
| G1 SWA | 0.9257 | 0.9306 |
| G2 SWA | 0.9259 | 0.9366 |
| G3 SWA (seed 37) | 0.9265 | 0.9395 |
| mean-probability ensemble G1-G3 | 0.9280 | 0.9374 |
| **weight soup G1-G3** (`coat96_g123soup_swa.pt`) | **0.9278** | **0.9375** |

The weight soup matches the 3-model ensemble at the inference cost of one model, so it replaces G1 in the
384 slot. Soup vs G1 per finding: OAI holdout equal or better for all 9 (PF OA +0.015, Baker's +0.021);
gold58 better for 8 of 12, lower for MCL (-0.016) and Synovitis (-0.011) (58 studies, noisy).

H1 (G1 recipe on the 224 checkpoint, self-distilled 224 targets; release-224 0.9138 / 0.9263): epoch 2
0.9177 / 0.9348.

Notebooks built for Oct 17: `knee-h958-soup` (h958 + soup in the 384 slot) and `knee-h958-soup-h1` (+ H1 as
our 224 stage, 0.65 / 0.35), both with the pre-registered A96 weights.

### Frozen 0.959 candidates (2026-10-17)

| Candidate | Kaggle kernel version | Notebook sha256 (16) | Data sources |
|---|---|---|---|
| `gkdnzz31/knee-h958-soup` | 1 (repo commit 46bd138) | 16725ca9e2212ab7 | knee-r6-fe055406, nartaa swa weights 20261007, orthofoundation fold0, knee-coat96-oai-g |
| `gkdnzz31/knee-h958v2-soup` | 1 (repo commit e1b54f9) | abc97eab42748faa | the same + goodpjw2008 2.5D reader dataset and notebook |

Our checkpoint in both: `coat96_g123soup_swa.pt`, sha256 ce00925d4b546611634e5fe0618b95a10b0c3f9335821a5ea9e323a9516eea61
(292,824,895 bytes), in the `knee-coat96-oai-g` version of 2026-10-16 18:29 UTC (6 files; the kernels were pushed
after it; the dataset is now at version 4, 2026-10-17 00:12, 7 files). Other sources at push time: knee-r6-fe055406
v1, nartaa swa weights v1, orthofoundation fold0 v1, goodpjw2008 2.5D reader v1 (all still at version 1). Later dataset versions keep every earlier file; the upload watcher now checks names and sizes, not counts,
and writes `upload_sha256.txt` next to the local copies.

### Tar-cache repeated-slice bug (found by the Oct 17 review)

`order_series_tar` returned one BytesIO per slice; when `build_volume` picked the same slice twice, the second read
hit the end of the stream and the slice stayed empty. Measured on the pod with `scripts/count_cache_gaps.py` (empty slices inside slots that have a
series; not all of them need be caused by the bug): 33 of 7,988 OAI knee-visits (33 partial slots), 40 slices in
total (mostly V06 axial: 20 knees, 26 slices); the folder-based competition caches, which never had the bug, show 4
studies / 5 slices, so some of these gaps are blank slices. Output saved in ~/knee_g_seeds/logs/cache_gaps.txt. Negligible, so no cache rebuild; fixed for the future (raw bytes, a fresh stream per read;
`tests/test_coat96_read.py`).

### Paired SWA evaluation and two no-training candidates (2026-10-17)

`eval_all` (per-study predictions in ~/knee_g_seeds/logs/eval_all_preds.npz) and the score ideas 1-2
(`eval_ideas_t1`, `eval_ideas_t08`), same gold58 / OAI holdout:

| Checkpoint | gold58 | OAI holdout |
|---|---|---|
| release (start) | 0.9226 | 0.9207 |
| G0 SWA (no OAI) | 0.9229 | 0.9134 |
| G1 SWA | 0.9257 | 0.9306 |
| G2 SWA | 0.9259 | 0.9366 |
| G3 SWA | 0.9265 | 0.9395 |
| G1-G3 soup | 0.9278 | 0.9374 |
| wise75 = 0.75 soup + 0.25 release weights | 0.9280 | 0.9363 |
| soup, attention temperature 0.8 | 0.9279 | 0.9385 |

With SWA on both sides, G1 vs G0 is +0.0028 gold / +0.017 OAI. wise75 recovers part of Fracture (0.935 -> 0.942) and
Lateral Meniscus (0.891 -> 0.896) but loses on PF OA and Lateral OA; T = 0.8 moves OAI +0.001 and gold +0.0001.
Neither is a clear gain over the soup, so neither goes into a submission (T = 0.8 would also need a patch of the
published inference model).

### Last-day probes and final selection (Oct 21)

All LB 0.959. Unrounded order read from Kaggle's auto-selection marker (unselect, read, re-select):
`knee-final-b-ralt` (V2 base + soup + second view for the soup and the public Raptor 384 / 224; stages stored
compressed to stay under Kaggle's ~1 MB notebook limit) > `knee-final-a-ralt` > `knee-final-b`; `knee-final-a-ralt-a50`
and `knee-final-b-ralt-a50` (A96 0.5 outside the pre-registered findings) are below final-b-ralt. A probability-space
blend was rejected before submitting: R6 outputs are near 0 / 1 while Raptor and the soup are soft, so averaging
probabilities would let R6 decide the order alone. Finals changed with the owner's go: **final-b-ralt + final-a-ralt**
(Kaggle's own top-2), verified 2/2 after reload. Rank 210 / 6328 at the time (the 0.959 block keeps growing).
Oct 22: `knee-final-b-ralt-a10` (A96 0.1 outside the pre-registered findings) and `knee-final-b-ralt-p75` (0.75 on
the pre-registered findings) both 0.959 and both below final-b-ralt on the marker. Raising or lowering the soup share
does not beat the pre-registered weights; finals stay final-b-ralt + final-a-ralt (verified 2/2 after reload).

**Private LB (Oct 23, preliminary):** 0.943, rank 345 / 6425 (public 223, so 122 places down). Every submission
from Oct 16 on, finals and probes alike, scored 0.943 private; the three-way blends 0.941, the soup alone 0.933, the
public 0.9576 base 0.941. On that base the matched pair G0 (competition targets only) and G1 (+ OAI MOAKS) both
scored 0.942 private, so fine-tuning itself gave +0.001 and the OAI effect is not visible at three decimals; the
h958 per-finding weights plus the soup gave the remaining +0.001 (h958 alone was never scored, so the two cannot be
separated). The second view and all late weight probes made no visible difference. The winner scored 0.960 private
(0.969 public).
Public -> private drop: ours 0.016 (0.959 -> 0.943), the same as the public 0.9576 base (0.957 -> 0.941), while the
45 private top-50 teams found on the public LB dropped 0.008-0.014 (median 0.011). The extra ~0.005 came with the
public base, whose per-finding weights were chosen on the public LB (30% of the test); our own component kept its
small gain on both splits. 256 teams had public >= 0.959, many on the same public base, so public rank 223 overstated
the position.

### Label audit vs the actual training targets (Oct 20)

An LLM review audited 18 non-gold training reports against the host's finding definitions (46 cells with a 0/1 draft, 15 abstained;
local only in outputs/label_audit_20261020/). `knee-teacher-audit` recomputed the release-384 predictions, i.e. the
G recipe's competition targets, for these studies. Mean |teacher - draft| 0.299 vs |PL3 - draft| 0.249; the teacher is
on the other side of 0.5 in 10 of 46 cells, mostly calling report-small findings positive (Baker's 4 cells incl. a
small partly ruptured cyst at 0.94, Contusion 2, MCL / menisci 3). So self-distillation pulled the competition
targets away from the official definitions for small cysts and reactive oedema. Fixing it needs severity-aware report
labels for all training studies and a retrain (with the OAI caches gone), which does not fit before Oct 22; noted as
the main label lesson.

### Size-aware Baker's / effusion targets: checked on gold58 first (Oct 20)

`scripts/report_severity.py` (multilingual keyword rules) tags each report's Baker's cyst / effusion mention as small,
large, absent or unclear (train: Baker's 362 small / 184 large; Effusion 851 small / 542 large). On gold58, against the
image-based expert labels: Baker's small 2 negative / 1 positive, large 4 / 4 positive; Effusion small 10 negative /
8 positive (44% positive), large 7 of 9 positive. A report saying "small effusion" is not a reliable negative for the
image label, and Baker's has only 3 small cases to check. The planned retrain (lower targets for report-small
findings) is therefore not supported and was not run; no pod was opened.

### Anatomical-mirror TTA for the public 224 model (Oct 20)

RSNA_Versia_7 v23 averages plain + anatomical-mirror predictions of the release-224 model; our 224 stage used the
plain view only. `knee-eval-224mirror` (Kaggle GPU, fp16, gold58): plain 0.9139, plain + mirror mean 0.9155, mirror
only 0.9145. On the findings where 224 has a share in our blend: Lateral Meniscus +0.008, Fracture +0.010, Lateral OA
+0.004, Medial OA +0.002, MCL -0.004, PF OA -0.002, Contusion -0.001. `--r224-mirror` patches the published 224
script (mirror of uint8 triplets before crop and normalisation, identical to coat96.mirror on synthetic data).
`knee-final-c` = final-a-ralt + this only; commit run fine (224 predictions change, 384 unchanged); submitted Oct 20.
LB 0.959. With final-a-ralt temporarily unselected, Kaggle's auto-selection marker pointed at final-a-ralt, not
final-c, i.e. final-c's unrounded public score is lower; the validated gain is small and only for the 224 component,
so the finals stay final-a-ralt + final-b (re-selected and verified 2/2 after reload).

### Second view for the public Raptor model (Oct 19)

`knee-eval-alt-release` (Kaggle GPU, fp16): release Raptor-384 on gold58 0.9223 primary -> 0.9231 with the mean of
primary + alternative-series view (MCL +0.014, Fracture +0.003, Lateral Meniscus -0.015); weak labels 0.9669 ->
0.9666. Smaller and more mixed than for our soup (+0.0017). `--raptor-alt` adds the same second view to the
published Raptor 384 / 224 scripts (same checkpoints and pinning, own output files); commit run fine.
`knee-final-a-ralt` (Oct 19): LB 0.959, scoring took ~6 h wall time (vs ~3-4 h for final-a), so the extra Raptor
stages cost time for no visible gain at three decimals. Rank 172 / 6219. (The ~6 h is scoring wall time including
queueing, not measured notebook runtime.) Revised on Oct 19 evening: the submissions page marked final-a-ralt and
final-b as Kaggle's auto-selection candidates, i.e. the two highest unrounded public scores; with per-stage time
guards (an extra stage is skipped with < 40 min left) final-a-ralt was selected as a final after all.

### Final candidates on the LB (Oct 19)

`knee-final-a` (h958 + G1-G3 soup + second view, not required) 0.959; `knee-final-b` (h958v2 with the Part B time
limit + the same, not required) 0.959. Both ran on the hidden test without errors in the final (non-required) form.
Rank 167 / 6206 (0.959 block ranks 122-183, 0.960 block 82-121, 0.961 block 64-81).

### A1: G1 recipe with view swap (2026-10-18)

| Epoch | A1 gold / OAI | G1 (same seed 11) gold / OAI |
|---|---|---|
| 0 | 0.9233 / 0.9157 | 0.9194 / 0.9242 |
| 1 | 0.9265 / 0.9354 | 0.9177 / 0.9317 |
| 2 | 0.9263 / 0.9303 | 0.9241 / 0.9300 |
| 3 | 0.9283 / 0.9321 | 0.9247 / 0.9303 |

A1 is ahead of G1 at the last epoch on both sets (+0.0036 gold, +0.0018 OAI, primary view only), so the checkpoint
passes and A2 / A3 continue. A2 (seed 23) epoch 3: 0.9265 / 0.9370 vs G2 0.9274 / 0.9371 (equal).
The evidence-based A96 variant scored 0.958: moving the soup's full share to other findings (and away from
Synovitis / Baker's) is worse than the pre-registered weights, which stay.

A1-A3 SWA (gold / OAI): 0.9285 / 0.9315, 0.9248 / 0.9373, 0.9230 / 0.9394 (G1-G3: 0.9257 / 0.9306, 0.9259 / 0.9366,
0.9265 / 0.9395). Soups, `eval_asoup` and `eval_alt2` (bf16):

| Soup | gold58 primary | gold58 primary + second view | OAI | weak alt primary / mean |
|---|---|---|---|---|
| G1-G3 | 0.9278 | 0.9297 | 0.9374 | 0.9582 / 0.9583 |
| A1-A3 (view swap) | 0.9257 | 0.9262 | 0.9381 | 0.9578 / 0.9579 |

The view-swap soup is worse on gold58 with and without the second view, so it is not used (pre-registered rule:
not worse than the G1-G3 soup on both). A1 alone looked better than G1, but A2 / A3 did not repeat it; picking a
single seed on gold58 would be selection on the validation set. All checkpoints backed up; pod stopped Oct 18 ~20:40.
The night gate first printed DO NOT USE only because its expected list for eval_asoup.json left out the soup
itself (a spec error in the call, not missing output); rerun with the full list: USABLE, 14 checkpoints verified on
Kaggle by name and size. The decision does not change (the soup is worse on validation).

### Window filter (2026-10-18)

G1-G3 soup, bf16, same gold58 / OAI holdout (reference 0.9278 / 0.9374): dropping the 8 windows that span two slots
(94 -> 86) 0.9276 / 0.9375; also dropping windows with an empty slice 0.9280 / 0.9375. No fallback to the full list
on these masks; DataLoader workers use fork, so the filter reached them. No gain; dropped.

### LB, Oct 18

| Notebook | LB |
|---|---|
| `knee-h958-soup3w` (three-way, soup 0.30 of the final rank) | 0.957 |
| `knee-h958v2-soup3w` | 0.957 |
| `knee-h958-soup-alt` (soup + alternative-series second view) | 0.959 |
| `knee-h958v2-soup-alt` | 0.959 |
| `knee-h958-soup-alt-ev` (A96 1.0 where soup beats release on gold58 by >= 0.005, OAI not worse) | 0.958 |

The three-way blend is worse on both bases (0.957 vs 0.959): our soup works best replacing part of the Raptor slot,
not as its own share; dropped. The second view keeps 0.959 (not separated at three decimals; +0.0017 on gold58).
Rank 150 / 6165 on Oct 18 (0.960 block now ranks 70-104, 0.959 block 105-163; rank 20 = 0.964). No public notebook
above 0.958 displayed.

### Second view from alternative series (2026-10-17/18)

1,599 of 4,407 train studies (36%; 31 of gold58) have an unused series with the same plane / fluid / fat flags as a
primary pick (`coat96.pick_alternative_series`). Cache `knee-build-cache-coat96alt` (Kaggle CPU, 22 min, 0 failures,
93.4 filled slices per study). `knee-eval-alt` (Kaggle GPU, fp16) scored the G1-G3 soup:

| Set | primary view | mean of primary + alternative |
|---|---|---|
| gold58 (31 studies changed) | 0.9282 | **0.9299** |
| weak labels (pl3), 1,568 alternative studies | 0.9582 | 0.9583 |

Gold gains: MCL +0.012, Synovitis +0.010, Lateral OA +0.008, Fracture +0.006; PF OA -0.009. The weak-label set
favours the primary view (seen in training), so a tie there still suggests the second view carries information.
Notebook: `--alt` adds a copy of our 384 stage that passes the reader only the chosen series in slot order, so the
reader's own picker selects the alternatives (checked equal to the validated picker on all 1,599 studies); studies
without an alternative are skipped and keep the primary prediction. Commit run: 1 of 3 public studies used the
second view, valid submission. `knee-h958-soup-alt` submitted Oct 18.

### G5 / G6 and the 5-seed soup (2026-10-17 evening)

G5 (seed 53) epoch 3: gold 0.9244, OAI 0.9359; G6 (seed 67) epoch 2: 0.9245 / 0.9350. Soup of G1, G2, G3, G5, G6:
gold 0.9271, OAI 0.9379 vs the 3-seed soup 0.9278 / 0.9374. `scripts/check_g_outputs.py`: USABLE (members exactly
g1 g2 g3 g5 g6; all 10 checkpoints backed up locally and in `knee-coat96-oai-g` by name + size; eval_all / eval_soup5
JSON + NPZ complete; output in ~/knee_g_seeds/logs/gate_soup5.txt). Backup verified; pod stopped 19:54 UTC.
Pre-registered rule (gold + OAI change >= 0, neither below -0.002): -0.0007 + 0.0005 < 0, so soup5 is not used;
more seeds give no measurable gain beyond three. Oct 18 submissions: the two three-way blends only.

### LB, Oct 17 (all five submitted 00:03 UTC)

| Notebook | LB |
|---|---|
| `knee-h958-soup` (h958 + G1-G3 soup, pre-registered A96) | **0.959** |
| `knee-h958v2-soup` (heliosli V2 weights + 2.5D reader + soup) | **0.959** |
| `knee-h958-soup-h1` (+ H1 224 stage) | 0.958 |
| `knee-h958-soup-a100` (soup takes the whole Raptor slot) | 0.958 |
| `knee-soup-alone` (our soup only) | 0.949 |

Rank 129 / 6133 (0.959 block: 48 teams, ranks 90-137; the 0.958 block grew to 785 teams). Our soup alone shows the same rounded
score as the release Raptor model alone (0.949), while replacing G1 by the soup moved the blend from 0.958 to
0.959: the soup helps inside the blend (diversity and/or slightly better ranking; the rounded standalone scores
cannot separate the two). Giving it the whole Raptor slot (a100) or adding H1
did not help at three decimals.

### G4: aggressive recipe (oai_weight 1.0, 6 epochs, seed 41)

Epochs 0-5 gold / OAI: 0.9243/0.9109, 0.9223/0.9145, 0.9233/0.9343, 0.9227/0.9382, 0.9233/0.9345, 0.9203/0.9355.
SWA ep4-5: gold 0.9220, OAI 0.9349, below the G1-G3 soup (0.9278 / 0.9375) and below G1 on gold. Doubling the OAI
weight and training longer does not help; the 0.5 / 4-epoch recipe stays. No extra credit for this direction.

### Soup vs release per finding (2026-10-16 night)

| Finding | gold58 release -> soup | OAI holdout release -> soup |
|---|---|---|
| ACL | 0.964 -> 0.969 | 0.963 -> 0.974 |
| MCL | 0.943 -> 0.961 | - |
| Medial Meniscus | 0.983 -> 0.988 | 0.932 -> 0.953 |
| Lateral Meniscus | 0.896 -> 0.891 | 0.872 -> 0.928 |
| Medial OA | 0.981 -> 0.988 | 0.920 -> 0.948 |
| Lateral OA | 0.774 -> 0.789 | 0.863 -> 0.881 |
| PF OA | 0.857 -> 0.874 | 0.895 -> 0.899 |
| Effusion | 0.974 -> 0.981 | 0.971 -> 0.988 |
| Synovitis | 0.828 -> 0.843 | 0.971 -> 0.967 |
| Baker's | 0.976 -> 0.980 | 0.900 -> 0.901 |
| Contusion | 0.935 -> 0.934 | - |
| Fracture | 0.958 -> 0.935 | - |

The soup beats the release 384 model on 9 of 12 findings on gold58 and 7 of 9 on the OAI holdout (in-domain, so
optimistic). Fracture is the only clear loss (few positives). This supports giving the soup most of the Raptor
slot everywhere except Fracture / Contusion; `knee-h958-soup-a100` tests the extreme.

`knee-h958v2(-soup)`: heliosli's V2 weights (Synovitis 0.85, Lateral OA 0.15, PF OA 0.65, Baker's 0.30) plus
goodpjw2008's 2.5D reader as a small-weight Part B cell (Apache 2.0, sha-checked, keeps the stage-3 file on
failure), on our private R6 copy. Prepared in case V2 scores above 0.958.

## Infrastructure lessons

- Kaggle allows 2 concurrent GPU sessions; a third push fails with "Maximum batch GPU session count of 2 reached". Queue pushes with a retry loop.
- Weekly GPU quota (~30 h) runs out fast: a 5-fold run is 3-6 h, a CoAtNet-384 fine-tune ~4.5 h.
- Build caches in CPU kernels (free): 22-55 min for 4407 studies.
- Kaggle kernel output is capped at ~20 GB: 80 x 320 px needs JPEG (q95, ~3x smaller, 11.8 GB). Apply the same JPEG round-trip at test time so train and test pixels match; encode from the original pixels, not from an already round-tripped stack.
- Never `glob('/kaggle/input/**')`: it walks the 500 GB competition mount (9 min). Walk only attached dataset/notebook folders.
- Kaggle may unpack a dataset folder flat: rebuild the `knee/` package folder at runtime.
- Kaggle OAuth logins (`kaggle auth login`) expire after a few hours; for unattended work use a
  non-expiring API token in `~/.kaggle/access_token`.
- `kaggle competitions submit` for code competitions needs both `-v <version>` and `-f submission.csv`; an early 403 was transient.
- A notebook's live log is not available via the CLI until it finishes.
- Inference with many groups and three preprocessing layouts takes ~4 h on the hidden test; keep an eye on the 9 h limit.
- Test the full train/infer path locally on fake data before spending GPU time; it caught several bugs (empty-study NaN, persistent workers, layout mismatch).

## Rules decisions

- LLM APIs for labels: allowed (host).
- KneeCoT: prohibited. OAI: ambiguous; the owner accepted the risk on 2026-10-12, so OAI data and OAI-trained
  public weights are used (with the NDA acknowledgement in any write-up).
- Public OAI-free Kaggle weights/notebooks (Raptor, community stack): used with credit.
