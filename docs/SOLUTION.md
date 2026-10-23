# RSNA Knee Abnormality Detection — solution summary (team YVL)

Public LB 0.959 (rank 223 of 6,425). Private LB 0.943, rank 345 of 6,425 (top 5.4%, bronze range; preliminary
until Kaggle verifies the results). Both finals and every late candidate scored 0.943 on the private split.

This entry joined in the last ten days of a two-month competition. It is built on the strongest public pipelines
and adds one component that no public notebook had: a CoAtNet model fine-tuned with expert MRI readings from the
Osteoarthritis Initiative (OAI). The work went into making that component correct, measuring its value with
controlled experiments, and adding it to the public blend safely.

## 1. Base: public models, private copies

- **Raptor slot:** nartaa's two OAI-trained CoAtNet-RMLP-2 MIL checkpoints (384 px with anatomical-mirror TTA, and
  a 224 px view of a 0.8 centre crop), 0.65 / 0.35 rank blend.
- **Compact slot:** heliosli's R6 ConvNeXt-S bundle (4 folds, flip TTA, own effusion / Baker's stackers). The public
  dataset was later overwritten, so a private copy of the original version is used.
- **Blend:** per-finding percentile-rank blend of the two slots with heliosli's per-finding weights (public 0.958).
  One final also adds goodpjw2008's 2.5D ConvNeXt reader at small weights.

## 2. The added component: OAI MOAKS fine-tune

**Data.** OAI knee MRIs (NDA access) from visits V00, V01, V03 and V06, mapped onto the competition's five series
slots (sagittal IW-TSE, sagittal DESS, coronal IW-TSE, coronal MPR / FLASH, axial MPR) with exactly the same 96-slice
preprocessing as the public 384 model (verified byte for byte on competition studies). Only the needed series of
labelled knees were downloaded and cached; raw images were deleted after caching.

**Labels.** MOAKS expert semi-quantitative readings were mapped to nine of the twelve findings: cartilage loss per
compartment (medial / lateral / patellofemoral OA), meniscal tears, effusion-synovitis, Hoffa synovitis, popliteal
cyst, ACL tear. Two independent reviews of the mapping found real errors that were fixed before training:
- follow-up visits use +/-0.5 "change within grade" codes, which are resolved only against the same reader's earlier
  reading (otherwise masked) instead of being parsed as grades;
- meniscal code 7 (progressive partial maceration) counts as a tear;
- MCL, contusion and fracture get no OAI target (masked loss).
Result: 7,988 labelled knee-visits, regression tests for the mapping.

**Recipe.** Start from nartaa's 384 checkpoint. Competition targets are that checkpoint's own predictions
(self-distillation, so the model keeps what it knows); OAI targets are the MOAKS labels with a masked loss where
`oai_weight` really scales the OAI gradient (loss = weighted sum / number of known cells). 4 epochs, batch 4, 16
training windows per study, AdamW (backbone 2e-5, head 2e-4), cosine schedule, SWA of epochs 2-3, JPEG q95 cache
with the same JPEG round-trip at inference.

**Controlled test.** Same start, seed, steps and fixed OAI holdout (375 knees, subject-level split):

| Model (SWA) | gold58 | OAI holdout |
|---|---|---|
| release checkpoint | 0.9226 | 0.9207 |
| G0: competition targets only | 0.9229 | 0.9134 |
| G1: + corrected OAI MOAKS | **0.9257** | **0.9306** |

**Weight soup.** Three seeds of the G1 recipe averaged in weight space (gold58 0.9278, OAI 0.9375) match the
three-model probability ensemble at one model's inference cost. In the blend (replacing part of the Raptor slot,
pre-registered per-finding shares) it moved the public LB from 0.958 to 0.959.
On the private split the matched G0 / G1 pair tied (0.942 each, vs 0.941 for the base), so the OAI gain seen on the
holdouts was too small to show on the hidden test at three decimals.

## 3. Second view from unused series

36% of studies contain a second series with the same plane / fluid / fat-suppression flags that the five-slot layout
never reads. The final notebooks also run our soup (and, in one final, the public Raptor 384 / 224 models) on a
volume built from these alternative series and average the two probabilities (gold58 for the soup: 0.9282 ->
0.9299 at equal precision). The reader is given only the chosen series in slot order, so its own picker selects
them; studies without an alternative keep their single-view prediction.

## 4. Robust finals

Both finals are built so that a failing or slow extra stage cannot break the submission: every added stage has a
time guard and a per-study success mask, and failures fall back to the public base. Selected finals:
- h958 base + G1-G3 soup + second view for the soup and the public Raptor models;
- h958v2 base (V2 per-finding weights + 2.5D reader with a time limit) + the same soup + second view for the soup
  and the public Raptor models.

## 5. What did not help (measured, then dropped)

| Idea | Result |
|---|---|
| Our own ConvNeXt / CoAtNet models trained on report labels | 0.90-0.91 gold; lowered strong blends |
| Blind LLM relabelling of reports | 0.838 vs 0.855 for the best public labels on gold58 |
| More seeds (5-seed soup), higher OAI weight + longer training | no validation gain |
| Training on the alternative-series views (view swap) | soup worse on gold58 |
| Soup as its own share of the final blend (three-way) | LB 0.957 vs 0.959 |
| Window filtering, attention temperature, weight interpolation to the release model | no gain |
| 224 mirror TTA for the public 224 model | +0.0016 for that model on gold58, not visible on the LB |
| Size-aware report targets for Baker's / effusion | "small effusion" in a report is positive in 44% of gold image labels |

## 6. Lessons

- Start from the strongest public pipeline on day one and spend effort on what it lacks.
- Every new-data experiment needs a matched control and a fixed holdout; label mappings need value-by-value checks
  and an independent review.
- The bottleneck is label quality against the official finding definitions (severity, size, image-based reading),
  not model size. Self-distilled targets inherit the teacher's view of small findings.
- Score a component alone once, and judge public differences below ~0.002 as direction, not proof.

## Acknowledgements

Built on public work with thanks: dreaddevelopment's Raptor CoAtNet/MIL recipe; nartaa's published SWA weights and
inference notebooks (384 anatomical mirror, 224 crop); heliosli's R6 bundle and blend notebooks; goodpjw2008's 2.5D
ConvNeXt reader (Apache 2.0); the public report-label sets of stevenleehans, pilkwang and riadmohamed42 used by these
models.

Data and/or research tools used in the preparation of this work were obtained and analyzed from the controlled access
datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive
(NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National
Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace
of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): NDA Study 3427, DOI [10.15154/cb9j-k762](https://doi.org/10.15154/cb9j-k762).
