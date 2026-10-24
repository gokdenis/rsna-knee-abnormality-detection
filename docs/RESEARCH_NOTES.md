# Forum + Public Notebook Notes (2026-10-09)

## Official host rulings
- **Using an LLM API to extract labels from reports is ALLOWED** (OpenAI/Anthropic/Google). Disc 733965.
- LLM-derived labels/embeddings may be used for training (reports are not available at test time).
- **KneeCoT is PROHIBITED** (requires an institutional agreement). Disc 734109.
- **OAI:** "not a rule violation if accessible without institutional review/sign-off and free of charge". Still ambiguous; some got individual NDA approval ("Institutional Approval Required: No"), some countries are blocked. No explicit "yes" → **risky, avoid**. Disc 741819, 743416.
- MRNet (no derivative works), KneeMRI (CC BY-NC-ND), fastMRI (DSA) → contested, avoid.

## Data facts
- 4,407 training studies; **only 58 expert-labeled** ("gold"). Every gold study has ≥1 positive, mean 4.14 positives.
- Reports in ~9 languages.
- Labeling criteria (host): borderline → negative. ACL = high-grade partial/full tear; MCL = high-grade acute; meniscus = signal reaching the surface on ≥2 images; OA = >50% cartilage loss over ≥1 cm; effusion/Baker's = moderate-large.
- Some gold labels contradict the report (e.g. "No effusion" but Effusion=1) → report ≠ ground truth; annotators read the images.
- ~7 studies contain both knees in the same DICOM.
- File names (SOP UIDs) are random → sort slices by ImagePositionPatient projected onto the slice normal.
- Field of view varies widely (71 distinct values, median 160 mm) → **crop in millimetres** (140–154 mm).
- Metadata-only model reaches 0.60 AUC → no shortcut.

## Labels are the main bottleneck (forum consensus)
- Regex: 0.81 gold AUC; LLM: 0.88; best public set ~0.89.
- **"Not addressed" ≠ negative:** Synovitis is absent from 84% of reports yet 34% of those are gold-positive; Baker's absent from report → only 3% positive. Treat silence per finding.
- Filling undecided Synovitis cells from Effusion: 0.678 → 0.790. Synovitis is everyone's worst class (0.6–0.79).
- Public label sets: stevenleehans (v2/v4), pilkwang, lixin73, laymond (qwen3-8b), yunus, riadmohamed42 (JEV + 16 extra verified studies → gold-74), karttikjangid05 consensus (CC0). No statistically clear difference between the 8 sets on gold.
- **Pseudo-label loop** gives large gains: train a teacher on weak labels → relabel with teacher/OOF predictions → train a student. Many 0.95+ single models do this.
- API labeling cost is low (one participant reported <$5 total).

## Model recipes that work
- **2.5D**: 3 neighbouring slices = RGB channels; fixed slots per series/plane.
- **Per-finding attention MIL pooling** (12 separate attention heads) → the most important component (Raptor).
- Raptor (dreaddevelopment): 5 slots (Sag fluid 18 / Sag non-fluid 14 / Cor fluid 12 / Cor 8 / Ax 12) = 64 slices, 6–94% span, 140 mm crop, 336 px, CoAtNet `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` @384. Soft BCE. Train: 12 random windows; eval: all windows. Average of top-3 epoch weights. → 0.924–0.938.
  - Slice **density** matters: 96 slices / 94 windows > 64 / 42 (+0.003–0.008).
- 0.949 (Danial Zakaria): Raptor + JEV labels + **OAI** + native 384 + 320 center crop + anatomical-mirror TTA. (Uses OAI → risky)
- Simple CNNs are strong: ResNet50 224 px 5-fold 0.954; ConvNeXt-T 5-fold 0.959; EffV2-S 0.956; DINOv2 ViT-S 0.951; Qwen3.5-2B VLM fine-tune 0.950.
- Scaling the encoder does not help (DINOv2 S→B +0.001). Match normalization constants to the pretrained checkpoint.
- RadImageNet init helped Synovitis in one report (0.62→0.78); another found it worse than ImageNet.
- Width flip on all planes hurts; **anatomical mirror** (reverse slice order on sagittal) +0.001.
- Augmentation: light (laterality carries signal).

## Validation
- 58 gold is too small (cannot resolve <0.02); public LB (~390 studies) is noisy.
- Recommended (Tucker Arrants, #19): **5-fold OOF against report labels**, ~0.003 noise threshold; submit only real improvements. Group folds by report text.
- Expect a large shake-up.

## Inference / efficiency
- Notebook limit 9 h. Public blend stacks take 6.5–8 h!
- Speed-ups: parse DICOM headers without pydicom, 16-thread reads, percentiles on a subsample, fp16 weights, one model per GPU (no DataParallel). 40 min → <10 min.
- Cache: 4407 × 6 slots × 9 slices × 224² uint8 = 11 GiB (instead of 500 GB of DICOM).

## Public notebooks
- Top: heliosli 0.954 = 0.949 OAI-Raptor (0.7) + ConvNeXt-S (0.3) rank blend.
- Most 0.95 notebooks are forks of the same blends; beware of "AI slop".
- Useful resources:
  - dreaddevelopment/knee-mri-training-the-twelve-finding-model (training code) + knee-raptor-corpus datasets (preprocessed, 44×336×336 uint8 per study)
  - dreaddevelopment/knee-mri-twelve-findings-from-a-single-model (inference)
  - goodpjw2008 ConvNeXt-T + study transformer, 256 px, 3-fold 0.929 standalone
  - stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache
  - karttikjangid05 label sets compared (CC0 consensus)
  - pilkwang/rsna-knee-baseline-v1, rsna-knee-llm-labels

## Score expectations
- Current public #1 is 0.964; gold threshold estimated ~0.96.
