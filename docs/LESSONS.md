# Lessons for future competitions

General takeaways from this competition that apply beyond it. Competition-specific results live in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Strategy

- **Track the public notebooks daily from day one.** In this competition the best copyable public notebook
  went 0.937 -> 0.949 -> 0.957 within a few days. Building our own models from scratch (~0.93) was overtaken
  before it paid off. Start from the strongest public pipeline and spend effort on what it lacks.
- **Public notebooks raise the floor, not your rank.** Hundreds of teams end up tied at the best public
  score. A lasting edge needs information or compute the public pipelines do not use (extra expert-labelled
  data, better labels, a different modality), built as a component that can be added to *any* base.
- **Design your work to be base-agnostic.** Keep your own model as a separate stage that writes its own
  predictions and is blended last, so swapping in a newer public base costs one run.
- **Weak members hurt strong blends.** Models ~0.01-0.02 below the base and highly correlated with it lower
  the blend. Only add a member that has proven itself alone or in a blend on the leaderboard.
- **When labels are noisy, the label source was our ceiling.** Our students did not beat their pseudo-label teacher,
  and more resolution, slices or architectures did not help; better labels beat bigger models here. This is not a
  general law (Noisy Student is a counter-example), just what our recipe showed.
- **Displayed scores are truncated.** Ties at the shown precision are ranked by the unrounded score, so a
  "no change" submission can still move you up many places.
- **Do the paperwork early.** External data that needs approval (here: NDA/OAI) took a day to obtain;
  request it in the first week if it might matter.

- **Map report text to the official finding definitions, not to mentions.** The host's definitions require e.g.
  moderate/large effusion and Baker's cyst, and OA by depth and extent of cartilage loss; reports that say "small
  cyst" or "minimal effusion" got 0.64-0.76 soft targets in the public label blends (Oct 20 audit of 18 reports).
  Extract severity, size, location and uncertainty, abstain on contradictions, and do this in week one: late label
  fixes need a retrain, which a fine-tune on a model's own predictions (self-distillation) does not pick up.
- **When most training labels come from free-text reports, invest in label extraction first.** Our image
  models did not beat the 0.926 gold level of the public weak labels with our recipe (a plateau, not a proof
  of a hard ceiling); strong teams build their own
  labels (several LLMs, multilingual reports, clinician review). Budget an LLM API and the first week for it.
- **Check the strongest public notebooks every morning.** A new public version (0.957 -> 0.9576) moved
  ~125 teams past us in a day before we noticed.
- **Start early.** Teams at the top had two months; two weeks is enough to follow the public frontier,
  not to build a clearly better pipeline.

## Kaggle mechanics

- Code competitions: 5 submissions/day (reset 00:00 UTC), notebook <= 9 h, no internet. Submit with
  `kaggle competitions submit -k <user/kernel> -v <version> -f submission.csv`.
- 2 concurrent GPU sessions, ~30 GPU-hours per week, ~20 GB output per notebook. Build caches on CPU.
- Copying a public notebook: pull it with `kaggle kernels pull -m`, change the id/title in the metadata,
  keep the same dataset sources and docker image, and push as private.
- Count the day's submissions (UTC day, including morning ones) before planning more; a verified notebook can
  still be rejected with HTTP 400 once the 5 are used.
- Never glob the full `/kaggle/input` tree; competition mounts can be hundreds of GB.

## Rented GPUs and external data

- Check that the CLI and the browser are logged into the same cloud account before loading credit; a
  "no instances available" error can really mean "balance too low" on the CLI's account.
- Cheap community GPUs can run out of stock for hours; budget for the secure price and decide quickly.
- Measure throughput on the rented GPU first (a 2-minute benchmark) and size the plan from it.
- Download only what you need: find the metadata file of a data package first (here `image03.txt`) and
  select files from it instead of pulling whole terabyte packages.
- Make every long download resumable (`curl -C -`, retries); multi-GB HTTP downloads break.
- Process large external data in stages (download -> unpack -> compact cache -> delete raw) so the disk
  never needs to hold everything.
- A stopped pod keeps only its volume (`/workspace`); the container disk is reset, so installed packages and
  `~/.kaggle` are gone after a restart. Keep setup in a script, pin torch (an unpinned `pip install timm`
  can upgrade torch), and check imports before launching a long job. A pod volume (not a network volume)
  ties the pod to one host: "not enough free GPUs" can block the restart, so retry or plan for it.
- Data portals often have a separate API password (NDA: profile -> Update Password), different from the
  single-sign-on login.

## Working practice

- **Map external labels from the data dictionary, value by value.** MOAKS follow-up visits use +/-0.5 as
  "change within grade" codes; parsing them as grades silently created ~1,000 wrong OA targets.
- **Every new-data experiment needs a matched control** (same start, seed and optimizer steps), and every
  run must be scored on the same fixed holdout.
- **Make train and inference pixels identical** (here: JPEG q95 cache vs raw DICOM at test time).
- **Weight soups of seed runs are a free ensemble.** Fine-tunes from the same start with a small learning rate stay
  in one basin: averaging three seeds' weights matched the 3-model probability ensemble on validation at one
  model's inference cost and moved the LB blend from 0.958 to 0.959.
- **Score your component alone once.** Ours alone showed 0.949, the same rounded score as the public model it
  replaced; giving it the whole slot did not help, so it earns its place inside the blend, not as a replacement. A single diagnostic submission saved several blind ones.
- **Paired differences can be less noisy than single scores.** Two similar submissions on the same public rows
  have correlated errors, and Var(A-B) = Var(A) + Var(B) - 2 Cov(A,B) can be below either score's variance;
  measure it from paired predictions. A rounded tie
  means "not separated at this precision", not "equivalent"; keep per-study validation predictions to measure
  paired differences.
- **In-memory file objects are single-use.** A shared BytesIO consumed by one reader is empty for the next (and
  races across threads); keep raw bytes and open a fresh stream per read.
- **Have an independent review before scaling an approach**; it found errors our own checks missed.

- Test the full train/infer path locally on fake data before spending GPU time.
- Keep a running log of every result and lesson in the repo; it survives context loss and lets you
  reuse what you learned.
- Treat downloaded data and public code as untrusted: separate directories, isolated Python (`-I`).
