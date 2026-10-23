"""Score coat96 checkpoints on gold58 and the fixed OAI holdout (the same sets as the fine-tune runs),
plus the mean-probability ensemble of all of them; optionally write a weight soup first.
Per-study predictions, targets and ids are saved next to --out as <out>_preds.npz (OAI ids: ID-SIDE-VISIT,
so paired bootstraps can resample subjects).

Example (on the pod):
    python3 scripts/eval_coat96.py --soup /workspace/runs/soup/coat96_g123_soup.pt \
        --soup-from /workspace/runs/g1/coat96_g1_swa.pt /workspace/runs/g2/coat96_g2_swa.pt /workspace/runs/g3/coat96_g3_swa.pt \
        --ckpts /workspace/runs/g1/coat96_g1_swa.pt /workspace/runs/g2/coat96_g2_swa.pt /workspace/runs/g3/coat96_g3_swa.pt \
        --comp-cache /workspace/cache-coat96-0 /workspace/cache-coat96-1 --comp-labels /workspace/labels/coat96_targets_distill.csv \
        --oai-cache /workspace/cache-oai-* --oai-labels data/oai/oai_labels.csv --out /workspace/runs/eval_g123.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402
from knee.coat96_train import CachedStudies, competition_sets, masked_auc, oai_sets, predict, swa  # noqa: E402
from knee.constants import LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--soup", help="write the weight average of --soup-from here and score it too")
    ap.add_argument("--soup-from", nargs="*", default=[])
    ap.add_argument("--comp-cache", nargs="+", required=True)
    ap.add_argument("--comp-labels", required=True)
    ap.add_argument("--oai-cache", nargs="*", default=[])
    ap.add_argument("--oai-labels")
    ap.add_argument("--holdout", type=float, default=0.05)
    ap.add_argument("--holdout-seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--att-temp", type=float, default=1.0, help="attention softmax temperature (1.0 = as trained)")
    ap.add_argument("--window-filter", default="none", choices=["none", "slot", "slot+empty"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    coat96.WINDOW_FILTER = args.window_filter
    ckpts = list(args.ckpts)
    if args.soup:
        Path(args.soup).parent.mkdir(parents=True, exist_ok=True)
        swa(args.soup_from, args.soup)
        ckpts.append(args.soup)
    _, gold = competition_sets(args.comp_cache, pd.read_csv(args.comp_labels))
    sets = {"gold": [CachedStudies(d, p["row"], p[LABELS].values, 1.0, None, False) for d, p in gold if len(p)]}
    ids = {"gold": [u for _, p in gold if len(p) for u in p["StudyInstanceUID"]]}
    if args.oai_cache:
        oai = pd.read_csv(args.oai_labels, dtype={"ID": str})
        _, val = oai_sets(args.oai_cache, oai, args.holdout, args.holdout_seed)
        sets["oai"] = [CachedStudies(d, p["row"], p[LABELS].values, 1.0, None, False) for d, p in val if len(p)]
        ids["oai"] = [f"{i}-{s}-{v}" for _, p in val if len(p) for i, s, v in zip(p["ID"], p["SIDE"], p["VISIT"])]

    res, probs, ys, saved = {}, {k: [] for k in sets}, {}, {}
    for ck in ckpts:
        model, view = coat96.load_model_view(ck, "cuda")
        model.att_temp = args.att_temp
        res[ck] = {}
        for k, ds in sets.items():
            y, p = predict(model, ds, "cuda", args.workers, view)
            if ck != args.soup:
                probs[k].append(p)
            per = masked_auc(y, p)
            res[ck][k] = float(np.nanmean(list(per.values())))
            res[ck][k + "_per_label"] = per
            ys[k] = y  # same order for every checkpoint
            saved[f"{k}__{Path(ck).stem}"] = p.astype(np.float32)
        print(f"{Path(ck).name}: " + " ".join(f"{k} {res[ck][k]:.4f}" for k in ("gold", "oai") if k in res[ck]), flush=True)
        del model
    if len(ckpts) - bool(args.soup) > 1:
        res["mean_prob"] = {}
        for k in [k for k in probs if probs[k]]:
            per = masked_auc(ys[k], np.mean(probs[k], axis=0))
            res["mean_prob"][k] = float(np.nanmean(list(per.values())))
            res["mean_prob"][k + "_per_label"] = per
        print("mean-prob ensemble: " + " ".join(f"{k} {v:.4f}" for k, v in res["mean_prob"].items()
                                                 if not k.endswith("per_label")), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    # per-study predictions for paired comparisons: ids and targets per set, one prediction array per checkpoint
    for k in sets:
        saved[f"{k}__target"] = ys[k].astype(np.float32)
        saved[f"{k}__id"] = np.array(ids[k])
    import torch.multiprocessing as mp
    print(f"window filter {coat96.WINDOW_FILTER} | worker start method {mp.get_start_method()} | autocast bf16", flush=True)
    np.savez_compressed(str(Path(args.out).with_suffix("")) + "_preds.npz", labels=np.array(LABELS),
                        window_filter=np.array(coat96.WINDOW_FILTER), start_method=np.array(mp.get_start_method()), **saved)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
