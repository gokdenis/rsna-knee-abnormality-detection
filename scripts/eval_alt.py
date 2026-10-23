"""Second-view evaluation on the pod: for each checkpoint, gold58 and weak-label AUC of the primary view and of the
mean of primary + alternative-series view (same logic as the knee-eval-alt Kaggle kernel).

Usage: python3 scripts/eval_alt.py --ckpts a.pt b.pt --prim /workspace/cache-coat96-0 /workspace/cache-coat96-1 \
    --alt /workspace/cache-coat96alt --labels /workspace/labels/train_labels_pl3.csv --out /workspace/logs/eval_alt2.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402
from knee.coat96_train import CachedStudies, masked_auc, predict  # noqa: E402
from knee.constants import LABELS  # noqa: E402


def ids_of(d):
    return np.load(f"{d}/ids.npy", allow_pickle=True).astype(str)


def run(model, view, d, rows, workers):
    ds = CachedStudies(d, np.asarray(rows), np.full((len(rows), len(LABELS)), np.nan), 1.0, None, False)
    return predict(model, [ds], "cuda", workers, view)[1]


def views(model, view, prim, alt, ids, workers):
    """(primary predictions, alternative predictions or NaN rows) for the given study ids."""
    p = np.full((len(ids), len(LABELS)), np.nan)
    for d in prim:
        row = {s: i for i, s in enumerate(ids_of(d))}
        sel = [k for k, s in enumerate(ids) if s in row]
        if sel:
            p[sel] = run(model, view, d, [row[ids[k]] for k in sel], workers)
    a = np.full_like(p, np.nan)
    row = {s: i for i, s in enumerate(ids_of(alt))}
    sel = [k for k, s in enumerate(ids) if s in row]
    if sel:
        a[sel] = run(model, view, alt, [row[ids[k]] for k in sel], workers)
    return p, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--prim", nargs="+", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lab = pd.read_csv(args.labels).set_index("StudyInstanceUID")
    gold = lab.index[lab["is_gold"] == 1].astype(str).tolist()
    alt_ids = ids_of(args.alt)
    weak = [s for s in alt_ids if lab.loc[s, "is_gold"] == 0]
    y_gold = lab.loc[gold, LABELS].values.astype(float)
    y_weak = (lab.loc[weak, LABELS].values >= 0.5).astype(float)
    res, saved = {}, {"gold_id": np.array(gold), "weak_id": np.array(weak), "labels": np.array(LABELS)}
    for ck in args.ckpts:
        model, view = coat96.load_model_view(ck, "cuda")
        r = {}
        for name, ids, y in (("gold58", gold, y_gold), ("weak_alt", weak, y_weak)):
            p, a = views(model, view, args.prim, args.alt, ids, args.workers)
            avg = np.where(np.isnan(a), p, 0.5 * (p + np.nan_to_num(a)))
            for kind, pp in (("primary", p), ("avg", avg)):
                per = masked_auc(y, pp)
                r[f"{name}_{kind}"] = {"auc": float(np.nanmean(list(per.values()))), "per_label": per}
            saved[f"{name}__{Path(ck).stem}__primary"], saved[f"{name}__{Path(ck).stem}__alt"] = p, a
        res[ck] = r
        print(Path(ck).name, " ".join(f"{k} {v['auc']:.4f}" for k, v in r.items()), flush=True)
        del model
    json.dump(res, open(args.out, "w"), indent=1)
    np.savez_compressed(str(Path(args.out).with_suffix("")) + "_preds.npz", **saved)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
