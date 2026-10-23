"""Does a second view from alternative series of the same type help our soup? (Kaggle GPU, no training.)

For every train study with an alternative (cache knee-build-cache-coat96alt) the soup predicts the primary view
(caches knee-build-cache-coat96-0/1) and the alternative view; the two probabilities are averaged. Scored on gold58
(studies without an alternative keep the primary prediction) and on the weak labels (pl3, binarised at 0.5) of the
alternative studies. Writes /kaggle/working/eval_alt.json and eval_alt_preds.npz.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, code_dir())
from knee import coat96  # noqa: E402
from knee.coat96_train import CachedStudies, masked_auc  # noqa: E402
from knee.constants import LABELS  # noqa: E402

CKPT = globals().get("CKPT", "coat96_g123soup_swa.pt")


def cache_dir(slug):
    hits = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{slug}/**/masks.npy", recursive=True)]
    hits += [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/{slug}/masks.npy")]
    assert hits, slug
    return hits[0]


def ids_of(d):
    return np.load(f"{d}/ids.npy", allow_pickle=True).astype(str)


@torch.no_grad()
def run(model, view, d, rows):
    """coat96_train.predict with fp16 autocast (native on T4; bf16 is emulated there)."""
    ds = CachedStudies(d, np.asarray(rows), np.full((len(rows), len(LABELS)), np.nan), 1.0, None, False)
    model.eval()
    out = []
    for x, c, _, _ in torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4):
        p = 0
        flips = (False, True) if view.mirror_tta else (False,)
        for flip in flips:
            xx = coat96.mirror(x, c[0].numpy()) if flip else x
            with torch.autocast("cuda", dtype=torch.float16):
                p = p + torch.sigmoid(model(coat96.to_input(xx, "cuda", view=view)).float()) / len(flips)
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    prim = [cache_dir("knee-build-cache-coat96-0"), cache_dir("knee-build-cache-coat96-1")]
    alt = cache_dir("knee-build-cache-coat96alt")
    lab = pd.read_csv(find_file("train_labels_pl3.csv")).set_index("StudyInstanceUID")
    model, view = coat96.load_model_view(find_file(CKPT), "cuda")
    alt_ids = ids_of(alt)
    print("alternative studies", len(alt_ids), flush=True)
    p_alt = run(model, view, alt, np.arange(len(alt_ids)))
    p_prim = np.zeros_like(p_alt)
    for d in prim:
        row = {s: i for i, s in enumerate(ids_of(d))}
        sel = [k for k, s in enumerate(alt_ids) if s in row]
        if sel:
            p_prim[sel] = run(model, view, d, [row[alt_ids[k]] for k in sel])
    gold_ids = lab.index[lab["is_gold"] == 1].astype(str).tolist()
    g_prim = np.zeros((len(gold_ids), len(LABELS)))
    for d in prim:
        row = {s: i for i, s in enumerate(ids_of(d))}
        sel = [k for k, s in enumerate(gold_ids) if s in row]
        if sel:
            g_prim[sel] = run(model, view, d, [row[gold_ids[k]] for k in sel])
    pos = {s: k for k, s in enumerate(alt_ids)}
    g_avg = g_prim.copy()
    for k, s in enumerate(gold_ids):
        if s in pos:
            g_avg[k] = 0.5 * (g_prim[k] + p_alt[pos[s]])
    y_gold = lab.loc[gold_ids, LABELS].values.astype(float)
    weak = [s for s in alt_ids if lab.loc[s, "is_gold"] == 0]
    wk = [pos[s] for s in weak]
    y_weak = (lab.loc[weak, LABELS].values >= 0.5).astype(float)
    res = {}
    for name, (y, p) in {"gold58_primary": (y_gold, g_prim), "gold58_avg": (y_gold, g_avg),
                         "weak_alt_primary": (y_weak, p_prim[wk]), "weak_alt_avg": (y_weak, 0.5 * (p_prim[wk] + p_alt[wk])),
                         "weak_alt_alt_only": (y_weak, p_alt[wk])}.items():
        per = masked_auc(y, p)
        res[name] = {"auc": float(np.nanmean(list(per.values()))), "per_label": per, "n": int(len(y))}
        print(f"{name:18s} n={len(y):5d} auc {res[name]['auc']:.4f} | " +
              " ".join(f"{k[:6]} {v:.3f}" for k, v in per.items()), flush=True)
    json.dump(res, open("/kaggle/working/eval_alt.json", "w"), indent=1)
    np.savez_compressed("/kaggle/working/eval_alt_preds.npz", labels=np.array(LABELS), alt_ids=alt_ids, p_alt=p_alt,
                        p_prim=p_prim, gold_ids=np.array(gold_ids), g_prim=g_prim, g_avg=g_avg)


main()
