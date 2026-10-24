"""Release-224 Raptor model on gold58: plain view (as published) vs mean of plain + anatomical mirror (Kaggle GPU).

The mirror is applied to the uint8 windows before normalisation (coat96.mirror), as for the 384 model.
Writes /kaggle/working/eval_224mirror.json and _preds.npz.
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

CKPT = "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt"


def cache_dir(slug):
    hits = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{slug}/**/masks.npy", recursive=True)]
    hits += [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/{slug}/masks.npy")]
    assert hits, slug
    return hits[0]


@torch.no_grad()
def run(model, view, d, rows):
    ds = CachedStudies(d, np.asarray(rows), np.full((len(rows), len(LABELS)), np.nan), 1.0, None, False)
    model.eval()
    plain, mirr = [], []
    for x, c, _, _ in torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            plain.append(torch.sigmoid(model(coat96.to_input(x, "cuda", view=view)).float()).cpu().numpy())
            xm = coat96.mirror(x, c[0].numpy())
            mirr.append(torch.sigmoid(model(coat96.to_input(xm, "cuda", view=view)).float()).cpu().numpy())
    return np.concatenate(plain), np.concatenate(mirr)


def main():
    lab = pd.read_csv(find_file("train_labels_pl3.csv")).set_index("StudyInstanceUID")
    gold = lab.index[lab["is_gold"] == 1].astype(str).tolist()
    model, view = coat96.load_model_view(find_file(CKPT), "cuda")
    print("view", view, flush=True)
    p = np.full((len(gold), len(LABELS)), np.nan); m = p.copy()
    for slug in ("knee-build-cache-coat96-0", "knee-build-cache-coat96-1"):
        d = cache_dir(slug)
        row = {s: i for i, s in enumerate(np.load(f"{d}/ids.npy", allow_pickle=True).astype(str))}
        sel = [k for k, s in enumerate(gold) if s in row]
        if sel:
            p[sel], m[sel] = run(model, view, d, [row[gold[k]] for k in sel])
    y = lab.loc[gold, LABELS].values.astype(float)
    res = {}
    for name, pp in (("plain", p), ("plain_mirror_mean", 0.5 * (p + m)), ("mirror_only", m)):
        per = masked_auc(y, pp)
        res[name] = {"auc": float(np.nanmean(list(per.values()))), "per_label": per}
        print(f"{name:18s} {res[name]['auc']:.4f} | " + " ".join(f"{k[:6]} {v:.3f}" for k, v in per.items()), flush=True)
    json.dump(res, open("/kaggle/working/eval_224mirror.json", "w"), indent=1)
    np.savez_compressed("/kaggle/working/eval_224mirror_preds.npz", labels=np.array(LABELS), gold_ids=np.array(gold),
                        plain=p, mirror=m)


main()
