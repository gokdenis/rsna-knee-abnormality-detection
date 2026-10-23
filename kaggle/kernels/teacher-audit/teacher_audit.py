"""Release-384 predictions (= the G recipe's competition targets, w_preds 1.0) for the 18 studies of the Oct 20 label
audit, so the draft report labels can be compared with what the model was actually trained on. Mirror TTA as in
coat96_train.predict_caches. Writes /kaggle/working/teacher_audit.csv."""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, code_dir())
from knee import coat96  # noqa: E402
from knee.coat96_train import CachedStudies, predict  # noqa: E402
from knee.constants import LABELS  # noqa: E402

IDS = "1.2.826.0.1.3680043.8.498.20733151481921261415295204209910573639,1.2.826.0.1.3680043.8.498.28873180831637763124652074775631752918,1.2.826.0.1.3680043.8.498.37413349933694832641902637847629969234,1.2.826.0.1.3680043.8.498.12348960754791091583201828167607825033,1.2.826.0.1.3680043.8.498.13393268272072264469105890769640363680,1.2.826.0.1.3680043.8.498.10829579914809383212986717269028170263,1.2.826.0.1.3680043.8.498.75378323910171368568360074903297446681,1.2.826.0.1.3680043.8.498.13695989890112408005495264068064110179,1.2.826.0.1.3680043.8.498.11159546221032123495135103648570690656,1.2.826.0.1.3680043.8.498.10274956886032850418742861294277331382,1.2.826.0.1.3680043.8.498.69260280362382086729793018979234418276,1.2.826.0.1.3680043.8.498.20933151653103548000061448393771614270,1.2.826.0.1.3680043.8.498.83569065014916050100714967945484998462,1.2.826.0.1.3680043.8.498.11716605112302071535035323666308656381,1.2.826.0.1.3680043.8.498.11441489276425334980205891153179893797,1.2.826.0.1.3680043.8.498.82272793832748838882752586678639564181,1.2.826.0.1.3680043.8.498.85108719393967663738580518985514195538,1.2.826.0.1.3680043.8.498.35923753382244056317223397206343376734".split(",")
model, view = coat96.load_model_view(find_file("raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"), "cuda")
rows = []
for slug in ("knee-build-cache-coat96-0", "knee-build-cache-coat96-1"):
    d = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{slug}/**/masks.npy", recursive=True)
         + glob.glob(f"/kaggle/input/{slug}/masks.npy")][0]
    ids = np.load(f"{d}/ids.npy", allow_pickle=True).astype(str)
    sel = [i for i, s in enumerate(ids) if s in IDS]
    if not sel:
        continue
    ds = CachedStudies(d, np.array(sel), np.full((len(sel), len(LABELS)), np.nan), 1.0, None, False)
    _, p = predict(model, [ds], "cuda", 4, view)
    rows.append(pd.DataFrame(p, columns=LABELS).assign(StudyInstanceUID=ids[sel]))
out = pd.concat(rows)[["StudyInstanceUID"] + LABELS]
out.to_csv("/kaggle/working/teacher_audit.csv", index=False)
print("studies", len(out), "of", len(IDS))
