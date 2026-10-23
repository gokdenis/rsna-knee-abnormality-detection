"""Competition targets for the coat96 fine-tune: our pseudo-labels mixed with the release model's predictions.

Mixing in the starting checkpoint's own predictions (self-distillation) keeps what it already knows
while the OAI MOAKS targets add new information. Gold rows keep their expert labels (validation only).

Usage:
    python -I scripts/make_coat96_targets.py --labels /workspace/labels/train_labels_pl3.csv \
        --preds /workspace/runs/release384_train_preds.csv --w-preds 0.5 --out /workspace/labels/coat96_targets.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.constants import LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--w-preds", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels)
    pred = pd.read_csv(args.preds).rename(columns={"id": "StudyInstanceUID"}).set_index("StudyInstanceUID")
    pred = pred.loc[lab["StudyInstanceUID"], LABELS].values
    gold = lab["is_gold"].values == 1
    aucs = [roc_auc_score(lab.loc[gold, c].round(), pred[gold, j]) for j, c in enumerate(LABELS)]
    print(f"release model on gold58: {np.mean(aucs):.4f} (in-sample if it was trained on gold)")
    corr = np.mean([np.corrcoef(lab.loc[~gold, c], pred[~gold, j])[0, 1] for j, c in enumerate(LABELS)])
    print(f"mean correlation with our labels on train: {corr:.3f}")
    out = lab.copy()
    out.loc[~gold, LABELS] = (1 - args.w_preds) * lab.loc[~gold, LABELS].values + args.w_preds * pred[~gold]
    out.to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
