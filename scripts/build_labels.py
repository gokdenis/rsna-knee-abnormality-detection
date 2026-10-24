"""Build the training target table with folds.

Output: data/processed/train_labels.csv with columns
    StudyInstanceUID, <12 soft labels>, is_gold, fold

Gold studies keep their expert labels (hard 0/1) and are flagged so training can either
exclude them or use them as an extra holdout.

Usage:
    .venv/bin/python scripts/build_labels.py
"""

import sys
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knee.constants import ID_COL, LABELS  # noqa: E402
from knee.labels import (  # noqa: E402
    build_training_labels,
    gold_labels,
    load_all_sources,
    load_train,
    score_against_gold,
)

N_FOLDS = 5
SEED = 42


def main() -> None:
    train = load_train(ROOT / "data/raw")
    gold = gold_labels(train)
    sources = load_all_sources(ROOT / "data/labels")

    labels = build_training_labels(sources)
    print("Blend vs gold:")
    print(score_against_gold(labels, gold).round(3).to_string(), "\n")

    labels.loc[gold.index, LABELS] = gold[LABELS].astype(float)
    labels["is_gold"] = labels.index.isin(gold.index).astype(int)
    labels = labels.reset_index()

    # Identical reports go to the same fold; stratify on effusion, the most common finding.
    reports = train.set_index(ID_COL).loc[labels[ID_COL], "Report"].values
    strat = (labels["Effusion"] > 0.5).astype(int).values
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    labels["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(labels, strat, groups=reports)):
        labels.loc[val_idx, "fold"] = fold

    assert len(labels) == len(train) and (labels["fold"] >= 0).all()
    out = ROOT / "data/processed/train_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)
    print(f"Wrote {out} ({len(labels)} rows)")
    print(labels.groupby("fold").agg(n=(ID_COL, "size"), gold=("is_gold", "sum")).T.to_string())
    print("\nMean target per finding:")
    print(labels[LABELS].mean().round(3).to_string())


if __name__ == "__main__":
    main()
