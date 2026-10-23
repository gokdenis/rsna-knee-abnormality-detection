"""Mix report-derived targets with out-of-fold model predictions (pseudo-label round).

Each study's OOF prediction comes from a model that never trained on it, and the folds are
kept unchanged, so the student's validation folds stay clean. Gold studies keep expert labels.

Usage:
    .venv/bin/python scripts/build_pseudo_labels.py --oof outputs/train-v2/oof.csv --out train_labels_pl1.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True, help="OOF csv(s); several are averaged")
    ap.add_argument("--base", default="data/processed/train_labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report_weight", type=float, default=0.5)
    args = ap.parse_args()

    base = pd.read_csv(ROOT / args.base).set_index(ID_COL)
    oof = sum(pd.read_csv(ROOT / p).set_index(ID_COL)[LABELS] for p in args.oof) / len(args.oof)
    oof = oof.reindex(base.index)
    assert oof.notna().all().all(), "OOF must cover every study"

    # Report targets before gold rows were overwritten, so the gold diagnostic is honest.
    report = build_training_labels(load_all_sources(ROOT / "data/labels")).reindex(base.index)
    mixed = args.report_weight * report + (1 - args.report_weight) * oof

    gold = gold_labels(load_train(ROOT / "data/raw"))
    print("Mixed targets vs gold:")
    print(score_against_gold(mixed, gold).round(3).to_string())

    out = base.copy()
    is_gold = out["is_gold"] == 1
    out.loc[~is_gold, LABELS] = mixed.loc[~is_gold, LABELS]
    path = ROOT / "data/processed" / args.out
    out.reset_index().to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
