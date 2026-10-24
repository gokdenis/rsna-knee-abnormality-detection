"""Score every public label set against the 58 expert-labeled studies.

Usage:
    .venv/bin/python scripts/compare_labels.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knee.labels import gold_labels, load_all_sources, load_train, score_against_gold  # noqa: E402


def main() -> None:
    train = load_train(ROOT / "data/raw")
    gold = gold_labels(train)
    sources = load_all_sources(ROOT / "data/labels")

    table = pd.DataFrame({name: score_against_gold(df, gold) for name, df in sources.items()}).T
    table = table.sort_values("macro", ascending=False)
    pd.set_option("display.width", 250)
    print(f"Gold studies: {len(gold)}\n")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
