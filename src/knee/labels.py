"""Loading and scoring of report-derived label sets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import ID_COL, LABELS

# Public label sets that share the plain wide format: StudyInstanceUID + 12 finding columns.
WIDE_SOURCES = {
    "steven_v2": "stevenleehans/llm_labels_v2.csv",
    "steven_v4": "stevenleehans/llm_labels_v4_blend.csv",
    "steven_full": "stevenleehans/llm_labels_full.csv",
    "lixin": "lixin73/labels_llm_gpt56sol.csv",
    "jev_final2": "riadmohamed42/FINAL2_labels_scores.csv",
    "jev_hybrid": "riadmohamed42/HYBRID_labels_scores.csv",
    "yunus": "yunusgmsoy/yunus_llm_labels.csv",
    "dread": "dreaddevelopment/labels_llm_soft.csv",
    "consensus": "karttikjangid05/consensus_labels.csv",
    "pilkwang": "pilkwang/report_labels_v2.csv",
}

# Sources averaged into the training target. Chosen on principle (strong, independently built
# sets), not by searching blends on gold. jev_hybrid's thresholds may have seen gold, so it is
# only one vote among several.
BLEND_SOURCES = ["steven_v4", "jev_hybrid", "consensus", "yunus", "pilkwang"]


def load_train(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / "train.csv")


def gold_labels(train: pd.DataFrame) -> pd.DataFrame:
    """The expert-labeled studies (rows where every finding is present)."""
    gold = train[train[LABELS].notna().all(axis=1)]
    return gold[[ID_COL] + LABELS].set_index(ID_COL).astype(int)


def load_wide_source(labels_dir: Path, rel_path: str) -> pd.DataFrame:
    df = pd.read_csv(Path(labels_dir) / rel_path)
    df[ID_COL] = df[ID_COL].astype(str)
    return df.set_index(ID_COL)[LABELS].astype(float)


def load_all_sources(labels_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: load_wide_source(labels_dir, rel) for name, rel in WIDE_SOURCES.items()}


def silent_mask(steven_full: pd.DataFrame, col: str) -> pd.Series:
    """True where the report does not address `col` (that labeler encodes silence as exactly 0.5)."""
    return steven_full[col].eq(0.5)


def fill_silent_synovitis(
    synovitis: pd.Series, effusion: pd.Series, silent: pd.Series
) -> pd.Series:
    """Return a Synovitis target where silent reports get a value derived from Effusion.

    Reports rarely mention synovitis (~84% silent), yet ~34% of those silent studies are
    positive on gold. Effusion is reported far more often and co-occurs with synovitis
    (gold: P(syn | eff) = 0.63, P(syn | no eff) = 0.22). Explicit report statements
    (silent == False) must be kept unchanged.

    Args:
        synovitis: current soft Synovitis target in [0, 1], indexed by StudyInstanceUID.
        effusion: soft Effusion target in [0, 1], same index.
        silent: boolean mask, True where the report does not address synovitis.
    """
    p_no_eff, p_eff = 0.22, 0.63
    from_effusion = p_no_eff + (p_eff - p_no_eff) * effusion.clip(0, 1)
    return synovitis.where(~silent, from_effusion)


def score_against_gold(pred: pd.DataFrame, gold: pd.DataFrame) -> pd.Series:
    """Per-finding ROC AUC of a label set on the gold studies it covers, plus the macro mean."""
    common = gold.index.intersection(pred.index)
    scores = {}
    for col in LABELS:
        y, p = gold.loc[common, col].values, pred.loc[common, col].values
        mask = ~np.isnan(p)
        scores[col] = roc_auc_score(y[mask], p[mask]) if len(np.unique(y[mask])) == 2 else np.nan
    out = pd.Series(scores)
    out["macro"] = out.mean()
    out["n_gold"] = len(common)
    return out


def build_training_labels(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average BLEND_SOURCES, then fill Synovitis where the report is silent about it."""
    # Mean over the sources that cover each study (one source misses a single study).
    stacked = pd.concat([sources[name] for name in BLEND_SOURCES], keys=BLEND_SOURCES)
    blend = stacked.groupby(level=1).mean()
    silent = silent_mask(sources["steven_full"], "Synovitis").reindex(blend.index, fill_value=False)
    blend["Synovitis"] = fill_silent_synovitis(blend["Synovitis"], blend["Effusion"], silent)
    return blend.clip(0, 1)
