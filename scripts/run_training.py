"""Run KneeMIL fold training or Raptor fine-tuning outside Kaggle (e.g. on a rented GPU).

Examples:
    python scripts/run_training.py kneemil --cache /workspace/cache-v2 --labels /workspace/labels/train_labels_pl3.csv \
        --out /workspace/runs/cns-v2-pl3 --folds 0 1 2 3 4 --config '{"layout": "v2", "arch": "convnext_small.fb_in22k_ft_in1k"}'
    python scripts/run_training.py raptorft --cache /workspace/cache-raptor --labels ... --weights raptor_ft_coatnet_v9_full.pt \
        --out /workspace/runs/raptorft --config '{"tag": "c", "lr_backbone": 3e-5}'
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kneemil", "raptorft"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--weights", default=None, help="Raptor checkpoint to start from (raptorft)")
    ap.add_argument("--config", default="{}", help="JSON overrides for the run config")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    overrides = json.loads(args.config)

    if args.mode == "kneemil":
        from knee.train import Config, train_fold
        cfg = Config(**overrides)
        for fold in args.folds:
            best = train_fold(cfg, fold, args.cache, labels, args.out, args.device)
            print(f"[fold {fold}] best val auc {best:.4f}", flush=True)
    else:
        from knee.raptor_train import RaptorFTConfig, finetune
        finetune(RaptorFTConfig(**overrides), args.weights, args.cache, labels, args.out, args.device)


if __name__ == "__main__":
    main()
