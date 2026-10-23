"""Fine-tune nartaa's coat96 OAI checkpoint on competition pseudo-labels + OAI MOAKS targets.

Example (on the pod):
    python scripts/train_coat96.py --weights /workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt \
        --comp-cache /workspace/cache-coat96-0 /workspace/cache-coat96-1 \
        --comp-labels /workspace/labels/train_labels_pl3.csv \
        --oai-cache /workspace/cache-oai-V00 /workspace/cache-oai-V03 --oai-labels data/oai/oai_labels.csv \
        --out /workspace/runs/coat96-a --config '{"tag": "a", "epochs": 4}'
    # afterwards: --swa ep1 ep2 ep3 averages those epoch checkpoints into coat96_<tag>_swa.pt
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.coat96_train import Coat96Config, finetune, predict_caches, swa  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--comp-cache", nargs="*", default=[])
    ap.add_argument("--comp-labels")
    ap.add_argument("--comp-extra", nargs="*", default=[], help="extra-view caches of competition studies (train rows only)")
    ap.add_argument("--comp-alt", help="alternative-view cache: train studies use it with p=0.5 instead of the primary view")
    ap.add_argument("--oai-cache", nargs="*", default=[])
    ap.add_argument("--oai-labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="{}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--swa", nargs="*", help="epoch names (ep1 ep2 ...) to average; skips training")
    ap.add_argument("--predict", help="write mirror-TTA predictions of --weights on --comp-cache to this csv; skips training")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = Coat96Config(**json.loads(args.config))
    if args.predict:
        predict_caches(args.weights, args.comp_cache, args.device, cfg.workers).to_csv(args.predict, index=False)
        print("wrote", args.predict)
        return
    if args.swa:
        out = Path(args.out) / f"coat96_{cfg.tag}_swa.pt"
        swa([Path(args.out) / f"coat96_{cfg.tag}_{e}.pt" for e in args.swa], out)
        print("wrote", out)
        return
    comp = pd.read_csv(args.comp_labels)
    oai = pd.read_csv(args.oai_labels, dtype={"ID": str}) if args.oai_labels else None
    finetune(cfg, args.weights, args.comp_cache, comp, args.oai_cache, oai, args.out, args.device, comp_extra=args.comp_extra, comp_alt=args.comp_alt)


if __name__ == "__main__":
    main()
