#!/usr/bin/env bash
# Aggressive variant of the G1 recipe: full OAI weight and longer training (same start, targets, OAI labels and
# holdout), SWA of the last two epochs, then the evaluator against G1 and the G1-G3 soup.
# Usage (on the pod): bash scripts/run_g_aggr.sh g4:41
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; M=$R/ga_manifest.txt; mkdir -p $R
W=/workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
OAI=$(ls -d /workspace/cache-oai-* | tr '\n' ' ')
COMP="/workspace/cache-coat96-0 /workspace/cache-coat96-1"
EPOCHS=${EPOCHS:-6}
BASE="\"epochs\": $EPOCHS, \"batch_size\": 4, \"k_train\": 16, \"workers\": 8, \"holdout_seed\": 2026, \"time_budget_min\": 2000, \"oai_weight\": 1.0"
python3 -c "import torch, timm, pandas, sklearn, cv2, pydicom; assert torch.__version__.startswith('2.9.1')" \
  || { echo "python env broken" >> $M; echo GA_DONE >> $M; exit 1; }
{
  echo "start checkpoint sha256: $(sha256sum $W | cut -c1-64)"
  echo "oai labels md5: $(md5sum data/oai/oai_labels.csv | cut -c1-32)"
  echo "competition targets md5: $(md5sum /workspace/labels/coat96_targets_distill.csv | cut -c1-32)"
  echo "config: $BASE; runs: $*"
} > $M
for ts in "$@"; do
  t=${ts%:*}; ulimit -n 65536
  python3 scripts/train_coat96.py --weights $W --comp-cache $COMP --comp-labels /workspace/labels/coat96_targets_distill.csv \
    --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
    --out $R/$t --config "{\"tag\": \"$t\", $BASE, \"seed\": ${ts#*:}}" > $L/train_$t.log 2>&1
  python3 scripts/train_coat96.py --out $R/$t --swa ep$((EPOCHS-2)) ep$((EPOCHS-1)) --config "{\"tag\": \"$t\"}" >> $L/train_$t.log 2>&1
  [ -f $R/$t/coat96_${t}_swa.pt ] && echo "$t done: $R/$t/coat96_${t}_swa.pt" >> $M || { echo "$t FAILED" >> $M; break; }
done
CK=$(for ts in "$@"; do f=$R/${ts%:*}/coat96_${ts%:*}_swa.pt; [ -f $f ] && echo $f; done)
python3 scripts/eval_coat96.py --ckpts $R/g1/coat96_g1_swa.pt $R/soup/coat96_g123_soup.pt $CK \
  --comp-cache $COMP --comp-labels /workspace/labels/coat96_targets_distill.csv \
  --oai-cache $OAI --oai-labels data/oai/oai_labels.csv --workers 4 --out $R/eval_ga.json > $L/eval_ga.log 2>&1
grep -E "^coat96|mean-prob" $L/eval_ga.log >> $M
echo GA_DONE >> $M
