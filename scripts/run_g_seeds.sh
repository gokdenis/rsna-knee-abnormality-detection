#!/usr/bin/env bash
# More seeds of the G1 recipe (same start, targets, OAI labels, holdout and config; only the seed differs),
# then optionally the same recipe on the 224 release checkpoint (view c08r224) when W224 exists.
# Usage (on the pod): bash scripts/run_g_seeds.sh g2:23 g3:37
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; mkdir -p $R
W=/workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
W224=${W224:-}
OAI=$(ls -d /workspace/cache-oai-* | tr '\n' ' ')
COMP="/workspace/cache-coat96-0 /workspace/cache-coat96-1"
BASE='"epochs": 4, "batch_size": 4, "k_train": 16, "workers": 8, "holdout_seed": 2026, "time_budget_min": 2000, "oai_weight": 0.5'
{
  echo "start checkpoint sha256: $(sha256sum $W | cut -c1-64)"
  echo "oai labels md5: $(md5sum data/oai/oai_labels.csv | cut -c1-32)"
  echo "competition targets md5: $(md5sum /workspace/labels/coat96_targets_distill.csv | cut -c1-32)"
  echo "oai caches: $OAI"
  echo "runs: $*"
} > $R/gs_manifest.txt
run() {  # tag weights targets config
  ulimit -n 65536
  python3 scripts/train_coat96.py --weights $2 --comp-cache $COMP --comp-labels $3 \
    --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
    --out $R/$1 --config "{\"tag\": \"$1\", $BASE, $4}" > $L/train_$1.log 2>&1
  python3 scripts/train_coat96.py --out $R/$1 --swa ep2 ep3 --config "{\"tag\": \"$1\"}" >> $L/train_$1.log 2>&1
  if [ ! -f $R/$1/coat96_$1_swa.pt ]; then echo "$1 FAILED (see $L/train_$1.log)" >> $R/gs_manifest.txt; echo GS_DONE >> $R/gs_manifest.txt; exit 1; fi
  echo "$1 done: $R/$1/coat96_$1_swa.pt" >> $R/gs_manifest.txt
}
for ts in "$@"; do
  run ${ts%:*} $W /workspace/labels/coat96_targets_distill.csv "\"seed\": ${ts#*:}"
done
if [ -n "$W224" ] && [ -f "$W224" ]; then
  # self-distillation targets of the 224 release checkpoint (its own view), then the G1 recipe
  python3 scripts/train_coat96.py --weights $W224 --comp-cache $COMP --out $R/h1 \
    --predict $R/release224_train_preds.csv --config '{"workers": 8}' > $L/predict_224.log 2>&1
  python3 -I scripts/make_coat96_targets.py --labels /workspace/labels/train_labels_pl3.csv \
    --preds $R/release224_train_preds.csv --w-preds 1.0 \
    --out /workspace/labels/coat96_targets_distill224.csv >> $L/predict_224.log 2>&1
  run h1 $W224 /workspace/labels/coat96_targets_distill224.csv '"seed": 11'
fi
echo GS_DONE >> $R/gs_manifest.txt
