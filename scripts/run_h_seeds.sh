#!/usr/bin/env bash
# More seeds of the H1 recipe (G1 recipe on nartaa's 224 checkpoint, self-distilled 224 targets), then the weight
# soup of H1 + the new seeds, scored with the evaluator. Usage (on the pod): bash scripts/run_h_seeds.sh h2:23 h3:37
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; M=$R/hs_manifest.txt; mkdir -p $R
W224=/workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt
T224=/workspace/labels/coat96_targets_distill224.csv
OAI=$(ls -d /workspace/cache-oai-* | tr '\n' ' ')
COMP="/workspace/cache-coat96-0 /workspace/cache-coat96-1"
BASE='"epochs": 4, "batch_size": 4, "k_train": 16, "workers": 8, "holdout_seed": 2026, "time_budget_min": 2000, "oai_weight": 0.5'
python3 -c "import torch, timm, pandas, sklearn, cv2, pydicom; assert torch.__version__.startswith('2.9.1')" \
  || { echo "python env broken" >> $M; echo HS_DONE >> $M; exit 1; }
{
  echo "start checkpoint sha256: $(sha256sum $W224 | cut -c1-64)"
  echo "oai labels md5: $(md5sum data/oai/oai_labels.csv | cut -c1-32)"
  echo "competition targets md5: $(md5sum $T224 | cut -c1-32)"
  echo "config: $BASE; runs: $*"
} > $M
for ts in "$@"; do
  t=${ts%:*}; ulimit -n 65536
  python3 scripts/train_coat96.py --weights $W224 --comp-cache $COMP --comp-labels $T224 \
    --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
    --out $R/$t --config "{\"tag\": \"$t\", $BASE, \"seed\": ${ts#*:}}" > $L/train_$t.log 2>&1
  python3 scripts/train_coat96.py --out $R/$t --swa ep2 ep3 --config "{\"tag\": \"$t\"}" >> $L/train_$t.log 2>&1
  [ -f $R/$t/coat96_${t}_swa.pt ] && echo "$t done: $R/$t/coat96_${t}_swa.pt" >> $M || { echo "$t FAILED" >> $M; break; }
done
CK="$R/h1/coat96_h1_swa.pt $(for ts in "$@"; do f=$R/${ts%:*}/coat96_${ts%:*}_swa.pt; [ -f $f ] && echo $f; done)"
mkdir -p $R/hsoup
python3 scripts/eval_coat96.py --soup $R/hsoup/coat96_hsoup_swa.pt --soup-from $CK --ckpts $W224 $CK \
  --comp-cache $COMP --comp-labels $T224 --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
  --workers 4 --out $R/eval_hs.json > $L/eval_hs.log 2>&1
grep -E "^(coat96|raptor)|mean-prob" $L/eval_hs.log >> $M
echo HS_DONE >> $M
