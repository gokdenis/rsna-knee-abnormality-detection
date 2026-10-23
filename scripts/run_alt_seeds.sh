#!/usr/bin/env bash
# G1 recipe with view swap: a train study that has an alternative-series view uses it with p = 0.5 instead of its
# primary view (same rows, steps and seeds as G1-G3, so the soups pair up), then the soup of the three and its evaluation against the G1-G3 soup (primary and second view).
# Usage (on the pod): bash scripts/run_alt_seeds.sh a1:11 a2:23 a3:37
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; M=$R/as_manifest.txt
W=/workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
ALT=/workspace/cache-coat96alt
OAI=$(ls -d /workspace/cache-oai-* | tr '\n' ' ')
COMP="/workspace/cache-coat96-0 /workspace/cache-coat96-1"
T=/workspace/labels/coat96_targets_distill.csv
BASE='"epochs": 4, "batch_size": 4, "k_train": 16, "workers": 8, "holdout_seed": 2026, "time_budget_min": 2000, "oai_weight": 0.5'
until grep -q CURL_DONE $L/altcache_curl.log 2>/dev/null && grep -q PIP_DONE $L/pip.log 2>/dev/null; do sleep 30; done
python3 -c "import numpy as np, os; o = np.load('$ALT/offsets.npy'); print('blobs ok' if o.max() == os.path.getsize('$ALT/blobs.bin') else 'MISMATCH')" > $L/altcache_dl.log 2>&1
if ! grep -q "blobs ok" $L/altcache_dl.log || ! python3 -c "import torch, timm, pandas, sklearn, cv2; assert torch.__version__.startswith('2.9.1')"; then
  echo "alt cache or python env broken" > $M; echo AS_DONE >> $M; exit 1
fi
{
  echo "start checkpoint sha256: $(sha256sum $W | cut -c1-64)"
  echo "oai labels md5: $(md5sum data/oai/oai_labels.csv | cut -c1-32)"
  echo "competition targets md5: $(md5sum $T | cut -c1-32)"
  echo "alt cache studies: $(python3 -c "import numpy as np; print(len(np.load('$ALT/ids.npy', allow_pickle=True)))")"
  echo "config: $BASE; runs: $*"
} > $M
CK=""
for ts in "$@"; do
  t=${ts%:*}; ulimit -n 65536
  python3 scripts/train_coat96.py --weights $W --comp-cache $COMP --comp-alt $ALT --comp-labels $T \
    --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
    --out $R/$t --config "{\"tag\": \"$t\", $BASE, \"seed\": ${ts#*:}}" > $L/train_$t.log 2>&1
  python3 scripts/train_coat96.py --out $R/$t --swa ep2 ep3 --config "{\"tag\": \"$t\"}" >> $L/train_$t.log 2>&1
  if [ -f $R/$t/coat96_${t}_swa.pt ]; then echo "$t done" >> $M; CK="$CK $R/$t/coat96_${t}_swa.pt"; else echo "$t FAILED" >> $M; break; fi
done
mkdir -p $R/asoup
python3 scripts/eval_coat96.py --soup $R/asoup/coat96_asoup_swa.pt --soup-from $CK --ckpts $R/soup/coat96_g123_soup.pt $CK \
  --comp-cache $COMP --comp-labels $T --oai-cache $OAI --oai-labels data/oai/oai_labels.csv --workers 4 \
  --out $R/eval_asoup.json > $L/eval_asoup.log 2>&1
python3 scripts/eval_alt.py --ckpts $R/soup/coat96_g123_soup.pt $R/asoup/coat96_asoup_swa.pt --prim $COMP --alt $ALT \
  --labels /workspace/labels/train_labels_pl3.csv --out $R/eval_alt2.json > $L/eval_alt2.log 2>&1
echo "soup of:$CK" >> $M
grep -hE "^coat96" $L/eval_asoup.log $L/eval_alt2.log >> $M
echo AS_DONE >> $M
