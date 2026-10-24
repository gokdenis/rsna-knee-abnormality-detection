#!/usr/bin/env bash
# Two more seeds of the G1 recipe (run_g_seeds.sh), then the weight soup of G1-G6 scored against the G1-G3 soup.
# Usage (on the pod): bash scripts/run_soup5.sh g5:53 g6:67
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; M=$R/g56_manifest.txt
python3 -c "import torch, timm, pandas, sklearn, cv2, pydicom; assert torch.__version__.startswith('2.9.1')" \
  || { echo "python env broken" > $M; echo G56_DONE >> $M; exit 1; }
bash scripts/run_g_seeds.sh "$@"
cp $R/gs_manifest.txt $M
CK=$(for t in g1 g2 g3 $(for ts in "$@"; do echo ${ts%:*}; done); do f=$R/$t/coat96_${t}_swa.pt; [ -f $f ] && echo $f; done)
mkdir -p $R/soup5
python3 scripts/eval_coat96.py --soup $R/soup5/coat96_soup5_swa.pt --soup-from $CK --ckpts $R/soup/coat96_g123_soup.pt \
  --comp-cache /workspace/cache-coat96-0 /workspace/cache-coat96-1 --comp-labels /workspace/labels/coat96_targets_distill.csv \
  --oai-cache /workspace/cache-oai-* --oai-labels data/oai/oai_labels.csv --workers 4 --out $R/eval_soup5.json > $L/eval_soup5.log 2>&1
echo "soup of: $CK" >> $M
grep -E "^coat96" $L/eval_soup5.log >> $M
echo G56_DONE >> $M
