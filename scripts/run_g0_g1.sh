#!/usr/bin/env bash
# Phase B controlled comparison on the pod: release (start) vs G1 (+ corrected OAI) vs G0 (competition only).
# Same start checkpoint, seed, batch, epochs and optimizer steps; the same fixed OAI holdout for both.
# Waits for the OAI pipeline, runs G1 first (its epoch size sets G0's), then SWA of epochs 2-3 for both.
set -u
cd /workspace/new_knee
L=/workspace/logs; R=/workspace/runs; mkdir -p $R
W=/workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt
until grep -q PIPELINE_DONE $L/pipeline_tar.log 2>/dev/null; do sleep 60; done
OAI=$(ls -d /workspace/cache-oai-* | tr '\n' ' ')
COMMON='"epochs": 4, "batch_size": 4, "k_train": 16, "workers": 8, "seed": 11, "holdout_seed": 2026, "time_budget_min": 2000'
{
  echo "start checkpoint sha256: $(sha256sum $W | cut -c1-64)"
  echo "oai labels md5: $(md5sum data/oai/oai_labels.csv | cut -c1-32)"
  echo "competition targets md5: $(md5sum /workspace/labels/coat96_targets_distill.csv | cut -c1-32)"
  echo "oai caches: $OAI"
} > $R/g_manifest.txt
run() {  # tag config
  ulimit -n 65536
  python3 scripts/train_coat96.py --weights $W --comp-cache /workspace/cache-coat96-0 /workspace/cache-coat96-1 \
    --comp-labels /workspace/labels/coat96_targets_distill.csv --oai-cache $OAI --oai-labels data/oai/oai_labels.csv \
    --out $R/$1 --config "{\"tag\": \"$1\", $COMMON, $2}" > $L/train_$1.log 2>&1
  python3 scripts/train_coat96.py --out $R/$1 --swa ep2 ep3 --config "{\"tag\": \"$1\"}" >> $L/train_$1.log 2>&1
}
run g1 '"oai_weight": 0.5'
N=$(grep -oE "train: competition [0-9]+, OAI [0-9]+" $L/train_g1.log | head -1 | grep -oE "[0-9]+" | awk "{s+=\$1} END {print s}")
echo "G1 studies per epoch: $N" >> $R/g_manifest.txt
run g0 "\"oai_fraction\": 0.0, \"epoch_size\": $N"
echo G_DONE >> $R/g_manifest.txt
