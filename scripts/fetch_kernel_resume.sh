#!/usr/bin/env bash
# Resumable download of all output files of a Kaggle kernel: fetch_kernel_resume.sh <owner/kernel> <dir>
set -u
mkdir -p "$2"
python3 /workspace/new_knee/scripts/kaggle_output_urls.py "$1" | while IFS=$'\t' read -r name url; do
  for i in $(seq 1 30); do
    curl -sS -L -C - --retry 5 --retry-delay 5 -o "$2/$name" "$url" && break
    echo "curl retry $i for $name"; sleep 5
  done
done
python3 -c "
import numpy as np, os, sys
o = np.load(sys.argv[1] + '/offsets.npy'); s = os.path.getsize(sys.argv[1] + '/blobs.bin')
print(sys.argv[1], 'blobs ok' if o.max() == s else f'SIZE MISMATCH {o.max()} vs {s}')" "$2"
