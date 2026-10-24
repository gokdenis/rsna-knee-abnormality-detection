#!/usr/bin/env bash
# One-time setup on a rented GPU pod. Expects the repo code in /workspace/new_knee and
# KAGGLE_API_TOKEN in the environment. Downloads caches, labels and Raptor weights to /workspace.
set -euo pipefail
cd /workspace
pip install -q kaggle timm pydicom opencv-python-headless pandas scikit-learn pyarrow
K="kaggle"
dl_kernel() { [ -f "/workspace/$2/masks.npy" ] || $K kernels output "gkdnzz31/$1" -p "/workspace/$2" -q; }
dl_kernel knee-build-cache-v2 cache-v2
dl_kernel knee-build-cache-raptor cache-raptor
[ -d /workspace/labels ] || $K datasets download -d gkdnzz31/knee-labels -p /workspace/labels --unzip -q
[ -f /workspace/weights/raptor_ft_coatnet_v9_full.pt ] || \
  $K datasets download -d dreaddevelopment/raptor-knee-finespacing -p /workspace/weights --unzip -q
du -sh /workspace/cache-* /workspace/labels /workspace/weights
nvidia-smi --query-gpu=name,memory.total --format=csv
