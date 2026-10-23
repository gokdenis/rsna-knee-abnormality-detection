#!/usr/bin/env bash
# Phase B pod setup (OAI + MOAKS fine-tune of the coat96 checkpoint). Run once on the pod after the repo
# was copied to /workspace/new_knee (rsync from the laptop, including data/oai/oai_labels.csv).
# Needs the Kaggle token on the pod (~/.kaggle/access_token or KAGGLE_API_TOKEN), entered by the owner.
# NDA credentials are entered by the owner too:  python -c "import keyring, getpass; \
#   keyring.set_password('nda-tools', input('NDA user: '), getpass.getpass())"
set -euo pipefail
cd /workspace
pip install -q --break-system-packages kaggle timm pydicom opencv-python-headless pandas scikit-learn pyarrow nda-tools keyrings.alt
K="kaggle"
dl_kernel() { [ -f "/workspace/$2/masks.npy" ] || $K kernels output "gkdnzz31/$1" -p "/workspace/$2" -q; }
dl_kernel knee-build-cache-coat96-0 cache-coat96-0
dl_kernel knee-build-cache-coat96-1 cache-coat96-1
[ -d /workspace/labels ] || $K datasets download -d gkdnzz31/knee-labels -p /workspace/labels --unzip -q
W=/workspace/weights
[ -f $W/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt ] || \
  $K datasets download -d nartaa/rsna-knee-publication-swa-weights-20261007 -p $W --unzip -q
sha256sum $W/*.pt
du -sh /workspace/cache-* /workspace/labels $W
df -h /workspace
nvidia-smi --query-gpu=name,memory.total --format=csv
