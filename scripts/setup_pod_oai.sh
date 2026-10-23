#!/usr/bin/env bash
# Rebuild everything phase B needs on a fresh pod (after the owner stored the Kaggle token and NDA password):
# competition coat96 caches + labels + release weights from Kaggle, then the OAI package listings, the
# series selection (split into parts for a small disk) and the OAI download/cache pipeline.
set -u
cd /workspace
L=/workspace/logs; mkdir -p $L /workspace/oai
S=/workspace/new_knee/scripts
# 1. Kaggle inputs (resumable downloads)
bash $S/fetch_kernel_resume.sh gkdnzz31/knee-build-cache-coat96-0 /workspace/cache-coat96-0 > $L/fetch0.log 2>&1 &
bash $S/fetch_kernel_resume.sh gkdnzz31/knee-build-cache-coat96-1 /workspace/cache-coat96-1 > $L/fetch1.log 2>&1 &
[ -d /workspace/labels ] || kaggle datasets download -d gkdnzz31/knee-labels -p /workspace/labels --unzip -q
[ -f /workspace/weights/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt ] || \
  kaggle datasets download -d nartaa/rsna-knee-publication-swa-weights-20261007 -p /workspace/weights --unzip -q
# 2. OAI package listings and series selection
for vp in V00:<V00_PACKAGE_ID> V01:<V01_PACKAGE_ID> V03:<V03_PACKAGE_ID> V06:<V06_PACKAGE_ID>; do
  v=${vp%:*}; p=${vp#*:}; mkdir -p /workspace/oai/$v
  M=/root/NDA/nda-tools/downloadcmd/packages/$p/package_file_metadata_$p.txt
  [ -f $M ] || downloadcmd -dp $p -u $NDA_USER --verify -d /workspace/oai/$v/raw < /dev/null > $L/verify_$v.log 2>&1
  [ -f /workspace/oai/$v/image03.txt ] || downloadcmd -dp $p -u $NDA_USER -d /workspace/oai/$v \
      --file-regex "image03\.txt" < /dev/null > $L/meta_$v.log 2>&1
  parts=2; [ $v = V06 ] && parts=1
  python3 -I $S/oai_select.py --image03 /workspace/oai/$v/image03.txt --sizes $M --labels /workspace/new_knee/data/oai/oai_labels.csv \
    --visit $v --out /workspace/oai/$v/links.txt --parts $parts | head -1
done
wait
for d in /workspace/cache-coat96-0 /workspace/cache-coat96-1; do tail -1 $L/fetch${d: -1}.log; done
# 3. OAI download -> unpack -> cache, part by part (raw images deleted after each part)
bash $S/oai_pipeline.sh V00p0:<V00_PACKAGE_ID> V00p1:<V00_PACKAGE_ID> V03p0:<V03_PACKAGE_ID> V03p1:<V03_PACKAGE_ID> V06:<V06_PACKAGE_ID> V01p0:<V01_PACKAGE_ID> V01p1:<V01_PACKAGE_ID>
echo SETUP_DONE
