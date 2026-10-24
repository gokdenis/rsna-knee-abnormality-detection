#!/usr/bin/env bash
# OAI visits one after another on the pod: download the selected series (links.txt from oai_select.py)
# while unpacking them, build the coat96 cache, then delete the raw images to free the disk.
# Usage: bash scripts/oai_pipeline.sh V00:<V00_PACKAGE_ID> V03:<V03_PACKAGE_ID> V06:<V06_PACKAGE_ID> V01:<V01_PACKAGE_ID>
# A visit given as V01p0:<V01_PACKAGE_ID> processes links_p0.txt of V01 into cache-oai-V01p0 (oai_select.py --parts),
# so a visit can be handled in parts when the disk is small. A failed download (e.g. HTTP 502) is retried
# with the links that are not unpacked yet.
set -u
cd /workspace/new_knee
for vp in "$@"; do
  tag=${vp%:*}; p=${vp#*:}; v=${tag%%p*}; part=${tag#$v}; L=/workspace/logs
  D=/workspace/oai/$tag; LINKS=/workspace/oai/$v/links${part:+_$part}.txt; mkdir -p $D
  if [ -f /workspace/cache-oai-$tag/stats.json ]; then echo "$tag cache exists, skipping"; continue; fi
  echo "[$(date -u +%H:%M)] $tag: download + unpack"
  if ! grep -q "Finished processing all download requests" $L/dl_$tag.log 2>/dev/null; then
    pgrep -f "[d]ownloadcmd -dp $p .*-t" >/dev/null || \
      downloadcmd -dp $p -u $NDA_USER -d $D/raw -wt 32 -t $LINKS < /dev/null > $L/dl_$tag.log 2>&1 &
  fi
  pgrep -f "[o]ai_extract.py --raw $D/raw" >/dev/null || \
    python3 -I scripts/oai_extract.py --raw $D/raw --out $D/img --follow $L/dl_$tag.log --workers 6 > $L/extract_$tag.log 2>&1 &
  # wait for the download and the unpacking of this visit
  while pgrep -f "[d]ownloadcmd -dp $p .*-t" >/dev/null || pgrep -f "[o]ai_extract.py --raw $D/raw" >/dev/null; do
    if ! pgrep -f "[d]ownloadcmd -dp $p .*-t" >/dev/null && ! grep -q "Finished processing all download requests" $L/dl_$tag.log; then
      # downloadcmd died (e.g. HTTP 502): retry with the series that are not unpacked yet
      python3 -c "
import os, sys
done = {d for d in os.listdir(sys.argv[1]) if not d.endswith('.tmp')} if os.path.isdir(sys.argv[1]) else set()
links = [l.strip() for l in open(sys.argv[2]) if l.strip()]
open(sys.argv[3], 'w').write('\\n'.join(l for l in links if os.path.basename(l)[:-7] not in done) + '\\n')" $D/img $LINKS $D/links_rest.txt
      find $D/raw -name "*.partial" -delete 2>/dev/null
      mv $L/dl_$tag.log $L/dl_$tag.$(date +%s).log
      downloadcmd -dp $p -u $NDA_USER -d $D/raw -wt 32 -t $D/links_rest.txt < /dev/null > $L/dl_$tag.log 2>&1 &
      echo "[$(date -u +%H:%M)] $tag: download restarted"
    fi
    sleep 60
  done
  echo "[$(date -u +%H:%M)] $tag: build cache"
  python3 -I scripts/build_oai_cache.py --root $D/img --visit $v --labels data/oai/oai_labels.csv \
    --out /workspace/cache-oai-$tag --workers 16 > $L/cache_$tag.log 2>&1
  tail -12 $L/cache_$tag.log
  if [ -f /workspace/cache-oai-$tag/stats.json ]; then rm -rf $D/img $D/raw; fi
  df -h /workspace | tail -1
done
echo "[$(date -u +%H:%M)] PIPELINE_DONE"
