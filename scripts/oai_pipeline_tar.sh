#!/usr/bin/env bash
# OAI parts one after another, without unpacking: download the part's tarballs (retrying missing ones),
# build the coat96 cache straight from the tarballs, then delete them. The next part's download starts while
# the current part is being cached.
# Usage: bash scripts/oai_pipeline_tar.sh V00p0:<V00_PACKAGE_ID> V00p1:<V00_PACKAGE_ID> ... V06:<V06_PACKAGE_ID>
set -u
cd /workspace/new_knee
L=/workspace/logs
links_of() { local tag=$1 v=${1%%p*} part=${1#${1%%p*}}; echo /workspace/oai/$v/links${part:+_$part}.txt; }
download() {  # tag package: download until every listed tarball is present
  local tag=$1 p=$2 D=/workspace/oai/$1 LINKS; LINKS=$(links_of $tag)
  for attempt in 1 2 3 4 5 6; do
    python3 -c "
import os, sys
raw, links, rest = sys.argv[1:]
need = [l.strip() for l in open(links) if l.strip()]
miss = [u for u in need if not os.path.isfile(os.path.join(raw, 'image03', u.split('/', 4)[4]))]
open(rest, 'w').write('\n'.join(miss) + ('\n' if miss else ''))
print(len(miss))" $D/raw $LINKS $D/links_missing.txt > $D/missing_count
    [ "$(cat $D/missing_count)" = 0 ] && return 0
    find $D/raw -name "*.partial" -delete 2>/dev/null
    echo "[$(date -u +%H:%M)] $tag: downloading $(cat $D/missing_count) files (attempt $attempt)"
    downloadcmd -dp $p -u $NDA_USER -d $D/raw -wt 32 -t $D/links_missing.txt < /dev/null > $L/dl_$tag.$attempt.log 2>&1
  done
  return 1
}
prev=""
for vp in "$@"; do
  tag=${vp%:*}; p=${vp#*:}; mkdir -p /workspace/oai/$tag
  [ -f /workspace/cache-oai-$tag/stats.json ] && { echo "$tag cache exists"; continue; }
  download $tag $p || echo "[$(date -u +%H:%M)] $tag: download incomplete, caching what is present"
  [ -n "$prev" ] && wait $prev   # the previous part's cache build
  v=${tag%%p*}
  ( echo "[$(date -u +%H:%M)] $tag: build cache"
    python3 -I scripts/build_oai_cache_tar.py --links $(links_of $tag) --table /workspace/oai/$v/links.csv \
      --raw /workspace/oai/$tag/raw --visit $v --out /workspace/cache-oai-$tag --workers 9 > $L/cache_$tag.log 2>&1
    tail -12 $L/cache_$tag.log
    [ -f /workspace/cache-oai-$tag/stats.json ] && rm -rf /workspace/oai/$tag/raw /workspace/oai/$tag/img
    echo "[$(date -u +%H:%M)] $tag: done"; df -h /workspace | tail -1 ) &
  prev=$!
done
wait
echo "[$(date -u +%H:%M)] PIPELINE_DONE"
