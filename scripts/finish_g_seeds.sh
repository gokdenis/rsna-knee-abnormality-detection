#!/usr/bin/env bash
# Local watcher for run_g_seeds.sh: wait until the pod writes GS_DONE, copy the SWA checkpoints and logs home,
# publish a new version of gkdnzz31/knee-coat96-oai-g with every G checkpoint (a version replaces all files),
# verify it on Kaggle, then stop the pod. The pod is stopped even if training failed or the pod stopped answering.
# Usage: bash scripts/finish_g_seeds.sh <ssh_config> <pod_id> <local_dir> [manifest marker "tags" "logs"]
set -u
CFG=$1; POD=$2; OUT=$3
MAN=${4:-gs_manifest.txt}; MARK=${5:-GS_DONE}; TAGS=${6:-"g0 g1 g2 g3 h1"}; LOGS=${7:-"train_g2.log train_g3.log train_h1.log predict_224.log"}
K="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/kaggle"
SSH="ssh -F $CFG pod"
mkdir -p "$OUT/upload" "$OUT/logs"
fails=0
until $SSH "grep -q $MARK /workspace/runs/$MAN" 2>/dev/null; do
  if $SSH true 2>/dev/null; then fails=0; else fails=$((fails+1)); fi
  [ $fails -ge 6 ] && { echo "[$(date -u +%H:%M)] pod unreachable for 30 min"; break; }
  sleep 300
done
echo "[$(date -u +%H:%M)] training finished:"; $SSH "cat /workspace/runs/$MAN" 2>&1
for f in runs/$MAN $(for l in $LOGS; do echo logs/$l; done); do rsync -rlt -e "ssh -F $CFG" pod:/workspace/$f "$OUT/logs/" 2>&1 | tail -1; done
rsync -rlt -e "ssh -F $CFG" pod:/workspace/runs/release224_train_preds.csv "$OUT/" 2>/dev/null
for t in $TAGS; do
  rsync -rlt --partial -e "ssh -F $CFG" pod:/workspace/runs/$t/coat96_${t}_swa.pt "$OUT/upload/" 2>/dev/null \
    && echo "copied $t" || echo "no $t checkpoint"
done
ls -l "$OUT/upload"
expected=$(ls "$OUT/upload"/*.pt | wc -l | tr -d ' ')
if [ "$expected" -ge 3 ]; then
  echo '{"title": "knee-coat96-oai-g", "id": "gkdnzz31/knee-coat96-oai-g", "licenses": [{"name": "other"}]}' \
    > "$OUT/upload/dataset-metadata.json"
  $K datasets version -p "$OUT/upload" -m "coat96 G checkpoints: $(ls "$OUT/upload" | grep -c pt) files" -q 2>&1 | tail -2
  ok=0
  # verify by name and size: every local checkpoint must be in the new version (not just the same count)
  want=$(cd "$OUT/upload" && for f in *.pt; do echo "$f $(stat -f %z "$f")"; done | sort)
  for i in $(seq 1 90); do
    have=$($K datasets files gkdnzz31/knee-coat96-oai-g 2>/dev/null | awk '/\.pt /{print $1, $2}' | sort)
    [ "$have" = "$want" ] && { ok=1; break; }
    sleep 60
  done
  echo "kaggle files (name size), ok=$ok:"; echo "$have"
  (cd "$OUT/upload" && shasum -a 256 *.pt) > "$OUT/upload_sha256.txt"
else
  echo "fewer than 3 checkpoints copied; nothing uploaded"
fi
runpodctl pod stop "$POD" 2>&1 | grep -E '"desiredStatus"|error'
runpodctl pod get "$POD" 2>&1 | grep -E '"desiredStatus"'
echo FINISH_DONE
