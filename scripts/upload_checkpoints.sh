#!/usr/bin/env bash
# Upload coat96 checkpoints to the private Kaggle dataset gkdnzz31/knee-coat96-oai (new version if it exists).
# Usage (on the pod): bash scripts/upload_checkpoints.sh "<version message>" ckpt1.pt [ckpt2.pt ...]
set -euo pipefail
MSG="$1"; shift
D=/workspace/upload-coat96
rm -rf "$D" && mkdir -p "$D"
for f in "$@"; do ln "$f" "$D/$(basename "$f")" 2>/dev/null || cp "$f" "$D/"; done
cat > "$D/dataset-metadata.json" <<EOF
{"title": "knee-coat96-oai", "id": "gkdnzz31/knee-coat96-oai", "licenses": [{"name": "other"}]}
EOF
if kaggle datasets status gkdnzz31/knee-coat96-oai >/dev/null 2>&1; then
  kaggle datasets version -p "$D" -m "$MSG" -q
else
  kaggle datasets create -p "$D" -q
fi
for i in $(seq 1 60); do s=$(kaggle datasets status gkdnzz31/knee-coat96-oai 2>&1); echo "$s" | grep -qi ready && break; sleep 20; done
echo "dataset status: $s"
