#!/usr/bin/env bash
# Package src/knee as the private Kaggle dataset <user>/knee-code (create or new version).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
USER_NAME="gkdnzz31"
BUILD="$ROOT/outputs/knee-code"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp -r "$ROOT/src/knee" "$BUILD/knee"
find "$BUILD" -name "__pycache__" -prune -exec rm -rf {} +
cat > "$BUILD/dataset-metadata.json" <<JSON
{"title": "knee-code", "id": "$USER_NAME/knee-code", "licenses": [{"name": "CC0-1.0"}]}
JSON
if "$KAGGLE" datasets status "$USER_NAME/knee-code" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$BUILD" -m "${1:-update}" --dir-mode zip -q
else
  "$KAGGLE" datasets create -p "$BUILD" --dir-mode zip -q
fi
