#!/usr/bin/env bash
# Push kaggle/kernels/<name>. The pushed script is: kaggle/kaggle_paths.py + optional
# <name>/overrides.py + the code file (taken from kaggle/kernels/<base>/ when <name>/base exists).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$1"
SRC="$ROOT/kaggle/kernels/$NAME"
BUILD="$ROOT/outputs/kernels/$NAME"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cp "$SRC/kernel-metadata.json" "$BUILD/"
CODE_FILE=$("$ROOT/.venv/bin/python" -c "import json,sys;print(json.load(open(sys.argv[1]))['code_file'])" "$SRC/kernel-metadata.json")
SCRIPT_DIR="$SRC"
[ -f "$SRC/base" ] && SCRIPT_DIR="$ROOT/kaggle/kernels/$(cat "$SRC/base")"
{
  cat "$ROOT/kaggle/kaggle_paths.py"; echo
  [ -f "$SRC/overrides.py" ] && { cat "$SRC/overrides.py"; echo; }
  cat "$SCRIPT_DIR/$CODE_FILE"
} > "$BUILD/$CODE_FILE"
"$ROOT/.venv/bin/python" -m py_compile "$BUILD/$CODE_FILE"
"$ROOT/.venv/bin/kaggle" kernels push -p "$BUILD"
