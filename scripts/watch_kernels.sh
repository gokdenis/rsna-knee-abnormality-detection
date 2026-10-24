#!/usr/bin/env bash
# Print one line per Kaggle kernel when it reaches a terminal state, then exit when all have.
# Usage: scripts/watch_kernels.sh knee-train-a knee-train-b ...
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="gkdnzz31"
remaining=("$@")
while [ ${#remaining[@]} -gt 0 ]; do
  next=()
  for k in "${remaining[@]}"; do
    s=$("$ROOT/.venv/bin/kaggle" kernels status "$USER_NAME/$k" 2>&1 || true)
    case "$s" in
      *COMPLETE*|*ERROR*|*CANCEL*) echo "$k -> ${s##*status }" ;;
      *) next+=("$k") ;;
    esac
  done
  remaining=("${next[@]}")
  [ ${#remaining[@]} -gt 0 ] && sleep 60
done
