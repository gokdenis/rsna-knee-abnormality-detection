#!/usr/bin/env bash
# Kill downloadcmd when its newest log shows no progress for 10 minutes (it can hang on a single file);
# oai_pipeline_tar.sh then retries the missing files.
while pgrep -f "[o]ai_pipeline_tar.sh" >/dev/null; do
  log=$(ls -t /workspace/logs/dl_*.log 2>/dev/null | head -1)
  if [ -n "$log" ] && pgrep -f "[d]ownloadcmd -dp" >/dev/null; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt 600 ]; then
      echo "[$(date -u +%H:%M)] no progress in $log for ${age}s: killing downloadcmd"
      pkill -f "[d]ownloadcmd -dp"
    fi
  fi
  sleep 60
done
