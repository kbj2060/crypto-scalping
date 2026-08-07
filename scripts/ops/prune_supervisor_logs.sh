#!/usr/bin/env bash
# Prune logs/supervisor/*.log older than N days. logs/supervisor grows one file
# per managed process per day with no built-in rotation (unlike ops_watchdog's
# own retain_history(), which already prunes data/live/ops_watchdog/history).
#
# Usage: run periodically from cron, e.g. daily:
#   30 3 * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/prune_supervisor_logs.sh >> logs/prune_supervisor_logs_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DAYS="${SUPERVISOR_LOG_RETENTION_DAYS:-30}"
LOG_DIR="$ROOT/logs/supervisor"

[[ -d "$LOG_DIR" ]] || exit 0

deleted=0
while IFS= read -r -d '' path; do
  rm -f -- "$path"
  deleted=$((deleted + 1))
done < <(find "$LOG_DIR" -maxdepth 1 -name '*.log' -type f -mtime "+${DAYS}" -print0)

echo "[$(date -Iseconds)] pruned ${deleted} log file(s) older than ${DAYS}d from ${LOG_DIR}"
