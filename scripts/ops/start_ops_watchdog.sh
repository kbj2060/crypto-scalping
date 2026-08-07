#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPERVISOR="$ROOT/scripts/ops/supervisor_ops_watchdog.sh"

if pgrep -f "[s]cripts/ops/_supervise.sh ops_watchdog " >/dev/null; then
  echo "ops_watchdog supervisor already running."
  exit 0
fi

mkdir -p "$ROOT/logs/supervisor"
nohup setsid "$SUPERVISOR" >/dev/null 2>&1 < /dev/null &
echo "ops_watchdog supervisor started (PID=$!)."
