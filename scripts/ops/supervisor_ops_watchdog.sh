#!/usr/bin/env bash
# Crash-restart watchdog for the read-only 24/7 operations monitor.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "ops_watchdog" \
  "$ROOT/data/live/.supervisor_ops_watchdog.lock" \
  "$ROOT/logs/supervisor/ops_watchdog" \
  "$PY" -u "$ROOT/scripts/ops_watchdog.py" --interval-seconds 30
