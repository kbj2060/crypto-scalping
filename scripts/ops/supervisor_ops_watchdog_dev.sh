#!/usr/bin/env bash
# Crash-restart watchdog for scripts/ops_watchdog_dev.py -- the dev-machine counterpart
# to the server's ops-watchdog.service, watching duckdb_persist_worker (tail_risk +
# microstructure, ETH/BTC/SOL) plus the Deribit GEX and F4-C altdata crons.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/anaconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "ops_watchdog_dev.py" \
  "$ROOT/data/live/.supervisor_ops_watchdog_dev.lock" \
  "$ROOT/logs/supervisor/ops_watchdog_dev" \
  "$PY" -u "$ROOT/scripts/ops_watchdog_dev.py" --interval-seconds 60
