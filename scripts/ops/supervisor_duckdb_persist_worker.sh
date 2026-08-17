#!/usr/bin/env bash
# Crash-restart watchdog for scripts/duckdb_persist_worker.py (MicrostructureScanner +
# TailRiskInterceptor -- feeds data/live/microstructure.duckdb and data/live/tail_risk.duckdb).
# Dev-machine-only research collector, not part of the server's live-trading supervision.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/anaconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

# duckdb_persist_worker.py imports microstructure_scanner/tail_risk_interceptor as
# top-level modules from the repo root, but running it by absolute path puts
# scripts/ (not the repo root) on sys.path[0] -- export PYTHONPATH to fix that
# without touching the script itself.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py" \
  "$ROOT/data/live/.supervisor_duckdb_persist_worker.lock" \
  "$ROOT/logs/supervisor/duckdb_persist_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
