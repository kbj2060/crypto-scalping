#!/usr/bin/env bash
# Crash-restart watchdog for l2_anomaly_snapshot_collector.py on BTCUSDT, 2026-08-28.
# ETH's instance runs unsupervised (started ad-hoc, default L2_ANOMALY_SYMBOL=ethusdt / default
# DB_PATH, both left untouched for backward compat with existing consumers) -- this and its sol/
# xrp/hype siblings are the first supervised instances, each writing to its OWN duckdb file
# (l2_anomaly_snapshots_btc.duckdb) via the newly env-parameterized DB_PATH (l2_anomaly_snapshot_
# collector.py, was hardcoded before today) -- same single-writer-per-file reasoning as every
# other worker here. Same _supervise.sh + crontab @reboot pattern as supervisor_xrp_worker.sh.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export L2_ANOMALY_SYMBOL="btcusdt"
export L2_ANOMALY_DB_PATH="$ROOT/data/live/l2_anomaly_snapshots_btc.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "l2_anomaly_snapshot_collector.py(BTC)" \
  "$ROOT/data/live/.supervisor_l2_anomaly_btc.lock" \
  "$ROOT/logs/supervisor/l2_anomaly_btc" \
  "$PY" -u "$ROOT/l2_anomaly_snapshot_collector.py"
