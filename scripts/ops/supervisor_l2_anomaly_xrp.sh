#!/usr/bin/env bash
# Crash-restart watchdog for l2_anomaly_snapshot_collector.py on XRPUSDT -- see
# supervisor_l2_anomaly_btc.sh for the full rationale (identical pattern, symbol swapped).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export L2_ANOMALY_SYMBOL="xrpusdt"
export L2_ANOMALY_DB_PATH="$ROOT/data/live/l2_anomaly_snapshots_xrp.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "l2_anomaly_snapshot_collector.py(XRP)" \
  "$ROOT/data/live/.supervisor_l2_anomaly_xrp.lock" \
  "$ROOT/logs/supervisor/l2_anomaly_xrp" \
  "$PY" -u "$ROOT/l2_anomaly_snapshot_collector.py"
