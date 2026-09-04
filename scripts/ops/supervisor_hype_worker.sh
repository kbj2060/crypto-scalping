#!/usr/bin/env bash
# Crash-restart watchdog for the server-side HYPE real-time data worker (microstructure +
# tail-risk + OI/long-short-ratio, all three) -- exact clone of supervisor_xrp_worker.sh's
# pattern (2026-08-27): no trading_bot.py process owns any collector for HYPE, so this single
# worker can safely collect everything for it with no risk of fighting another process for a
# symbol it owns. Same _supervise.sh + crontab @reboot pattern as the other duckdb_persist_
# worker.py deployments -- see supervisor_tail_risk_btc_sol_worker.sh's header for why this
# isn't a systemd unit.
#
# Writes to its OWN three duckdb files (microstructure_hype.duckdb / tail_risk_hype.duckdb /
# oi_lsratio_hype.duckdb), not the shared files other processes already hold open -- same
# single-writer-per-file reasoning as every other worker here.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export BOT_SYMBOLS="HYPEUSDT"
export COLLECT_MICROSTRUCTURE="true"
export COLLECT_TAIL_RISK="true"
export COLLECT_OI_LSRATIO="true"
export QUANT_MICRO_DB_PATH="$ROOT/data/live/microstructure_hype.duckdb"
export QUANT_TAIL_DB_PATH="$ROOT/data/live/tail_risk_hype.duckdb"
export QUANT_OI_LSRATIO_DB_PATH="$ROOT/data/live/oi_lsratio_hype.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py(HYPE microstructure+tail-risk+OI/long-short-ratio)" \
  "$ROOT/data/live/.supervisor_hype_worker.lock" \
  "$ROOT/logs/supervisor/hype_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
