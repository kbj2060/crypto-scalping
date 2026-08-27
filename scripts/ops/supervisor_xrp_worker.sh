#!/usr/bin/env bash
# Crash-restart watchdog for the server-side XRP real-time data worker (microstructure +
# tail-risk + OI/long-short-ratio, all three -- unlike the BTC/SOL and ETH+BTC+SOL workers,
# no trading_bot.py process owns any collector for XRP, so this single worker can safely
# collect everything for it with no risk of fighting another process for a symbol it owns.
# Same _supervise.sh + crontab @reboot pattern as the other duckdb_persist_worker.py deployments
# -- see supervisor_tail_risk_btc_sol_worker.sh's header for why this isn't a systemd unit.
#
# Writes to its OWN three duckdb files (microstructure_xrp.duckdb / tail_risk_xrp.duckdb /
# oi_lsratio_xrp.duckdb), not the shared microstructure.duckdb / tail_risk.duckdb /
# oi_lsratio.duckdb files other processes already hold open -- same single-writer-per-file
# reasoning as every other worker here (DuckDB refuses concurrent writers from different
# processes on one file, confirmed live 2026-08-17). Kept as three separate files rather than
# one shared file across collector types since that combination (different collector classes,
# same file, same process) has no precedent elsewhere in this repo to confirm it's safe.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

# duckdb_persist_worker.py imports microstructure_scanner/tail_risk_interceptor/oi_lsratio_collector
# as top-level modules from the repo root -- running it by absolute path puts scripts/ (not the
# repo root) on sys.path[0]. Same fix as the other duckdb_persist_worker.py supervisors.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export BOT_SYMBOLS="XRPUSDT"
export COLLECT_MICROSTRUCTURE="true"
export COLLECT_TAIL_RISK="true"
export COLLECT_OI_LSRATIO="true"
export QUANT_MICRO_DB_PATH="$ROOT/data/live/microstructure_xrp.duckdb"
export QUANT_TAIL_DB_PATH="$ROOT/data/live/tail_risk_xrp.duckdb"
export QUANT_OI_LSRATIO_DB_PATH="$ROOT/data/live/oi_lsratio_xrp.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py(XRP microstructure+tail-risk+OI/long-short-ratio)" \
  "$ROOT/data/live/.supervisor_xrp_worker.lock" \
  "$ROOT/logs/supervisor/xrp_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
