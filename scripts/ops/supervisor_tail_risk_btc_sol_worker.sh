#!/usr/bin/env bash
# Crash-restart watchdog for the server-side BTC/SOL liquidation (tail-risk) collector.
# Deliberately NOT a systemd unit: installing a new unit requires a sudo password not granted
# by this host's NOPASSWD sudoers whitelist (scripts/ops/systemd/deploy_watcher_sudoers only
# covers `restart` on already-installed unit names). This _supervise.sh + crontab @reboot
# pattern needs no elevated privileges and gives the same crash-restart/boot-start coverage --
# see scripts/ops/systemd/tail-risk-btc-sol-worker.service for the unit file to install instead
# if/when someone runs the one-time `sudo systemctl enable` step.
#
# Writes to its OWN duckdb file (tail_risk_btc_sol.duckdb), not data/live/tail_risk.duckdb.
# First deploy (2026-08-17) pointed at the shared file and caused two real write failures in
# trading_bot.py's TailRiskInterceptor (DuckDB refuses concurrent writers from different
# processes on the same file -- confirmed live via journalctl, immediately rolled back). A
# separate file makes that class of conflict structurally impossible instead of relying on
# retry timing, at the cost of tail-risk data living in two files (ETH in tail_risk.duckdb,
# BTC/SOL in tail_risk_btc_sol.duckdb) -- trading_bot.py is not touched or restarted by this.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

# duckdb_persist_worker.py imports microstructure_scanner/tail_risk_interceptor as top-level
# modules from the repo root -- running it by absolute path puts scripts/ (not the repo root)
# on sys.path[0]. Same fix as supervisor_duckdb_persist_worker.sh.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export BOT_SYMBOLS="BTCUSDT,SOLUSDT"
export COLLECT_MICROSTRUCTURE="false"
export COLLECT_TAIL_RISK="true"
export QUANT_TAIL_DB_PATH="$ROOT/data/live/tail_risk_btc_sol.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py(BTC+SOL tail-risk only)" \
  "$ROOT/data/live/.supervisor_tail_risk_btc_sol_worker.lock" \
  "$ROOT/logs/supervisor/tail_risk_btc_sol_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
