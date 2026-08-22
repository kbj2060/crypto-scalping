#!/usr/bin/env bash
# Crash-restart watchdog for the server-side ETH peg-maker fill shadow worker.
# Same _supervise.sh pattern as supervisor_oi_lsratio_worker.sh (crontab @reboot, not systemd --
# see that file's header for the sudoers reasoning).
#
# What it runs: scripts/maker_fill_shadow_worker.py -- virtual (no real orders, public streams
# only) peg/static maker legs every 5 minutes, recording effective cost per leg to validate the
# maker fill simulations (docs/experiments/eth_maker_fill_simulation_l2_20260822.md) against
# live fills. Completely independent from trading_bot.py -- no order path, no shared files.
#
# Writes to its OWN duckdb file (maker_fill_shadow.duckdb) -- single-writer-per-file principle,
# same as oi_lsratio.duckdb (DuckDB refuses concurrent writers across processes).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export MAKER_SHADOW_SYMBOL="ETHUSDT"
export MAKER_SHADOW_SPACING_S="300"
export MAKER_SHADOW_TIMEOUT_S="120"
export MAKER_SHADOW_POLICIES="peg,static"
export MAKER_SHADOW_DB_PATH="$ROOT/data/live/maker_fill_shadow.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "maker_fill_shadow_worker.py(ETH peg-maker fill shadow)" \
  "$ROOT/data/live/.supervisor_maker_fill_shadow.lock" \
  "$ROOT/logs/supervisor/maker_fill_shadow" \
  "$PY" -u "$ROOT/scripts/maker_fill_shadow_worker.py"
