#!/usr/bin/env bash
# Crash-restart watchdog for the server-side ETH/BTC/SOL OI / long-short-ratio collector.
# Same _supervise.sh + crontab @reboot pattern as supervisor_tail_risk_btc_sol_worker.sh -- see
# that file's header for why this isn't a systemd unit (installing a new unit needs a sudo
# password not granted by this host's NOPASSWD sudoers whitelist). The unit file at
# scripts/ops/systemd/oi-lsratio-worker.service is kept as the same kind of reference template,
# to install instead if/when someone runs the one-time `sudo systemctl enable` step.
#
# Writes to its OWN duckdb file (oi_lsratio.duckdb) -- not tail_risk.duckdb (ETH, owned by
# trading_bot.py's TailRiskInterceptor) or tail_risk_btc_sol.duckdb (BTC/SOL, owned by this same
# duckdb_persist_worker.py script but a different supervised instance). Same single-writer-
# per-file reasoning as the BTC/SOL worker: DuckDB refuses concurrent writers from different
# processes on one file, confirmed live there on first deploy (2026-08-17).
#
# Started ETH-only 2026-08-22, extended to BTC/SOL same day (BOT_SYMBOLS below) once asked --
# backs the ETH-focused Ilias liquidation-heatmap scoping line,
# docs/experiments/eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md; BTC/SOL
# ride along for cheap cross-asset transfer checks later, same as this repo's other axes.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

# duckdb_persist_worker.py imports microstructure_scanner/tail_risk_interceptor/oi_lsratio_collector
# as top-level modules from the repo root -- running it by absolute path puts scripts/ (not the
# repo root) on sys.path[0]. Same fix as supervisor_tail_risk_btc_sol_worker.sh.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export BOT_SYMBOLS="ETHUSDT,BTCUSDT,SOLUSDT"
export COLLECT_MICROSTRUCTURE="false"
export COLLECT_TAIL_RISK="false"
export COLLECT_OI_LSRATIO="true"
export QUANT_OI_LSRATIO_DB_PATH="$ROOT/data/live/oi_lsratio.duckdb"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py(ETH+BTC+SOL OI/long-short-ratio only)" \
  "$ROOT/data/live/.supervisor_oi_lsratio_worker.lock" \
  "$ROOT/logs/supervisor/oi_lsratio_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
