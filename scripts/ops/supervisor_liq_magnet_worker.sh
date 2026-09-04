#!/usr/bin/env bash
# Crash-restart watchdog for the server-side liquidation-magnet (liq_cluster_price) history
# collector. Same _supervise.sh + crontab @reboot pattern as supervisor_oi_lsratio_worker.sh -- see
# that file's header for why this isn't a systemd unit (installing a new unit needs a sudo
# password not granted by this host's NOPASSWD sudoers whitelist).
#
# Writes to its OWN duckdb file (liq_magnet_history.duckdb), read-only against dashboard_state.json
# -- never touches tail_risk.duckdb (owned by trading_bot.py's TailRiskInterceptor) or
# trading_bot.py/tail_risk_interceptor.py themselves. See liq_magnet_collector.py's module
# docstring for why this file is the cheapest already-computed source (no history of the "magnet"
# reading exists anywhere else -- tail_risk_1m only stores aggregated $ totals, not the per-event
# prices the cluster computation needs).
#
# Started 2026-08-25 per user request, following up on
# docs/experiments/eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md (the prior
# scoping that found no historical liquidation-heatmap-accuracy data exists anywhere, academic or
# vendor) and today's real-liquidation-vs-estimated-map comparisons.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

# liq_magnet_collector.py imports nothing repo-local beyond stdlib/duckdb, but run from ROOT for
# consistency with the other supervisor scripts and so its relative data/live/ path resolves the
# same way regardless of caller cwd.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "liq_magnet_collector.py" \
  "$ROOT/data/live/.supervisor_liq_magnet_worker.lock" \
  "$ROOT/logs/supervisor/liq_magnet_worker" \
  "$PY" -u "$ROOT/liq_magnet_collector.py"
