#!/usr/bin/env bash
# Crash-restart watchdog for the BTC multi-slot (N=3) shadow loop (2026-08-07).
# Shadow-only, no order submission -- live-forward A/B against the in-bot single-slot BTC shadow.
# Gate evidence: tmp/causal_regen_20260516/btc_swingtransition_multislot_20260807/report.json
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "btc_multislot_shadow" \
  "$ROOT/data/live/.supervisor_btc_multislot_shadow.lock" \
  "$ROOT/logs/supervisor/btc_multislot_shadow" \
  "$PY" -u "$ROOT/scripts/run_btc_multislot_shadow_loop_20260807.py"
