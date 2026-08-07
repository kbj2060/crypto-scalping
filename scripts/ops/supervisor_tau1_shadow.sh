#!/usr/bin/env bash
# Crash-restart watchdog for the Tau1 (sigma6_regime_tiebreak_shadow) live shadow tracker.
# Shadow-only, no order_submission_supported -- see the script's own docstring.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "tau1_shadow" \
  "$ROOT/data/live/.supervisor_tau1_shadow.lock" \
  "$ROOT/logs/supervisor/tau1_shadow" \
  "$PY" -u "$ROOT/scripts/live_sigma6_regime_tiebreak_shadow_20260801.py" --poll-seconds 90
