#!/usr/bin/env bash
# Crash-restart watchdog for trading_bot.py. Real order execution stays governed by
# trading_bot.py's own config (currently OFF/dry-run) -- this script only keeps the
# decision/shadow loop itself alive, it does not change what the bot is allowed to do.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "trading_bot.py" \
  "$ROOT/data/live/.supervisor_trading_bot.lock" \
  "$ROOT/logs/supervisor/trading_bot" \
  "$PY" -u "$ROOT/trading_bot.py"
