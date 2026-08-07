#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERVISOR_PID="$ROOT/data/live/trading_bot_supervisor.pid"
PID="$ROOT/data/live/trading_bot.pid"

if [[ -f "$SUPERVISOR_PID" ]] && kill -0 "$(cat "$SUPERVISOR_PID")" 2>/dev/null; then
  kill "$(cat "$SUPERVISOR_PID")" 2>/dev/null || true
fi

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill "$(cat "$PID")" 2>/dev/null || true
fi

pkill -f "[p]ython.*trading_bot.py" 2>/dev/null || true
rm -f "$SUPERVISOR_PID" "$PID"
echo "Trading bot stopped."
