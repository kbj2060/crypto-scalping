#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
SUPERVISOR="$ROOT/scripts/supervise_trading_bot.sh"
SUPERVISOR_PID="$ROOT/data/live/trading_bot_supervisor.pid"
PID="$ROOT/data/live/trading_bot.pid"

running_bot_pid() {
  ps -eo pid=,stat=,comm=,args= | awk '
    $3 ~ /^python([0-9.]+)?$/ {
      args = ""
      for (i = 4; i <= NF; i++) args = args " " $i
      if ($2 !~ /^Z/ && args ~ /(^|[[:space:]])([^[:space:]]*\/)?trading_bot\.py([[:space:]]|$)/) {
        print $1
        exit
      }
    }
  '
}

running_supervisor_pid() {
  ps -eo pid=,stat=,comm=,args= | awk '
    $3 ~ /^(bash|sh)$/ {
      args = ""
      for (i = 4; i <= NF; i++) args = args " " $i
      if ($2 !~ /^Z/ && args ~ /(^|[[:space:]])([^[:space:]]*\/)?supervise_trading_bot\.sh([[:space:]]|$)/) {
        print $1
        exit
      }
    }
  '
}

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  echo "Set PYTHON_BIN to the quant_ai Python path." >&2
  exit 1
fi

mkdir -p "$ROOT/data/live"
"$ROOT/scripts/ops/start_ops_watchdog.sh"

bot_pid="$(running_bot_pid)"
if [[ -n "$bot_pid" ]]; then
  echo "$bot_pid" >"$PID"
  echo "trading_bot.py already running (PID=$bot_pid)."
fi

if [[ -f "$SUPERVISOR_PID" ]] && kill -0 "$(cat "$SUPERVISOR_PID")" 2>/dev/null; then
  echo "Trading bot supervisor already running (PID=$(cat "$SUPERVISOR_PID"))."
else
  supervisor_pid="$(running_supervisor_pid)"
  if [[ -n "$supervisor_pid" ]]; then
    echo "$supervisor_pid" >"$SUPERVISOR_PID"
    echo "Trading bot supervisor already running (PID=$supervisor_pid)."
  else
    rm -f "$SUPERVISOR_PID"
    if command -v setsid >/dev/null 2>&1; then
      setsid -f env PYTHON_BIN="$PY" "$SUPERVISOR" >/dev/null 2>&1
    else
      nohup env PYTHON_BIN="$PY" "$SUPERVISOR" >/dev/null 2>&1 &
    fi
    sleep 1
    supervisor_pid="$(running_supervisor_pid)"
    if [[ -z "$supervisor_pid" ]]; then
      echo "Trading bot supervisor failed to start." >&2
      exit 1
    fi
    echo "$supervisor_pid" >"$SUPERVISOR_PID"
    echo "Trading bot supervisor started (PID=$supervisor_pid)."
  fi
fi

echo "LOG=$ROOT/data/live/trading_bot_stdout.log"
