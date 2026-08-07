#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PORT="${DASHBOARD_PORT:-8787}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
LOG="$ROOT/data/live/dashboard_external.log"
ERR="$ROOT/data/live/dashboard_external.err"
PID="$ROOT/data/live/dashboard_external.pid"
SUPERVISOR="$ROOT/dashboard/scripts/supervise_server.sh"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  echo "Set PYTHON_BIN to the quant_ai Python path." >&2
  exit 1
fi

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Dashboard server already running (PID=$(cat "$PID"), port=$PORT)."
  echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
  exit 0
fi

if curl -fsS "http://127.0.0.1:$PORT/dashboard/live/" >/dev/null 2>&1; then
  echo "Dashboard server already responding (port=$PORT)."
  echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
  exit 0
fi

mkdir -p "$ROOT/data/live"
rm -f "$LOG" "$ERR" "$PID" "$ROOT/data/live/dashboard_external.child.pid"
cd "$ROOT"
pkill -f "dashboard/server.py --host .* --port $PORT" 2>/dev/null || true
if command -v setsid >/dev/null 2>&1; then
  nohup setsid env PYTHON_BIN="$PY" DASHBOARD_HOST="$HOST" DASHBOARD_PORT="$PORT" "$SUPERVISOR" >/dev/null 2>&1 &
else
  nohup env PYTHON_BIN="$PY" DASHBOARD_HOST="$HOST" DASHBOARD_PORT="$PORT" "$SUPERVISOR" >/dev/null 2>&1 &
fi
echo "$!" > "$PID"

for _ in {1..40}; do
  if ! kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "Dashboard server exited during startup." >&2
    echo "Check $ERR" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/dashboard/live/" >/dev/null 2>&1; then
    echo "Dashboard supervisor started (PID=$(cat "$PID"), port=$PORT)."
    echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
    exit 0
  fi
  sleep 0.5
done

echo "Dashboard server started, but readiness check did not complete yet."
echo "Check $LOG and $ERR"
