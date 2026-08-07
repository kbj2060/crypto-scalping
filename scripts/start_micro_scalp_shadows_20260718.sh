#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-$ROOT/venv/bin/python}"
SUPERVISOR="$ROOT/scripts/supervise_micro_scalp_shadows_20260718.sh"
PID="$ROOT/data/live/micro_scalp_shadow_supervisor.pid"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  exit 1
fi

mkdir -p "$ROOT/data/live"

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Micro-scalp shadow supervisor already running (PID=$(cat "$PID"))."
  exit 0
fi

rm -f "$PID"
if command -v setsid >/dev/null 2>&1; then
  setsid -f env PYTHON_BIN="$PY" "$SUPERVISOR" >/dev/null 2>&1
else
  nohup env PYTHON_BIN="$PY" "$SUPERVISOR" >/dev/null 2>&1 &
fi

for _attempt in $(seq 1 20); do
  if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "Micro-scalp shadow supervisor started (PID=$(cat "$PID"))."
    echo "LOG=$ROOT/data/live/micro_scalp_shadow_supervisor.log"
    exit 0
  fi
  sleep 0.25
done

echo "Micro-scalp shadow supervisor failed to start." >&2
exit 1

