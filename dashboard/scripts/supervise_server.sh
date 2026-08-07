#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PORT="${DASHBOARD_PORT:-8787}"
HOST="${DASHBOARD_HOST:-0.0.0.0}"
LOG="$ROOT/data/live/dashboard_external.log"
ERR="$ROOT/data/live/dashboard_external.err"
CHILD_PID="$ROOT/data/live/dashboard_external.child.pid"
RESTART_DELAY="${DASHBOARD_RESTART_DELAY:-3}"

child=""

port_owner_pid() {
  ss -ltnp "sport = :$PORT" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | head -n 1
}

stop_child() {
  if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
    kill "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -f "$CHILD_PID"
}

trap 'stop_child; exit 0' INT TERM

cd "$ROOT"
mkdir -p "$ROOT/data/live"

while true; do
  owner="$(port_owner_pid || true)"
  if [[ -n "$owner" ]]; then
    {
      printf '[%s] port %s already in use by pid=%s; waiting %ss\n' "$(date -Is)" "$PORT" "$owner" "$RESTART_DELAY"
    } >>"$ERR"
    sleep "$RESTART_DELAY"
    continue
  fi

  {
    printf '[%s] dashboard server starting host=%s port=%s\n' "$(date -Is)" "$HOST" "$PORT"
  } >>"$LOG"

  "$PY" dashboard/server.py --host "$HOST" --port "$PORT" >>"$LOG" 2>>"$ERR" &
  child="$!"
  echo "$child" > "$CHILD_PID"

  set +e
  wait "$child"
  code="$?"
  set -e
  rm -f "$CHILD_PID"

  {
    printf '[%s] dashboard server exited code=%s; restarting in %ss\n' "$(date -Is)" "$code" "$RESTART_DELAY"
  } >>"$ERR"

  sleep "$RESTART_DELAY"
done
