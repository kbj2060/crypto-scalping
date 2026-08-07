#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID="$ROOT/data/live/dashboard_external.pid"
CHILD_PID="$ROOT/data/live/dashboard_external.child.pid"

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill "$(cat "$PID")"
  if [[ -f "$CHILD_PID" ]] && kill -0 "$(cat "$CHILD_PID")" 2>/dev/null; then
    kill "$(cat "$CHILD_PID")" 2>/dev/null || true
  fi
  rm -f "$PID" "$CHILD_PID"
  echo "Dashboard server stopped."
  exit 0
fi

pkill -f "dashboard/server.py --host" 2>/dev/null || true
rm -f "$PID" "$CHILD_PID"
echo "No dashboard server PID found."
