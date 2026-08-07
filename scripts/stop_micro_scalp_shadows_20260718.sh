#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID="$ROOT/data/live/micro_scalp_shadow_supervisor.pid"

if [[ ! -f "$PID" ]]; then
  echo "Micro-scalp shadow supervisor is not running."
  exit 0
fi

supervisor_pid="$(cat "$PID")"
if [[ ! "$supervisor_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid supervisor PID file: $PID" >&2
  exit 1
fi

args="$(ps -p "$supervisor_pid" -o args= 2>/dev/null || true)"
if [[ "$args" != *"supervise_micro_scalp_shadows_20260718.sh"* ]]; then
  echo "PID $supervisor_pid is not the micro-scalp supervisor." >&2
  exit 1
fi

kill "$supervisor_pid"
for _attempt in $(seq 1 40); do
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    rm -f "$PID"
    echo "Micro-scalp shadow supervisor stopped."
    exit 0
  fi
  sleep 0.25
done

echo "Micro-scalp shadow supervisor did not stop within 10 seconds." >&2
exit 1

