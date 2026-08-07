#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID="$ROOT/data/live/cloudflare_quick_tunnel.pid"

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill "$(cat "$PID")" 2>/dev/null || true
  rm -f "$PID"
  echo "Cloudflare quick tunnel stopped."
  exit 0
fi

rm -f "$PID"
echo "No Cloudflare quick tunnel PID found."
