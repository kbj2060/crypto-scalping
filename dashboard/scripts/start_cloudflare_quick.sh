#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${DASHBOARD_PORT:-8787}"
LOG="$ROOT/data/live/cloudflare_quick_tunnel.log"
ERR="$ROOT/data/live/cloudflare_quick_tunnel.err"
PID="$ROOT/data/live/cloudflare_quick_tunnel.pid"
CLOUDFLARED="${CLOUDFLARED_BIN:-$ROOT/.tools/cloudflared/cloudflared}"

if [[ ! -x "$CLOUDFLARED" ]]; then
  echo "cloudflared not found: $CLOUDFLARED" >&2
  exit 1
fi

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Cloudflare quick tunnel already running (PID=$(cat "$PID"))."
  grep -Eo 'https://[-a-z0-9]+\\.trycloudflare\\.com' "$LOG" | tail -n 1 || true
  exit 0
fi

if ! curl -fsS "http://127.0.0.1:$PORT/dashboard/live/" >/dev/null 2>&1; then
  "$ROOT/dashboard/scripts/start_external.sh" >/dev/null
fi

rm -f "$LOG" "$ERR" "$PID"
if command -v setsid >/dev/null 2>&1; then
  nohup setsid "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
else
  nohup "$CLOUDFLARED" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$LOG" 2>&1 &
fi
echo "$!" > "$PID"

for _ in {1..30}; do
  url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$LOG" | tail -n 1 || true)"
  if [[ -n "$url" ]]; then
    echo "PUBLIC_URL=$url"
    exit 0
  fi
  sleep 1
done

echo "Cloudflare quick tunnel started, but URL was not found yet. Check $LOG"
