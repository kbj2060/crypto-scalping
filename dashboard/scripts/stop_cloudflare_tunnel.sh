#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID="$ROOT/data/live/cloudflare_named_tunnel.pid"
SERVER_PID="$ROOT/data/live/cloudflare_named_tunnel.server.pid"
TOKEN_FILE="$ROOT/data/live/cloudflare_named_tunnel.token"

is_process() {
  local pid="${1:-}"
  local pattern="$2"
  local cmdline
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$pattern"* ]]
}

tunnel_pid="$(cat "$PID" 2>/dev/null || true)"
if is_process "$tunnel_pid" "cloudflared"; then
  kill "$tunnel_pid" 2>/dev/null || true
fi

server_pid="$(cat "$SERVER_PID" 2>/dev/null || true)"
if is_process "$server_pid" "dashboard/server.py"; then
  kill "$server_pid" 2>/dev/null || true
fi

rm -f "$PID" "$SERVER_PID" "$TOKEN_FILE"
echo "Cloudflare named tunnel stopped."
