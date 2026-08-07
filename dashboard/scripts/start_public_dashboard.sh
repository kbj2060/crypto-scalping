#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUPERVISOR="$ROOT/dashboard/scripts/supervise_cloudflare_tunnel.sh"
SUPERVISOR_PID="$ROOT/data/live/cloudflare_named_tunnel_supervisor.pid"

mkdir -p "$ROOT/data/live"

is_supervisor_process() {
  local pid="${1:-}"
  local cmdline
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *"$SUPERVISOR"* ]]
}

"$ROOT/dashboard/scripts/start_external.sh"

existing_supervisor_pid="$(cat "$SUPERVISOR_PID" 2>/dev/null || true)"
if is_supervisor_process "$existing_supervisor_pid"; then
  echo "Cloudflare tunnel supervisor already running (PID=$existing_supervisor_pid)."
else
  rm -f "$SUPERVISOR_PID"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$SUPERVISOR" >/dev/null 2>&1 &
  else
    nohup "$SUPERVISOR" >/dev/null 2>&1 &
  fi
  echo "$!" > "$SUPERVISOR_PID"
  echo "Cloudflare tunnel supervisor started (PID=$(cat "$SUPERVISOR_PID"))."
fi

echo "PUBLIC_URL=https://thesan.xyz/dashboard/live/"
