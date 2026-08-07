#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID="$ROOT/data/live/cloudflare_named_tunnel.pid"
LOG="$ROOT/data/live/cloudflare_named_tunnel_supervisor.log"
ERR="$ROOT/data/live/cloudflare_named_tunnel_supervisor.err"
RESTART_DELAY="${CLOUDFLARE_TUNNEL_RESTART_DELAY:-5}"

mkdir -p "$ROOT/data/live"

is_tunnel_process() {
  local pid="${1:-}"
  local cmdline
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *cloudflared* && "$cmdline" == *" tunnel "* && "$cmdline" == *"--token-file"* ]]
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$LOG"
}

err() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >>"$ERR"
}

stop_tunnel() {
  local tunnel_pid
  tunnel_pid="$(cat "$PID" 2>/dev/null || true)"
  if is_tunnel_process "$tunnel_pid"; then
    kill "$tunnel_pid" 2>/dev/null || true
  fi
}

trap 'stop_tunnel; exit 0' INT TERM

while true; do
  tunnel_pid="$(cat "$PID" 2>/dev/null || true)"
  if is_tunnel_process "$tunnel_pid"; then
    log "cloudflare tunnel running pid=$tunnel_pid"
    while is_tunnel_process "$tunnel_pid"; do
      sleep "$RESTART_DELAY"
    done
    err "cloudflare tunnel pid=$tunnel_pid exited; restarting in ${RESTART_DELAY}s"
    rm -f "$PID"
    sleep "$RESTART_DELAY"
    continue
  fi

  rm -f "$PID"
  log "starting cloudflare tunnel"
  "$ROOT/dashboard/scripts/start_cloudflare_tunnel.sh" >>"$LOG" 2>>"$ERR"
  code="$?"
  if [[ "$code" -ne 0 ]]; then
    err "start_cloudflare_tunnel.sh failed code=$code; retrying in ${RESTART_DELAY}s"
    sleep "$RESTART_DELAY"
    continue
  fi

  sleep 1
done
