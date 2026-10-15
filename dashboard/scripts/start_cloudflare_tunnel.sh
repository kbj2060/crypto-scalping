#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PORT="${DASHBOARD_PORT:-8787}"
TOKEN="${CLOUDFLARE_TUNNEL_TOKEN:-}"
ORIGIN_URL="${CLOUDFLARE_TUNNEL_ORIGIN_URL:-http://127.0.0.1:$PORT}"
LOG="$ROOT/data/live/cloudflare_named_tunnel.log"
ERR="$ROOT/data/live/cloudflare_named_tunnel.err"
PID="$ROOT/data/live/cloudflare_named_tunnel.pid"
TOKEN_FILE="$ROOT/data/live/cloudflare_named_tunnel.token"
CLOUDFLARED="${CLOUDFLARED_BIN:-}"

is_tunnel_process() {
  local pid="${1:-}"
  local cmdline
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmdline" == *cloudflared* && "$cmdline" == *" tunnel "* && "$cmdline" == *"--token-file"* ]]
}

if [[ -z "$TOKEN" ]]; then
  echo "CLOUDFLARE_TUNNEL_TOKEN is required." >&2
  echo "Create a Cloudflare Tunnel for thesan.xyz, then export its token:" >&2
  echo "export CLOUDFLARE_TUNNEL_TOKEN='...'" >&2
  exit 1
fi

if [[ -z "$CLOUDFLARED" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED="$(command -v cloudflared)"
  elif [[ -x "$ROOT/.tools/cloudflared/cloudflared" ]]; then
    CLOUDFLARED="$ROOT/.tools/cloudflared/cloudflared"
  elif [[ -x "$ROOT/.tools/cloudflared/cloudflared.exe" ]]; then
    CLOUDFLARED="$ROOT/.tools/cloudflared/cloudflared.exe"
  else
    echo "cloudflared not found. Run dashboard/scripts/start_cloudflare.ps1 once or install cloudflared." >&2
    exit 1
  fi
fi

mkdir -p "$ROOT/data/live"

existing_pid="$(cat "$PID" 2>/dev/null || true)"
if is_tunnel_process "$existing_pid"; then
  echo "Cloudflare tunnel already running (PID=$existing_pid)."
  exit 0
fi
rm -f "$PID"

# 🔴2026-10-16: 여기서 대시보드가 안 응답하면 server.py 를 **직접** 띄웠다. 부팅 때 대시보드는 콜드 기동
#   중이라(분 단위) 매번 이 길을 탔고, supervise_server.sh 의 자식과 포트를 다퉈 이긴 쪽이 **감독 없이**
#   서빙했다(10-15 pid 1468, 감독은 8시간 «port already in use»). 대시보드는 supervise_server.sh 만 띄운다 --
#   원본이 아직 안 떴으면 cloudflared 는 502 를 주다가 뜨면 그대로 붙는다.

cd "$ROOT"
umask 077
printf '%s' "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

nohup setsid env -u CLOUDFLARE_TUNNEL_TOKEN "$CLOUDFLARED" tunnel --no-autoupdate --url "$ORIGIN_URL" run --token-file "$TOKEN_FILE" >>"$LOG" 2>>"$ERR" &
echo "$!" > "$PID"
sleep 2

if ! kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Cloudflare tunnel failed to stay running. Check:" >&2
  echo "  $LOG" >&2
  echo "  $ERR" >&2
  rm -f "$PID"
  exit 1
fi

echo "Cloudflare tunnel started (PID=$(cat "$PID"))."
