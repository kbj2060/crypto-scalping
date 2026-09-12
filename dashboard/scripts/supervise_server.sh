#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PORT="${DASHBOARD_PORT:-8787}"
# 2026-09-12: 기본값을 0.0.0.0 -> 127.0.0.1. Cloudflare Access 가 터널 경로를 막았지만
# LAN 경로는 그대로 열려 있어 **집 네트워크의 누구나 인증 없이** 잔고·체결을 봤다.
# 수동 진입(실주문)이 켜지면 그 경로의 값이 "열람"에서 "내 돈으로 주문"으로 올라간다.
# 실제 소비자는 전부 루프백이다 -- cloudflared, 푸시 알림 데몬, 섀도우 러너.
# LAN 에서 봐야 하면 SSH 터널을 쓴다: ssh -N -L 8787:127.0.0.1:8787 <server>
# 워처 헬스체크는 `ss -ltn "sport = :8787"` 라 바인드 주소와 무관하게 통과한다.
HOST="${DASHBOARD_HOST:-127.0.0.1}"
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
