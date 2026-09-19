#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 2026-09-20 .env 를 여기서 싣는다(사용자 지시). 이 스크립트가 **모든 기동 경로가 지나는
# 한 곳**이다 -- crontab @reboot(직접 호출) · deploy_watcher 의 되살리기 · start_external.sh ·
# ops 스크립트. 여기 없으면 «누가 띄웠느냐»에 따라 대시보드가 보는 환경이 갈린다.
# 🔴실제로 갈려 있었다(2026-09-20 실측). @reboot 는 .env 를 안 읽으므로 재부팅한 대시보드는
#   DASHBOARD_MANUAL_EXEC_ENABLED 도 BINANCE_API_KEY 도 없이 올라온다 -- 수동 주문은 403,
#   계좌 패널은 빈 채로. 워처가 고쳐주지도 않는다(되살리기는 «리슨 안 될 때»만, 재기동은
#   «dashboard/ 를 건드린 배포»가 있을 때만 돈다). 그래서 설정이 배포 타이밍에 깜빡였다.
# ⚠️그 결과 재부팅 뒤에도 수동 주문 게이트가 **켜진 채** 올라온다. 지금까지는 꺼져 있었다.
#   끄고 싶으면 .env 의 DASHBOARD_MANUAL_EXEC_ENABLED 를 내리면 된다 -- 이제 한 곳이다.
# 🔴source 는 **무조건 대입**이라 호출자가 명시한 값을 덮는다. 호출자가 이겨야 한다:
#   start_external.sh 는 DASHBOARD_HOST 를 명시해서 넘기는데, .env 가 0.0.0.0 이면 그걸로
#   덮여 **LAN 에 열린다**(아래 HOST 기본값 주석의 2026-09-12 사고가 바로 그것이다).
_keep_host="${DASHBOARD_HOST-}"; _keep_port="${DASHBOARD_PORT-}"; _keep_py="${PYTHON_BIN-}"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env" || echo "[warn] .env 를 읽지 못했습니다 -- 기본값으로 계속합니다" >&2
  set +a
fi
if [[ -n "$_keep_host" ]]; then DASHBOARD_HOST="$_keep_host"; fi
if [[ -n "$_keep_port" ]]; then DASHBOARD_PORT="$_keep_port"; fi
if [[ -n "$_keep_py"   ]]; then PYTHON_BIN="$_keep_py"; fi

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
