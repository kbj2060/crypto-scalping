#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
PORT="${DASHBOARD_PORT:-8787}"
# 🔴2026-09-20: 여기가 0.0.0.0 이었다. supervise_server.sh 는 2026-09-12 에 127.0.0.1 로
# 고쳤지만 이 스크립트가 HOST 를 **명시해서** 넘기므로(아래 env ... DASHBOARD_HOST="$HOST")
# 그 수정이 이 경로에서는 한 번도 적용되지 않았다 -- 반쪽만 고쳐져 있었다.
# deploy_watcher 의 revive_serving_if_down() 이 이 스크립트를 환경변수 없이 부르고 .env 에도
# DASHBOARD_HOST 가 없으므로, **대시보드가 죽었다 되살아날 때마다 LAN 에 열렸다.**
# 수동 주문 게이트가 켜진 지금 그건 «열람»이 아니라 «인증 없이 내 돈으로 주문»이다.
HOST="${DASHBOARD_HOST:-127.0.0.1}"
LOG="$ROOT/data/live/dashboard_external.log"
ERR="$ROOT/data/live/dashboard_external.err"
PID="$ROOT/data/live/dashboard_external.pid"
SUPERVISOR="$ROOT/dashboard/scripts/supervise_server.sh"

if [[ ! -x "$PY" ]]; then
  echo "Python not found: $PY" >&2
  echo "Set PYTHON_BIN to the quant_ai Python path." >&2
  exit 1
fi

if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "Dashboard server already running (PID=$(cat "$PID"), port=$PORT)."
  echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
  exit 0
fi

if curl -fsS "http://127.0.0.1:$PORT/dashboard/live/" >/dev/null 2>&1; then
  echo "Dashboard server already responding (port=$PORT)."
  echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
  exit 0
fi

mkdir -p "$ROOT/data/live"

# 🔴여기까지 왔다는 건 위 두 검사(pid 파일 · curl)가 «안 돈다»고 봤다는 뜻인데, **둘 다 틀릴
#   수 있다**: pid 파일은 바로 아래에서 우리가 지우므로 낡아 있기 쉽고, curl 은 supervisor 가
#   자식을 재기동하는 3초 창에 실패한다. deploy_watcher 의 revive_serving_if_down() 이
#   정확히 그 창에서 이 스크립트를 부른다 -- 그래서 두 번째 supervisor 가 떠서 **이틀을
#   살았다**(2026-09-20 실측, pid 759). 그 유령은 진짜 supervisor 가 포트를 놓는 순간
#   끼어들어 **제 환경으로** 자식을 띄우므로, .env 로 켠 설정이 없는 대시보드가 올라온다.
#   잠금을 **잡지 않고 확인만** 한다(서브셸이 곧 놓는다). 실제 방어는 supervise_server.sh
#   안의 flock 이고, 여기 검사는 아래 rm -f / pkill 로 **남의 서빙을 건드리지 않기 위한** 것이다.
LOCK="$ROOT/data/live/dashboard_external.lock"
if ! ( exec 9>"$LOCK"; flock -n 9 ) 2>/dev/null; then
  echo "Dashboard supervisor already running (lock held) -- not starting a second one."
  echo "  자식이 멈춘 거라면 supervisor 가 ${DASHBOARD_RESTART_DELAY:-3}초 뒤 스스로 되살립니다."
  echo "  supervisor 자체를 갈아야 하면 그 프로세스를 먼저 종료하세요."
  echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
  exit 0
fi

rm -f "$LOG" "$ERR" "$PID" "$ROOT/data/live/dashboard_external.child.pid"
cd "$ROOT"
pkill -f "dashboard/server.py --host .* --port $PORT" 2>/dev/null || true
if command -v setsid >/dev/null 2>&1; then
  nohup setsid env PYTHON_BIN="$PY" DASHBOARD_HOST="$HOST" DASHBOARD_PORT="$PORT" "$SUPERVISOR" >/dev/null 2>&1 &
else
  nohup env PYTHON_BIN="$PY" DASHBOARD_HOST="$HOST" DASHBOARD_PORT="$PORT" "$SUPERVISOR" >/dev/null 2>&1 &
fi
echo "$!" > "$PID"

for _ in {1..40}; do
  if ! kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "Dashboard server exited during startup." >&2
    echo "Check $ERR" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/dashboard/live/" >/dev/null 2>&1; then
    echo "Dashboard supervisor started (PID=$(cat "$PID"), port=$PORT)."
    echo "LOCAL_URL=http://127.0.0.1:$PORT/dashboard/live/"
    exit 0
  fi
  sleep 0.5
done

echo "Dashboard server started, but readiness check did not complete yet."
echo "Check $LOG and $ERR"
