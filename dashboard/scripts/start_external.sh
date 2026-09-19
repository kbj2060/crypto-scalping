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
