#!/usr/bin/env bash
# 5분 호가 요약 워커의 크래시 재기동 래퍼 (2026-10-26 ETH · 2026-10-31 XRP).
#
# 워커: scripts/live_ob5m_20261026.py (depthdiff 파일·hot 테이프 읽기만, 주문·거래소 호출 없음) → data/live/ob5m/<심볼>.sqlite
# 대시보드 풋프린트 «호가 기울기·벽 점선·▲▼◆»(/api/ob5m)가 읽는다. 같은 코인이 둘 돌면 같은 봉을 두 번 쓰므로 켜지 않는다.
#   bash scripts/ops/supervisor_ob5m.sh            # ETH (인자 없는 옛 실행 형태 그대로)
#   bash scripts/ops/supervisor_ob5m.sh XRPUSDT
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
SYM="${1:-ETHUSDT}"
case "$SYM" in ETHUSDT|XRPUSDT) ;; *) echo "모르는 코인 $SYM (워커 COINS 에 없음)" >&2; exit 2 ;; esac   # 없으면 import KeyError 를 15초마다 재기동한다
TAG="$(echo "${SYM%USDT}" | tr '[:upper:]' '[:lower:]')"
cd "$ROOT"

RUNNER="scripts/live_ob5m_20261026.py"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸려 켜지지 않았다(10-14).
# 코인별: ETH 는 인자 없는 줄(끝 $), 그 밖은 --symbol <심볼> 줄 -- 다른 코인 워커를 «이미 실행 중»으로 오인하지 않게(10-31).
if [ "$SYM" = "ETHUSDT" ]; then
  PAT="[-]u .*/${RUNNER}\$"; ARGS=()
else
  PAT="[-]u .*/${RUNNER} --symbol ${SYM}"; ARGS=(--symbol "$SYM")
fi
if pgrep -f "$PAT" >/dev/null; then
  echo "[$(date -Iseconds)] ${SYM} ob5m 워커가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_ob5m_20261026.py (${SYM})" \
  "$ROOT/data/live/.supervisor_${TAG}_ob5m.lock" \
  "$ROOT/logs/supervisor/${TAG}_ob5m" \
  nice -n 10 "$PY" -u "$ROOT/$RUNNER" "${ARGS[@]}"
