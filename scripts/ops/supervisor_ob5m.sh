#!/usr/bin/env bash
# ETH 5분 호가 요약 워커의 크래시 재기동 래퍼 (2026-10-26).
#
# 워커: scripts/live_ob5m_20261026.py (depthdiff 파일·hot 테이프 읽기만, 주문·거래소 호출 없음) → data/live/ob5m/ethusdt.sqlite
# 대시보드 풋프린트 «호가 기울기·벽 점선·▲▼◆»(/api/ob5m)가 읽는다. 둘이 돌면 같은 봉을 두 번 쓰므로 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_ob5m_20261026.py"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸려 켜지지 않았다(10-14).
if pgrep -f "[-]u .*/${RUNNER}" >/dev/null; then
  echo "[$(date -Iseconds)] ob5m 워커가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_ob5m_20261026.py (ETHUSDT)" \
  "$ROOT/data/live/.supervisor_eth_ob5m.lock" \
  "$ROOT/logs/supervisor/eth_ob5m" \
  nice -n 10 "$PY" -u "$ROOT/$RUNNER"
