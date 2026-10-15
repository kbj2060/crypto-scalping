#!/usr/bin/env bash
# ETH 칸별 아이스버그 실시간 탐지기의 크래시 재기동 래퍼 (2026-10-16).
#
# 탐지기: scripts/live_eth_iceberg_20261016.py (depthdiff 파일·hot 테이프 읽기만, 주문·거래소 호출 없음) → data/live/iceberg/
# 대시보드 풋프린트 «아이스버그 · 검증 중 n/150»(/api/iceberg)이 읽는다. 둘이 돌면 같은 사건을 두 번 적을 수 있어 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_iceberg_20261016.py"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸려 켜지지 않았다(10-14).
if pgrep -f "[-]u .*/${RUNNER}" >/dev/null; then
  echo "[$(date -Iseconds)] 아이스버그 탐지기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_iceberg_20261016.py" \
  "$ROOT/data/live/.supervisor_eth_iceberg.lock" \
  "$ROOT/logs/supervisor/eth_iceberg" \
  "$PY" -u "$ROOT/$RUNNER"
