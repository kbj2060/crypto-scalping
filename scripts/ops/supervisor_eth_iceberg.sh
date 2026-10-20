#!/usr/bin/env bash
# ETH 칸별 아이스버그 실시간 탐지기의 크래시 재기동 래퍼 (2026-10-16).
#
# 탐지기: scripts/live_eth_iceberg_20261016.py (depthdiff 파일·hot 테이프 읽기만, 주문·거래소 호출 없음) → data/live/iceberg/
# 대시보드 풋프린트 «아이스버그 · 검증 중 n/150»(/api/iceberg)이 읽는다. 둘이 돌면 같은 사건을 두 번 적을 수 있어 켜지 않는다.
# 2026-10-20 코인 인자: supervisor_eth_iceberg.sh [ETHUSDT|XRPUSDT] (기본 ETHUSDT) -- 탐지기는 ICEBERG_SYMBOL 로 받는다 · 잠금·로그도 코인별.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_iceberg_20261016.py"
SYM="${1:-ETHUSDT}"
case "$SYM" in ETHUSDT|XRPUSDT) ;; *) echo "모르는 코인 $SYM" >&2; exit 2 ;; esac
COIN="$(echo "${SYM%USDT}" | tr '[:upper:]' '[:lower:]')"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸려 켜지지 않았다(10-14).
#   코인은 명령줄에 없고 환경변수라 /proc/<pid>/environ 으로 가른다(없으면 ETHUSDT -- 옛 실행).
for pid in $(pgrep -f "[-]u .*/${RUNNER}"); do
  run_sym="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^ICEBERG_SYMBOL=//p')"
  if [ "${run_sym:-ETHUSDT}" = "$SYM" ]; then
    echo "[$(date -Iseconds)] $SYM 아이스버그 탐지기가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ICEBERG_SYMBOL="$SYM"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_iceberg_20261016.py ($SYM)" \
  "$ROOT/data/live/.supervisor_${COIN}_iceberg.lock" \
  "$ROOT/logs/supervisor/${COIN}_iceberg" \
  "$PY" -u "$ROOT/$RUNNER"
