#!/usr/bin/env bash
# 하이퍼리퀴드 최우선 호가 + 자산 컨텍스트 수집기의 크래시 재기동 래퍼 (2026-09-23).
#
# 수집기: scripts/live_hyperliquid_book_ticker_collector_20260923.py (읽기 전용 시장 데이터, 주문 없음)
# 🔴체결 수집기(live_hyperliquid_trade_collector_20260916.py)와 **다른 프로세스**다 -- 그쪽은
# 자기 duckdb 를 쓰고 duckdb 는 writer 가 하나다. 호가는 소급 재구성이 불가능하다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 대상 수집기가
# 둘이면 서로의 출력을 망친다(duckdb 는 writer 가 하나고, .bt 는 둘이 append 하면 순서가 섞인다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_hyperliquid_book_ticker_collector_20260923.py"
TARGET="${HL_BT_COIN:-ETH}"
for pid in $(pgrep -f "[l]${RUNNER#scripts/l}"); do
  cur=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^HL_BT_COIN=' | cut -d= -f2)
  if [ "${cur:-ETH}" = "$TARGET" ]; then
    echo "[$(date -Iseconds)] 하이퍼리퀴드 호가 수집기($TARGET)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HL_BT_COIN="$TARGET"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_hyperliquid_book_ticker_collector_20260923.py($TARGET)" \
  "$ROOT/data/live/.supervisor_hyperliquid_book_ticker_$TARGET.lock" \
  "$ROOT/logs/supervisor/hyperliquid_book_ticker_$TARGET" \
  "$PY" -u "$ROOT/$RUNNER"
