#!/usr/bin/env bash
# 최우선 호가(bookTicker) 수집기의 크래시 재기동 래퍼 (2026-09-14).
#
# 수집기: scripts/live_book_ticker_collector_20260914.py (읽기 전용 시장 데이터, 주문 없음)
# 래스터와 같은 성질이다 -- 멈춘 시간 = 영원히 비는 구간. 소급 재구성이 불가능하다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 심볼 수집기가
# 둘이면 같은 .bt 파일에 둘이 append 해서 행 순서가 섞인다(u 검사가 각자라 못 걸러낸다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_book_ticker_collector_20260914.py"
SYM="${BT_SYMBOL:-ethusdt}"
for pid in $(pgrep -f "[l]ive_book_ticker_collector_20260914.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^BT_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ethusdt}" = "$SYM" ]; then
    echo "[$(date -Iseconds)] bookTicker 수집기($SYM)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BT_SYMBOL="$SYM"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_book_ticker_collector_20260914.py($SYM)" \
  "$ROOT/data/live/.supervisor_book_ticker_$SYM.lock" \
  "$ROOT/logs/supervisor/book_ticker_$SYM" \
  "$PY" -u "$ROOT/$RUNNER"
