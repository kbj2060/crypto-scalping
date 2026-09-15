#!/usr/bin/env bash
# 체결 테이프 수집기의 크래시 재기동 래퍼 (2026-09-16).
#
# 수집기: scripts/live_trade_tape_collector_20260916.py (읽기 전용 시장 데이터, 주문 없음)
# 호가 수집기들과 성질이 다르다 -- 멈춘 구간은 data.binance.vision 일별 zip 으로 **소급
# 재구성이 가능하다**. 그래도 재기동은 붙인다: 조용히 멈춰 있으면 그 사실을 아무도 모른다
# (멈춘 구간은 수집기가 gaps 표에 스스로 적는다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 심볼 수집기가
# 둘이면 같은 duckdb 에 둘이 쓰려다 IOException 으로 하나가 죽는다(duckdb 는 단일 writer).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_trade_tape_collector_20260916.py"
SYM="${TAPE_SYMBOL:-ethusdt}"
for pid in $(pgrep -f "[l]ive_trade_tape_collector_20260916.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^TAPE_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ethusdt}" = "$SYM" ]; then
    echo "[$(date -Iseconds)] 체결 테이프 수집기($SYM)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TAPE_SYMBOL="$SYM"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_trade_tape_collector_20260916.py($SYM)" \
  "$ROOT/data/live/.supervisor_trade_tape_$SYM.lock" \
  "$ROOT/logs/supervisor/trade_tape_$SYM" \
  "$PY" -u "$ROOT/$RUNNER"
