#!/usr/bin/env bash
# OKX 체결 테이프 수집기의 크래시 재기동 래퍼 (2026-09-23).
#
# 수집기: scripts/live_okx_trade_tape_collector_20260923.py (읽기 전용 시장 데이터, 주문 없음)
# 바이낸스 테이프와 성질이 같다 -- 멈춘 구간은 OKX 일별 덤프로 소급 재구성이 가능하다
# (okx.com/cdn/okex/traderecords, 🔴파일 경계가 UTC+8 이다). 그래도 재기동은 붙인다:
# 조용히 멈춰 있으면 그 사실을 아무도 모른다(멈춘 구간은 수집기가 gaps 표에 스스로 적는다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 대상 수집기가
# 둘이면 서로의 출력을 망친다(duckdb 는 writer 가 하나고, .bt 는 둘이 append 하면 순서가 섞인다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_okx_trade_tape_collector_20260923.py"
TARGET="${OKX_TAPE_INST:-ETH-USDT-SWAP}"
for pid in $(pgrep -f "[l]${RUNNER#scripts/l}"); do
  cur=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^OKX_TAPE_INST=' | cut -d= -f2)
  if [ "${cur:-ETH-USDT-SWAP}" = "$TARGET" ]; then
    echo "[$(date -Iseconds)] OKX 체결 테이프 수집기($TARGET)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OKX_TAPE_INST="$TARGET"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_okx_trade_tape_collector_20260923.py($TARGET)" \
  "$ROOT/data/live/.supervisor_okx_trade_tape_$TARGET.lock" \
  "$ROOT/logs/supervisor/okx_trade_tape_$TARGET" \
  "$PY" -u "$ROOT/$RUNNER"
