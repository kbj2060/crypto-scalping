#!/usr/bin/env bash
# OKX 컨텍스트(OI·마크·펀딩·청산) 수집기의 크래시 재기동 래퍼 (2026-09-23).
#
# 수집기: scripts/live_okx_context_collector_20260923.py (읽기 전용 시장 데이터, 주문 없음)
# 🔴OI·청산 스트림은 소급 복원이 안 된다(REST 는 최근 며칠뿐이고 청산은 아예 없다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 대상 수집기가
# 둘이면 서로의 출력을 망친다(duckdb 는 writer 가 하나고, .bt 는 둘이 append 하면 순서가 섞인다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_okx_context_collector_20260923.py"
TARGET="${OKX_CTX_INST:-ETH-USDT-SWAP}"
for pid in $(pgrep -f "[l]${RUNNER#scripts/l}"); do
  cur=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^OKX_CTX_INST=' | cut -d= -f2)
  if [ "${cur:-ETH-USDT-SWAP}" = "$TARGET" ]; then
    echo "[$(date -Iseconds)] OKX 컨텍스트 수집기($TARGET)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OKX_CTX_INST="$TARGET"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_okx_context_collector_20260923.py($TARGET)" \
  "$ROOT/data/live/.supervisor_okx_context_$TARGET.lock" \
  "$ROOT/logs/supervisor/okx_context_$TARGET" \
  "$PY" -u "$ROOT/$RUNNER"
