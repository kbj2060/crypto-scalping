#!/usr/bin/env bash
# 바이낸스 맥락(OI 1초·마크 1초·강제청산) 수집기의 크래시 재기동 래퍼 (2026-10-01 저장 재설계 4d).
#
# 수집기: scripts/live_binance_ctx_collector_20261001.py (읽기 전용 시장 데이터, 주문 없음) -> data/hot/binance_ctx.sqlite
# 🔴OI 1초·청산은 소급 복원이 안 된다(바이낸스는 1초 OI 이력을 안 주고 청산 REST 도 없다).
# 🔴OI REST 폴링(0.25초 × 종목)은 이 프로세스 **하나만** 한다 -- 둘이면 IP 한도 2,400/분을 넘는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_binance_ctx_collector_20261001.py"
if pid=$(pgrep -f "[l]${RUNNER#scripts/l}" | head -1) && [ -n "$pid" ]; then
  echo "[$(date -Iseconds)] 바이낸스 맥락 수집기가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_binance_ctx_collector_20261001.py" \
  "$ROOT/data/live/.supervisor_binance_ctx.lock" \
  "$ROOT/logs/supervisor/binance_ctx" \
  "$PY" -u "$ROOT/$RUNNER"
