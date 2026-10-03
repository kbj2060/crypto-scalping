#!/usr/bin/env bash
# Bybit 청산(allLiquidation) 수집기의 크래시 재기동 래퍼 (2026-10-03).
# 수집기: scripts/live_bybit_liquidation_collector_20261003.py (읽기 전용 시장 데이터, 주문 없음)
# 🔴청산 스트림은 소급 복원이 안 된다. 같은 수집기가 둘이면 같은 행이 두 번 들어간다 -- 이미 돌면 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
RUNNER="scripts/live_bybit_liquidation_collector_20261003.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] Bybit 청산 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_bybit_liquidation_collector_20261003.py" \
  "$ROOT/data/live/.supervisor_bybit_liq.lock" \
  "$ROOT/logs/supervisor/bybit_liq" \
  "$PY" -u "$ROOT/$RUNNER"
