#!/usr/bin/env bash
# Bybit 체결 테이프 + OI·마크·펀딩 수집기의 크래시 재기동 래퍼 (2026-10-06).
# 수집기: scripts/live_bybit_trade_tape_collector_20261006.py (읽기 전용 시장 데이터, 주문 없음, 5코인 한 프로세스)
# 🔴같은 수집기가 둘이면 같은 초를 두 번 쓴다(INSERT OR REPLACE 라 덮이지만 맥락 표는 행이 겹친다) -- 이미 돌면 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
RUNNER="scripts/live_bybit_trade_tape_collector_20261006.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] Bybit 테이프 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_bybit_trade_tape_collector_20261006.py" \
  "$ROOT/data/live/.supervisor_bybit_tape.lock" \
  "$ROOT/logs/supervisor/bybit_tape" \
  "$PY" -u "$ROOT/$RUNNER"
