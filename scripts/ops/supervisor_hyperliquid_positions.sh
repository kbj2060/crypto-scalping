#!/usr/bin/env bash
# 하이퍼리퀴드 고래 포지션 수집기의 크래시 재기동 래퍼 (2026-09-24).
# 수집기: scripts/live_hyperliquid_positions_collector_20260924.py (읽기 전용 공개 API, 주문 없음)
# 포지션 스냅샷은 소급이 불가능하다 -- 멈춘 구간은 영구 손실이라 재기동을 붙인다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_hyperliquid_positions_collector_20260924.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] 하이퍼리퀴드 포지션 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_hyperliquid_positions_collector_20260924.py" \
  "$ROOT/data/live/.supervisor_hyperliquid_positions.lock" \
  "$ROOT/logs/supervisor/hyperliquid_positions" \
  "$PY" -u "$ROOT/$RUNNER"
