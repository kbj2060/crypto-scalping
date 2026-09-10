#!/usr/bin/env bash
# 포지션 청산 감시자 워커의 크래시 재기동 래퍼 (2026-09-10).
#
# 워커: scripts/live_position_exit_advisor_20260910.py (판정만, 주문 없음, 5분봉 +90초 기동)
# supervisor_extreme_detector_worker.sh 와 같은 _supervise.sh 패턴이다.
# 바이낸스 키는 .env 에서 워커가 직접 읽는다(대시보드와 같은 키).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_position_exit_advisor_20260910.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 청산 감시자: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_position_exit_advisor_20260910.py" \
  "$ROOT/data/live/.supervisor_position_exit_advisor.lock" \
  "$ROOT/logs/supervisor/position_exit_advisor" \
  "$PY" -u "$ROOT/$RUNNER" --loop
