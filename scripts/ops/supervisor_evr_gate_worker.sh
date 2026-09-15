#!/usr/bin/env bash
# E|r| 게이트 워커의 크래시 재기동 래퍼 (2026-09-15).
#
# 워커: scripts/live_evr_gate_worker_20260915.py (채점만, **주문 없음**)
# supervisor_vol_forecast_worker.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 워커가 이미 돌고
# 있는데 이걸 켜면 같은 상태파일을 둘이 쓴다(원자적 rename 이라 깨지진 않지만 헛돈다).
#
# 🔴이 워커는 **20자산 × HGB 회귀 1개**를 돈다(사이클 실측 ~2초). 방향 분류기는 **호출하지
# 않는다** -- 실계좌 72왕복에서 적중 47.2% 였다(호메로스 §5.36-R).
# 🔴하루 한 번 `data.binance.vision` 일별 파일로 패널을 D-1 까지 메운다(라이브 metrics API 는
# 최근 41.7시간만 준다). 그 갱신은 20자산 × 하루치라 수 초면 끝난다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_evr_gate_worker_20260915.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] E|r| 게이트 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_evr_gate_worker_20260915.py" \
  "$ROOT/data/live/.supervisor_evr_gate_worker.lock" \
  "$ROOT/logs/supervisor/evr_gate_worker" \
  "$PY" -u "$ROOT/$RUNNER" --loop
