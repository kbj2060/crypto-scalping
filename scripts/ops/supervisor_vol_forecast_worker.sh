#!/usr/bin/env bash
# 변동성 전망 워커의 크래시 재기동 래퍼 (2026-09-10).
#
# 워커: scripts/live_eth_vol_forecast_worker_20260910.py (채점만, 주문 없음)
# supervisor_breakout_reversal_shadow.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 워커가 이미 돌고
# 있는데 이걸 켜면 같은 상태파일을 둘이 쓴다(원자적 rename 이라 깨지진 않지만 헛돈다).
#
# 이 워커는 로지스틱 회귀라 사이클이 ~0.9초다(GPU 안 쓴다). 그래도 워커로 뺀 이유는
# 2026-09-10 실장애 교훈이다 -- 아티팩트는 공유 자원이고 다른 세션이 언제든 무겁게 바꾼다.
# 신호가 **시간봉**이라 기본 주기가 300초다(5분봉 카드들과 다르다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_vol_forecast_worker_20260910.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 변동성 전망 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_vol_forecast_worker_20260910.py" \
  "$ROOT/data/live/.supervisor_vol_forecast_worker.lock" \
  "$ROOT/logs/supervisor/vol_forecast_worker" \
  "$PY" -u "$ROOT/$RUNNER" --loop
