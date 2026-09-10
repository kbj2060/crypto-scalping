#!/usr/bin/env bash
# 크기 가늠자 워커의 크래시 재기동 래퍼 (2026-09-11).
#
# 워커: scripts/live_eth_position_sizing_worker_20260911.py (상태파일만 쓴다, 주문 없음)
# supervisor_vol_forecast_worker.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 워커가 이미 돌고
# 있는데 이걸 켜면 같은 상태파일을 둘이 쓴다.
#
# 이 워커는 klines 1,500봉 + 롤링 평균이라 사이클이 1초 미만이다. 그래도 워커로 뺀 이유는
# 2026-09-10 실장애 교훈이다 -- 요청 경로에서 계산하면 to_thread 풀이 고갈된다.
# 계수(k_q)는 eth_position_sizing_calib.json 에 한 번만 적합해 굳힌다. 매 주기 재적합하면
# 눈금이 흔들려 어제 값과 오늘 값을 비교할 수 없다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_position_sizing_worker_20260911.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 크기 가늠자 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_position_sizing_worker_20260911.py" \
  "$ROOT/data/live/.supervisor_position_sizing_worker.lock" \
  "$ROOT/logs/supervisor/position_sizing_worker" \
  "$PY" -u "$ROOT/$RUNNER" --loop
