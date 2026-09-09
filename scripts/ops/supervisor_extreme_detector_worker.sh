#!/usr/bin/env bash
# 극점 탐지기 워커의 크래시 재기동 래퍼 (2026-09-10).
#
# 워커: scripts/live_eth_extreme_detector_worker_20260910.py (채점만, 주문 없음)
# supervisor_breakout_reversal_shadow.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 워커가 이미 돌고
# 있는데 이걸 켜면 같은 상태파일을 둘이 쓴다(원자적 rename 이라 깨지진 않지만 헛돈다).
#
# ⚠️ TabPFN 아티팩트를 쓰면 사이클이 ~5초다(HGB 0.5초). 대시보드 응답 경로 밖이라
# 문제가 안 되지만, GPU 는 V자 TabPFN·증거신호와 공유한다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_extreme_detector_worker_20260910.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 극점 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_extreme_detector_worker_20260910.py" \
  "$ROOT/data/live/.supervisor_extreme_detector_worker.lock" \
  "$ROOT/logs/supervisor/extreme_detector_worker" \
  "$PY" -u "$ROOT/$RUNNER" --loop
