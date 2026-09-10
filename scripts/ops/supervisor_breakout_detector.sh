#!/usr/bin/env bash
# 횡보→추세 전환 탐지 워커의 크래시 재기동 래퍼 (2026-09-11).
#
# 워커: scripts/live_eth_breakout_detector_worker_20260911.py (채점만, 주문 없음)
# supervisor_vol_forecast_worker.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 워커가 이미 돌고
# 있는데 이걸 켜면 같은 상태파일을 둘이 쓴다(원자적 rename 이라 안 깨지지만 헛돈다).
#
# ⚠️ pkill 주의: 이 스크립트 명령줄이 워커 경로를 인자로 담아 `pkill -f <워커>` 에 같이
# 잡힌다(2026-09-08 하루 3회 사고). 워커만 죽이려면 `grep -v _supervise.sh` 를 쓸 것.
#
# 이 신호는 5분봉이라 워커가 봉 마감 +20초에 정렬해 돈다. 백테스트가 **마감된 봉** 기준이므로
# 진행 중 봉을 쓰지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_breakout_detector_worker_20260911.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 전환 탐지 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_breakout_detector_worker_20260911.py" \
  "$ROOT/data/live/.supervisor_breakout_detector.lock" \
  "$ROOT/logs/supervisor/breakout_detector" \
  "$PY" -u "$ROOT/$RUNNER" --loop
