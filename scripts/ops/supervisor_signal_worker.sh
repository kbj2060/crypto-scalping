#!/usr/bin/env bash
# 신호 워커 공용 래퍼 (2026-09-14). scripts/live_signal_worker.py 를 감싼다.
#
# 사용: supervisor_signal_worker.sh <이름> <module:function> <상태파일> <주기초>
#   예) supervisor_signal_worker.sh v_rebound \
#         live_eth_sweep_v_rebound_signal_20260829:compute_eth_sweep_v_rebound_signal \
#         data/live/eth_v_rebound_state.json 60
#
# supervisor_extreme_detector_worker.sh 와 같은 _supervise.sh 패턴이다. 다른 점은 워커마다
# 파일을 새로 만들지 않고 인자로 구분한다는 것뿐 -- 워커 본체가 공용본이라 래퍼도 공용본이다.
#
# ⚠️중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 상태파일을 둘이
#   쓰면 원자적 rename 이라 깨지진 않지만 헛돈다. 그래서 여기서 먼저 pgrep 로 본다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

NAME="${1:?이름이 필요합니다}"
COMPUTE="${2:?module:function 이 필요합니다}"
STATE="${3:?상태파일 경로가 필요합니다}"
INTERVAL="${4:-60}"

if pgrep -f "live_signal_worker.py --compute ${COMPUTE} " >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] ${NAME} 워커: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# GPU 를 안 쓰는 워커는 CUDA 컨텍스트를 열지 않게 막는다(2026-09-14).
# 이 서버는 RTX 3070 Ti 8GB 한 장인데 이미 파이썬 9개가 컨텍스트를 들고 VRAM 이 96%(7,849/8,192)다.
# 컨텍스트 하나가 수백 MB 를 상주로 먹으므로, torch 를 안 쓰는 워커까지 여는 건 순손해다.
# NO_GPU=1 을 주면 그 워커는 CPU 전용이 된다 -- 레짐 3종(joblib/sklearn)·거시 달력(requests)이 대상.
if [[ "${NO_GPU:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_signal_worker.py --compute ${COMPUTE}" \
  "$ROOT/data/live/.supervisor_signal_worker_${NAME}.lock" \
  "$ROOT/logs/supervisor/signal_worker_${NAME}" \
  "$PY" -u "$ROOT/scripts/live_signal_worker.py" \
    --compute "$COMPUTE" --state "$STATE" --interval "$INTERVAL" --loop
