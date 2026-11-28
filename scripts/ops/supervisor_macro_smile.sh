#!/usr/bin/env bash
# /api/macro-trend · /api/opt-smile 파일 워커 crash-restart 래퍼(2026-11-28) -- supervisor_sr_proximity.sh 와 같은 _supervise.sh + crontab @reboot 패턴.
# data/live/macro_trend.json(1시간) · opt_smile_{eth,btc,xrp}.json(60초)를 쓴다 -- rtserve(8790)가 읽기만 한다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_macro_smile_worker_20261128.py" \
  "$ROOT/data/live/.supervisor_macro_smile.lock" \
  "$ROOT/logs/supervisor/macro_smile" \
  "$PY" -u "$ROOT/scripts/live_macro_smile_worker_20261128.py"
