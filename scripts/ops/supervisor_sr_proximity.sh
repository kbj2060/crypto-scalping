#!/usr/bin/env bash
# 지지·저항 «가까워짐» 텔레그램 워커 crash-restart 래퍼(2026-11-04) -- supervisor_push_notifier.sh 와 같은 _supervise.sh + crontab @reboot 패턴.
# 로컬 대시보드(127.0.0.1:8787/api/sr-watch)를 폴링하므로 대시보드가 먼저 떠 있어야 한다. 꺼져 있으면 다음 주기에 다시(크래시 루프 아님).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_sr_proximity_alert_20261104.py" \
  "$ROOT/data/live/.supervisor_sr_proximity.lock" \
  "$ROOT/logs/supervisor/sr_proximity" \
  "$PY" -u "$ROOT/scripts/live_sr_proximity_alert_20261104.py"
