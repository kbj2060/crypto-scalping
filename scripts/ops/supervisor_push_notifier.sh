#!/usr/bin/env bash
# Crash-restart watchdog for the dashboard web-push notifier daemon (2026-09-04).
# Same _supervise.sh + crontab @reboot pattern as supervisor_liq_magnet_worker.sh -- see that
# file's header for why this isn't a systemd unit.
#
# 사용자 요청("다른 작업하다가 계속 신호를 놓친다")으로 추가. 이 데몬이 별도 프로세스인 이유는
# 대시보드 서버가 조회가 있을 때만 신호를 계산하기 때문이다 -- 아무도 안 보고 있으면, 정확히
# 알림이 필요한 그 상황에서, 트리거가 될 계산 자체가 돌지 않는다. 자세한 내용은
# scripts/live_push_notifier_20260904.py 모듈 docstring 참고.
#
# 로컬 대시보드(127.0.0.1:8787)를 폴링하므로 대시보드가 먼저 떠 있어야 한다. 떠 있지 않으면
# fetch_all()이 빈 dict로 degrade하고 다음 주기에 재시도한다 -- 크래시 루프가 되지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_push_notifier_20260904.py" \
  "$ROOT/data/live/.supervisor_push_notifier.lock" \
  "$ROOT/logs/supervisor/push_notifier" \
  "$PY" -u "$ROOT/scripts/live_push_notifier_20260904.py"
