#!/usr/bin/env bash
# retail_shift 지속 신호(B2) 섀도우 러너 + 롱숏비 known_ts 수집기의 크래시 재기동 래퍼 (2026-09-05).
# supervisor_eth_fire_cont_shadow.sh 와 같은 _supervise.sh + crontab @reboot 패턴. 주문 없음.
#
# ⚠️ 중복 실행 방지: 같은 러너가 둘이면 같은 상태파일(data/live/retail_shift_b2_state.json)과
# 행 파일(retail_shift_b2_lsr_rows.jsonl)에 동시에 쓴다. 기동 전에 러너 프로세스를 확인하고 있으면 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_retail_shift_b2_shadow_runner_20260905.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] retail_shift B2: 러너가 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_retail_shift_b2_shadow_runner_20260905.py" \
  "$ROOT/data/live/.supervisor_eth_retail_shift_b2_shadow.lock" \
  "$ROOT/logs/supervisor/eth_retail_shift_b2_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --loop
