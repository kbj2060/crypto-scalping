#!/usr/bin/env bash
# 돌파/되돌림 앵커 섀도우 러너의 크래시 재기동 래퍼 (2026-09-08).
#
# 러너: scripts/live_eth_breakout_reversal_shadow_runner_20260908.py (가상 원장만, 주문 없음)
# supervisor_masht_anchor_shadow.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 기존 handoff/nohup 잡이
# 아직 돌고 있는데 이걸 켜면 같은 러너가 둘이 되어 **같은 상태파일
# (data/live/breakout_reversal_shadow_state.json)에 동시에 쓴다**. 그래서 기동 전에 러너
# 프로세스를 직접 확인하고, 이미 있으면 켜지 않고 종료한다.
#
# ⚠️ MASHT 와 달리 GPU 를 쓰지 않는다(HGB 5시드, CPU 수십 ms). 대시보드와 경합하지 않으므로
# 재시작 비용이 낮다. 상태는 매 사이클 원자적으로 저장되고 시작 시 복원된다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_breakout_reversal_shadow_runner_20260908.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 돌파/되돌림 섀도우: 러너가 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_breakout_reversal_shadow_runner_20260908.py" \
  "$ROOT/data/live/.supervisor_breakout_reversal_shadow.lock" \
  "$ROOT/logs/supervisor/breakout_reversal_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --loop
