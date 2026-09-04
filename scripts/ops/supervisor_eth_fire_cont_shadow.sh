#!/usr/bin/env bash
# 증거신호 발동지속 섀도우 러너의 크래시 재기동 래퍼 (2026-09-04).
#
# 왜 만들었나: 이 러너는 그동안 `handoff.sh launch`로 띄운 raw 잡이었다. 그러면 두 가지가
# 없다 -- (1) 죽어도 아무것도 되살리지 않고 (2) 재부팅을 못 넘긴다. 실제로 BTC/XRP 증거신호
# 섀도우가 2026-09-03 재부팅(22:05) 때 그렇게 조용히 사라졌고, 아무도 몰랐다.
# supervisor_liq_magnet_worker.sh / supervisor_push_notifier.sh와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh의 flock은 supervisor끼리만 막는다. 기존 handoff 잡이
# 아직 돌고 있는데 이걸 켜면 같은 러너가 둘이 되어 **같은 상태파일(data/live/fire_cont_shadow_state.json)에 동시에 쓴다**.
# 그래서 기동 전에 러너 프로세스를 직접 확인하고, 이미 있으면 켜지 않고 종료한다.
#
# 상태는 매 사이클 원자적으로 저장되고 시작 시 복원되므로 재시작 자체는 안전하다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_fire_cont_shadow_runner_20260904.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 증거신호 발동지속: 러너가 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_fire_cont_shadow_runner_20260904.py" \
  "$ROOT/data/live/.supervisor_eth_fire_cont_shadow.lock" \
  "$ROOT/logs/supervisor/eth_fire_cont_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --loop
