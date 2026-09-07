#!/usr/bin/env bash
# MASHT 앵커 방향 섀도우 러너의 크래시 재기동 래퍼 (2026-09-07).
#
# 러너: scripts/live_eth_masht_anchor_shadow_runner_20260907.py (가상 원장만, 주문 없음)
# supervisor_v_rebound_econ_shadow.sh 와 같은 _supervise.sh 패턴이다.
#
# ⚠️ 중복 실행 방지: _supervise.sh의 flock은 supervisor끼리만 막는다. 기존 handoff/nohup 잡이
# 아직 돌고 있는데 이걸 켜면 같은 러너가 둘이 되어 **같은 상태파일
# (data/live/masht_anchor_shadow_state.json)에 동시에 쓴다**. 그래서 기동 전에 러너 프로세스를
# 직접 확인하고, 이미 있으면 켜지 않고 종료한다.
#
# ⚠️ 이 러너는 TabPFN 문맥(3,555 x 2,784)을 첫 앵커에서 한 번 주입하고 프로세스 수명 동안
# 재사용한다. 재시작하면 다음 앵커에서 다시 주입되며(수 초), GPU 를 대시보드와 공유하므로
# 잦은 재시작은 피하는 편이 좋다. 상태는 매 사이클 원자적으로 저장되고 시작 시 복원된다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_masht_anchor_shadow_runner_20260907.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] MASHT 앵커 섀도우: 러너가 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_masht_anchor_shadow_runner_20260907.py" \
  "$ROOT/data/live/.supervisor_masht_anchor_shadow.lock" \
  "$ROOT/logs/supervisor/masht_anchor_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --loop
