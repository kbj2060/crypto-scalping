#!/usr/bin/env bash
# SOL 지속 규칙 섀도우 러너의 크래시 재기동 래퍼 (2026-09-05).
#
# ETH판(supervisor_eth_fire_cont_shadow.sh)과 같은 _supervise.sh 패턴이다. 러너는 **같은 파일**을
# `--asset sol`로 부르며, 셀·GAP·ATR·비용·발동 규칙은 ETH와 완전히 동일하다(자유도 0).
# 다른 것은 심볼 · 동시보유 2 · 상태 파일뿐이다.
# 근거·사전등록: docs/experiments/eth_crossasset_continuation_shadow_prereg_20260905.md
#
# ⚠️중복 실행 방지: pgrep 패턴에 `--asset sol`를 포함한다. ETH판 패턴("<러너> --loop")은
# 이 프로세스의 커맨드라인("<러너> --asset sol --loop")과 겹치지 않으므로 서로를 오탐하지 않는다.
# 상태는 매 사이클 원자적으로 저장되고 시작 시 복원되므로 재시작 자체는 안전하다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_fire_cont_shadow_runner_20260904.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --asset sol --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] SOL 지속 규칙: 러너가 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_fire_cont_shadow_runner_20260904.py(SOL)" \
  "$ROOT/data/live/.supervisor_sol_fire_cont_shadow.lock" \
  "$ROOT/logs/supervisor/sol_fire_cont_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --asset sol --loop
