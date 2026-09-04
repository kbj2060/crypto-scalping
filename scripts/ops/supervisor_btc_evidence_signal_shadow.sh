#!/usr/bin/env bash
# BTC 증거신호 섀도우 러너의 크래시 재기동 래퍼 (2026-09-04).
#
# 왜 뒤늦게 생겼나: `handoff.sh launch`로 띄운 raw 잡이라 부팅 경로가 없었다. 2026-09-03
# 22:05 재부팅 때 조용히 사라졌고, 상태파일이 20:4x에서 멈춘 채 17시간 넘게 아무도 몰랐다.
# 그 사건이 이 파일들이 생긴 이유다.
#
# ⚠️ 중복 실행 방지: _supervise.sh의 flock은 supervisor끼리만 막는다. 기존 잡이 돌고 있는데
# 켜면 같은 러너가 둘이 되어 **같은 상태파일(data/live/btc_evidence_signal_shadow_state.json)에 동시에 쓴다**.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_btc_evidence_signal_shadow_runner_20260902.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1} --loop" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] BTC 증거신호 섀도우가 이미 실행 중 -- 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_btc_evidence_signal_shadow_runner_20260902.py" \
  "$ROOT/data/live/.supervisor_btc_evidence_signal_shadow.lock" \
  "$ROOT/logs/supervisor/btc_evidence_signal_shadow" \
  "$PY" -u "$ROOT/$RUNNER" --loop
