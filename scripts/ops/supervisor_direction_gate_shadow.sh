#!/usr/bin/env bash
# 방향 게이트 섀도우의 크래시 재기동 래퍼 (2026-09-16).
#
# 러너: scripts/live_direction_gate_shadow_20260915.py (기록만, **주문 없음**)
# supervisor_evr_gate_worker.sh 와 같은 _supervise.sh 패턴이다.
#
# 🔴왜 이걸 켜는가: 09-15 18:13 에 동결한 1순위 칸(`1d × E|r| 상위10% × 20자산`)의 미달 관문
# 셋 중 둘(건당 순손익 CI 하한 · 모델−롱 증분 CI)은 **앞으로 쌓이는 독립 관측으로만** 갈린다.
# 과거를 더 붙이는 길은 09-15 15:22 에 닫혔다(독립일 +48% 인데 t 2.58→1.71).
#
# 🔴패널 갱신은 이 러너가 하지 않는다 — **E|r| 게이트 워커가 같은 `data/binance_vision/panel`
# 을 UTC 하루 한 번 D−1 까지 메운다**(supervisor_evr_gate_worker.sh). 그 워커가 죽으면 이
# 러너의 정산도 같이 멈춘다(결정은 REST 로 계속 나지만 만기 봉이 패널에 안 들어온다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_direction_gate_shadow_20260915.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1}" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 방향 게이트 섀도우: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_direction_gate_shadow_20260915.py" \
  "$ROOT/data/live/.supervisor_direction_gate_shadow.lock" \
  "$ROOT/logs/supervisor/direction_gate_shadow" \
  "$PY" -u "$ROOT/$RUNNER"
