#!/usr/bin/env bash
# 예산 사다리 **섀도우 기록기**의 크래시 재기동 래퍼 (2026-09-14).
#
# 러너: scripts/live_eth_exit_ladder_shadow_20260914.py (JSONL 만 쓴다, **주문 없음**)
# supervisor_position_sizing_worker.sh 와 같은 _supervise.sh 패턴이다.
#
# 왜 도는가: 사다리(`exit_fraction_required`)는 지금 화면 표시 전용이다. 자동 집행으로 올릴지
# 판단하려면 «실제로 얼마나 자주·언제·얼마나 크게 발동하는가»를 먼저 알아야 한다.
# 포지션이 없으면 아무것도 안 적으므로, 사용자가 진입한 구간만 표본이 된다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 러너가 이미 돌고
# 있는데 이걸 켜면 같은 JSONL 에 둘이 append 해서 틱이 두 번씩 들어간다(빈도가 2배로 보인다).
#
# ⚠️ 죽일 때: `pgrep -f live_eth_exit_ladder_shadow` 는 **supervisor 도 같이 잡는다**
# (명령줄에 러너 경로가 인자로 들어 있다). supervisor 를 먼저 죽이지 않으면 러너만 죽였다가
# 15초 뒤 되살아난다. 2026-09-08 에 하루 세 번 밟은 함정이다:
#     pkill -f "_supervise.sh live_eth_exit_ladder_shadow"   # 먼저 supervisor
#     pkill -f "live_eth_exit_ladder_shadow_20260914.py"     # 그 다음 러너
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_eth_exit_ladder_shadow_20260914.py"
if pgrep -f "[${RUNNER:0:1}]${RUNNER:1}" >/dev/null 2>&1; then
  echo "[$(date -Iseconds)] 사다리 섀도우: 이미 실행 중 -- supervisor를 켜지 않는다(중복 방지)." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_eth_exit_ladder_shadow_20260914.py" \
  "$ROOT/data/live/.supervisor_exit_ladder_shadow.lock" \
  "$ROOT/logs/supervisor/exit_ladder_shadow" \
  "$PY" -u "$ROOT/$RUNNER"
