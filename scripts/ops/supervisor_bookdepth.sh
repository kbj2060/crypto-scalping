#!/usr/bin/env bash
# ETH 호가 깊이 10열 워커의 크래시 재기동 래퍼 (2026-10-28).
#
# 워커: scripts/live_bookdepth_feat_20261028.py (depthdiff 파일 읽기만, 주문·거래소 호출 없음) → data/live/bookdepth/ethusdt.sqlite
# 방향 예측 «호가 깊이» 입력(bookDepth 와 같은 정의). 둘이 돌면 같은 봉을 두 번 쓰므로 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_bookdepth_feat_20261028.py"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸린다(supervisor_ob5m 과 같은 규칙).
if pgrep -f "[-]u .*/${RUNNER}" >/dev/null; then
  echo "[$(date -Iseconds)] 호가 깊이 워커가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_bookdepth_feat_20261028.py (ETHUSDT)" \
  "$ROOT/data/live/.supervisor_eth_bookdepth.lock" \
  "$ROOT/logs/supervisor/eth_bookdepth" \
  nice -n 10 "$PY" -u "$ROOT/$RUNNER"
