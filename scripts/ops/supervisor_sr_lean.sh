#!/usr/bin/env bash
# ETH «레벨 5분» 기울기(A19) 실시간 워커의 크래시 재기동 래퍼 (2026-11-18).
#
# 워커: scripts/live_sr_lean_20261118.py (@aggTrade 웹소켓 읽기만, 주문·계정 호출 없음) → data/live/sr_lean/ethusdt.jsonl
# 대시보드 /api/sr-lean(차트 표식·⑤ 예측 줄 «레벨 5분» 칸)이 읽는다. 둘이 돌면 같은 사건을 두 번 쓰므로 켜지 않는다.
# (재)시작 뒤 약 1시간은 체결 버퍼가 차는 동안 판정을 내지 않는다(화면 «준비 중»).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_sr_lean_20261118.py"
# 실제 실행 형태(-u …/RUNNER)만 센다 -- 이름만 쓰면 파일명을 담은 ssh·grep 셸에도 걸린다(supervisor_bookdepth 와 같은 규칙).
if pgrep -f "[-]u .*/${RUNNER}" >/dev/null; then
  echo "[$(date -Iseconds)] 레벨 5분 워커가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 라이브 서버라 한 코어로 묶는다(평상시 매초 사건 탐지 약 10ms)
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_sr_lean_20261118.py (ETHUSDT)" \
  "$ROOT/data/live/.supervisor_sr_lean.lock" \
  "$ROOT/logs/supervisor/sr_lean" \
  nice -n 10 "$PY" -u "$ROOT/$RUNNER"
