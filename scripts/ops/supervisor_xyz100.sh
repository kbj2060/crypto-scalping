#!/usr/bin/env bash
# 나스닥100 무기한(Hyperliquid xyz:XYZ100) 1분봉·초당 가격 수집기의 크래시 재기동 래퍼 (2026-10-14).
#
# 수집기: scripts/live_xyz100_collector_20261014.py (읽기 전용 시장 데이터, 주문 없음) → data/hot/hl_xyz.sqlite
# 대시보드 «⑥ 나스닥 · 금리 · 1시간»(/api/xyz100)이 이 파일을 읽는다. 둘이 돌면 같은 행을 두 번 덮어쓸 뿐이지만 켜지 않는다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_xyz100_collector_20261014.py"
if pgrep -f "[l]${RUNNER#scripts/l}" >/dev/null; then
  echo "[$(date -Iseconds)] XYZ100 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_xyz100_collector_20261014.py" \
  "$ROOT/data/live/.supervisor_xyz100.lock" \
  "$ROOT/logs/supervisor/xyz100" \
  "$PY" -u "$ROOT/$RUNNER"
