#!/usr/bin/env bash
# 오더플로우 호가 래스터 수집기의 크래시 재기동 래퍼 (2026-09-14).
#
# 수집기: scripts/live_orderflow_raster_collector_20260914.py (읽기 전용 시장 데이터, 주문 없음)
# 히트맵의 과거는 재구성이 불가능하므로 이 프로세스가 멈춘 시간 = 영원히 비는 구간이다.
# 그래서 다른 수집기보다 재기동이 중요하다(멈춰도 대시보드는 회색 열로 정직하게 표시된다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 심볼 수집기가
# 둘이면 같은 .f32 파일의 같은 오프셋에 둘이 쓴다(깨지진 않지만 헛돈다).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_orderflow_raster_collector_20260914.py"
SYM="${OF_SYMBOL:-ethusdt}"
for pid in $(pgrep -f "[l]ive_orderflow_raster_collector_20260914.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^OF_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ethusdt}" = "$SYM" ]; then
    echo "[$(date -Iseconds)] 오더플로우 래스터($SYM)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OF_SYMBOL="$SYM"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_orderflow_raster_collector_20260914.py($SYM)" \
  "$ROOT/data/live/.supervisor_orderflow_raster_$SYM.lock" \
  "$ROOT/logs/supervisor/orderflow_raster_$SYM" \
  "$PY" -u "$ROOT/$RUNNER"
