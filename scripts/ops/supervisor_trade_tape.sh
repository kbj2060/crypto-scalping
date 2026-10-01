#!/usr/bin/env bash
# 체결 테이프 수집기의 크래시 재기동 래퍼 (2026-09-16).
#
# 수집기: scripts/live_trade_tape_collector_20260916.py (읽기 전용 시장 데이터, 주문 없음)
# 호가 수집기들과 성질이 다르다 -- 멈춘 구간은 data.binance.vision 일별 zip 으로 **소급
# 재구성이 가능하다**. 그래도 재기동은 붙인다: 조용히 멈춰 있으면 그 사실을 아무도 모른다
# (멈춘 구간은 수집기가 gaps 표에 스스로 적는다).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 같은 심볼 수집기가
# 둘이면 같은 duckdb 에 둘이 쓰려다 IOException 으로 하나가 죽는다(duckdb 는 단일 writer).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

RUNNER="scripts/live_trade_tape_collector_20260916.py"
SYM="${TAPE_SYMBOL:-ethusdt}"
MKT="${TAPE_MARKET:-futures}"                      # 2026-10-01: spot = 현물 테이프(대시보드에서 옮김). 같은 심볼이라도 시장이 다르면 다른 수집기
KEY="$SYM"; [ "$MKT" = futures ] || KEY="${SYM}_$MKT"   # 선물은 옛 이름(락·로그 경로) 그대로
for pid in $(pgrep -f "[l]ive_trade_tape_collector_20260916.py"); do
  env_of=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null)
  sym=$(echo "$env_of" | grep '^TAPE_SYMBOL=' | cut -d= -f2)
  mkt=$(echo "$env_of" | grep '^TAPE_MARKET=' | cut -d= -f2)
  if [ "${sym:-ethusdt}" = "$SYM" ] && [ "${mkt:-futures}" = "$MKT" ]; then
    echo "[$(date -Iseconds)] 체결 테이프 수집기($KEY)가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TAPE_SYMBOL="$SYM" TAPE_MARKET="$MKT"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_trade_tape_collector_20260916.py($KEY)" \
  "$ROOT/data/live/.supervisor_trade_tape_$KEY.lock" \
  "$ROOT/logs/supervisor/trade_tape_$KEY" \
  "$PY" -u "$ROOT/$RUNNER"
