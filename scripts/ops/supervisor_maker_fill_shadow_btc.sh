#!/usr/bin/env bash
# BTC 메이커 체결 시뮬레이션(peg vs static) 워커의 크래시 재기동 래퍼 (2026-09-04).
#
# ETH 전용이던 maker_fill 섀도우를 5코인으로 넓히는 작업(사용자 지시). 워커는 심볼·DB·정책이
# 전부 환경변수라 코드 변경 없이 코인만 갈아끼우면 된다(scripts/maker_fill_shadow_worker.py:42~63).
# WS 스트림도 SYMBOL에서 유도된다(<symbol>@bookTicker/<symbol>@trade).
#
# 결정시점 동기화 arm 사용 -- BTC는 microstructure.duckdb에 orderbook_decision_snapshots_btc가 있다.
#
# ⚠️ 중복 실행 방지: _supervise.sh의 flock은 supervisor끼리만 막는다. 같은 코인 인스턴스가
# 둘이면 같은 duckdb에 붙어 단일 writer 제약에 걸린다. 그래서 environ의 심볼로 직접 확인한다.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

for pid in $(pgrep -f "[m]aker_fill_shadow_worker.py"); do
  sym=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep '^MAKER_SHADOW_SYMBOL=' | cut -d= -f2)
  if [ "${sym:-ETHUSDT}" = "BTCUSDT" ]; then
    echo "[$(date -Iseconds)] BTC maker_fill이 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
    exit 1
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MAKER_SHADOW_SYMBOL="BTCUSDT"
export MAKER_SHADOW_SPACING_S="300"
export MAKER_SHADOW_TIMEOUT_S="120"
export MAKER_SHADOW_POLICIES="peg,static"
export MAKER_SHADOW_DB_PATH="$ROOT/data/live/maker_fill_shadow_btc.duckdb"
export MAKER_SHADOW_DECISION_TABLE="orderbook_decision_snapshots_btc"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "maker_fill_shadow_worker.py(BTC)" \
  "$ROOT/data/live/.supervisor_maker_fill_shadow_btc.lock" \
  "$ROOT/logs/supervisor/maker_fill_shadow_btc" \
  "$PY" -u "$ROOT/scripts/maker_fill_shadow_worker.py"
