#!/usr/bin/env bash
# 하이퍼리퀴드 주소단위 체결 수집기의 크래시 재기동 래퍼 (2026-09-16).
#
# 수집기: scripts/live_hyperliquid_trade_collector_20260916.py (읽기 전용, 주문 없음)
# 🔴호가 래스터와 같은 성질이다 -- **소급 재구성이 불가능하다**. 노드 아카이브가 Requester-Pays
# 라 과거를 살 수 없으므로(실측 2026-09-16), 멈춘 시간은 영구 손실이다. 재기동이 필수다.
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 수집기가 둘이면
# 같은 duckdb 에 둘이 쓰려다 IOException 으로 하나가 죽는다(duckdb 는 단일 writer).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

if pgrep -f "[l]ive_hyperliquid_trade_collector_20260916.py" >/dev/null; then
  echo "[$(date -Iseconds)] 하이퍼리퀴드 체결 수집기가 이미 실행 중 -- 켜지 않는다." >&2
  exit 1
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HL_COINS="${HL_COINS:-ETH}"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_hyperliquid_trade_collector_20260916.py($HL_COINS)" \
  "$ROOT/data/live/.supervisor_hyperliquid_trades.lock" \
  "$ROOT/logs/supervisor/hyperliquid_trades" \
  "$PY" -u "$ROOT/scripts/live_hyperliquid_trade_collector_20260916.py"
