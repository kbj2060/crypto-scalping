#!/usr/bin/env bash
# 서버용 래퍼: scripts/live_deribit_block_trade_collector_20260928.py (Deribit ETH 옵션 체결·블록 거래).
# run_deribit_gex_collector_server.sh 처럼 서버 conda 경로를 박되, GEX 와 달리 **상주 WS 프로세스**라
# cron 한 번 실행이 아니라 _supervise.sh 로 크래시 재기동한다(supervisor_hyperliquid_trades.sh 관례).
# 멈춘 동안은 재기동 시 REST 백필이 메운다(24h 이내).
#
# ⚠️ 중복 실행 방지: _supervise.sh 의 flock 은 supervisor 끼리만 막는다. 수집기가 둘이면
# 같은 duckdb 에 둘이 쓰려다 하나가 죽는다(duckdb 는 단일 writer).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/llewyn/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"

if pid=$(pgrep -f "[l]ive_deribit_block_trade_collector_20260928.py"); then
  echo "[$(date -Iseconds)] Deribit 블록 거래 수집기가 이미 실행 중(pid $pid) -- 켜지 않는다." >&2
  exit 1
fi

exec "$ROOT/scripts/ops/_supervise.sh" \
  "live_deribit_block_trade_collector_20260928.py" \
  "$ROOT/data/live/.supervisor_deribit_block_trades.lock" \
  "$ROOT/logs/supervisor/deribit_block_trades" \
  "$PY" -u "$ROOT/scripts/live_deribit_block_trade_collector_20260928.py"
