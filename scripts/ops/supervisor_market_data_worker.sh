#!/usr/bin/env bash
# 2026-10-12 trading_bot.py(Omega4.6.1 섀도우, trading-bot.service) 은퇴 -- 봇이 품고 있던 수집만 넘겨받는 워커.
#   ETH·BTC·SOL 미시구조(microstructure.duckdb: microstructure_1m / _btc / _sol)
#   ETH 청산 요격기(tail_risk.duckdb: tail_risk_1m · liq_burst_state.json -- 대시보드 청산 게이지·급증 배지)
#   대시보드 최소 상태(dashboard_state.json: session 세 칸 + updated_at -- 화면 render() 갱신 신호)
# 봇과 같은 파일·같은 테이블에 쓴다(단일 writer). 🔴봇이 돌고 있으면 띄우지 말 것 -- duckdb 쓰기 락을 다툰다.
# 봇이 쓰던 orderbook_decision_snapshots(ETH·SOL 결정 시점 호가)는 넘기지 않는다 -- 결정 루프에 묶인
#   연구용 표였고 호가는 depth_diff·book_ticker 수집기가 따로 쌓는다.
# 같은 _supervise.sh + crontab @reboot 패턴(supervisor_hype_worker.sh 참고).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-$HOME/miniconda3/envs/quant_ai/bin/python}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

export BOT_SYMBOLS="ETHUSDT,BTCUSDT,SOLUSDT"
export COLLECT_MICROSTRUCTURE="true"
export COLLECT_TAIL_RISK="true"
export TAIL_SYMBOLS="ETHUSDT"
export COLLECT_OI_LSRATIO="false"
export WRITE_DASHBOARD_STATE="true"

exec "$ROOT/scripts/ops/_supervise.sh" \
  "duckdb_persist_worker.py(ETH/BTC/SOL microstructure + ETH tail-risk + dashboard_state)" \
  "$ROOT/data/live/.supervisor_market_data_worker.lock" \
  "$ROOT/logs/supervisor/market_data_worker" \
  "$PY" -u "$ROOT/scripts/duckdb_persist_worker.py"
