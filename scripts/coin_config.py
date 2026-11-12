#!/usr/bin/env python3
"""Central per-coin config for the Snapshot tab's dashboard-server-computed signal modules.

Single source of truth for values that used to be independently hardcoded per-script (e.g.
SYMBOL = "ETHUSDT" duplicated across live_liquidation_direction_signal_20260825.py and
live_liquidation_5m_signal_20260825.py) -- see
docs/eth_dashboard_multicoin_expansion_design_20260831.md section 6.1 for the audit that found
this duplication. New coins/fields get added here, not re-hardcoded in each script.

binance_symbol duplicates dashboard/server.py's MARKET_SYMBOLS (that dict serves the older "라이브"
tab chart/PnL code path, which this file doesn't touch) -- kept separate rather than unified today
to avoid a server.py <-> scripts/ circular import; see the design doc for the planned consolidation.

Does NOT cover live_spot_perp_basis_signal_20260827.compute_basis_liquidation_signal(symbol=...) or
live_liquidation_map_20260824.py's compute_spliced_levels()/compute_spliced_heatmap_history() --
those already take symbol/OHLCV data as a plain argument (a live Binance fetch, not a local duckdb
read) and only need binance_symbol from here, not a duckdb path.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COIN_CONFIG: dict[str, dict] = {
    "eth": {
        "binance_symbol": "ETHUSDT",
        # ETH 대시보드는 봇 state(dashboard_state.json)를 그대로 쓰므로 이 경로를 타지 않지만,
        # 코인 간 대칭을 위해 채워 둔다(다른 코인과 같은 코드 경로로 검증할 수 있다).
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure.duckdb",
        "microstructure_table": "microstructure_1m",
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk.duckdb",
        "tail_risk_table": "tail_risk_1m",
    },
    "btc": {
        "binance_symbol": "BTCUSDT",
        # 2026-09-03: BTC/SOL microstructure는 **메인 microstructure.duckdb 안의 접미사 테이블**로
        # 이미 수집되고 있었다(XRP/HYPE처럼 별도 파일이 아니라서 처음 찾을 때 놓쳤다).
        # 실측 최신 2026-09-03 06:18, 최근 24시간 whale 커버리지 93.5%(XRP 49.7%보다 좋다).
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure.duckdb",
        "microstructure_table": "microstructure_1m_btc",
        # Separate FILE, not a same-file suffixed table -- the BTC/SOL tail-risk worker writes here
        # to avoid a duckdb concurrent-writer conflict with ETH's dedicated writer in
        # tail_risk_interceptor.py (2026-08-17 incident, confirmed via
        # scripts/ops/supervisor_tail_risk_btc_sol_worker.sh's QUANT_TAIL_DB_PATH env var and
        # tail_risk_interceptor.py's self._table symbol-branch).
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_btc_sol.duckdb",
        "tail_risk_table": "tail_risk_1m_btc",
    },
    "sol": {
        "binance_symbol": "SOLUSDT",
        # BTC와 같은 파일의 다른 접미사 테이블. 최근 24시간 whale 90.9% / retail 94.7%.
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure.duckdb",
        "microstructure_table": "microstructure_1m_sol",
        # SAME file+worker as btc (scripts/ops/supervisor_tail_risk_btc_sol_worker.sh's
        # BOT_SYMBOLS="BTCUSDT,SOLUSDT"), separate table -- confirmed live 2026-08-31: table has
        # 19769+ rows (longer history than xrp's, same worker start as btc's), comfortably past
        # TRAIL_WIN's 2-day warmup.
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_btc_sol.duckdb",
        "tail_risk_table": "tail_risk_1m_sol",
    },
    "xrp": {
        "binance_symbol": "XRPUSDT",
        # 2026-09-03: XRP는 전용 워커가 microstructure까지 모은다(supervisor_xrp_worker.sh의
        # "microstructure + tail-risk + OI/long-short-ratio, all three"). nif_whale/nif_retail이
        # 여기 있으므로 대시보드의 수급흐름/리테일수급을 XRP에서도 **실제 XRP 값**으로 띄울 수 있다.
        # (그 전까지는 봇 state(ETH)만 읽어서 XRP 탭에도 ETH 값이 나왔다.)
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_xrp.duckdb",
        "microstructure_table": "microstructure_1m_xrp",
        # XRP gets its own fully dedicated worker + file (scripts/ops/supervisor_xrp_worker.sh,
        # QUANT_TAIL_DB_PATH=tail_risk_xrp.duckdb) -- unlike BTC/SOL it isn't sharing a combined
        # worker/file, so no single-writer contention concern here. Confirmed live 2026-08-31:
        # table has 5397+ rows spanning since 2026-08-27, comfortably past TRAIL_WIN's 2-day warmup.
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_xrp.duckdb",
        "tail_risk_table": "tail_risk_1m_xrp",
    },
    "hype": {
        "binance_symbol": "HYPEUSDT",
        # HYPE도 전용 워커가 microstructure를 모은다(supervisor_hype_worker.sh) -- XRP와 같은 모양.
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_hype.duckdb",
        "microstructure_table": "microstructure_1m_hype",
        # Own dedicated worker + file (scripts/ops/supervisor_hype_worker.sh), same shape as xrp's.
        # Confirmed live 2026-08-31: table has 4549+ rows spanning since 2026-08-28, past TRAIL_WIN's
        # 2-day warmup. NOTE: HYPEUSDT has no Binance SPOT listing (only the perp exists) -- the
        # 베이시스청산압박 signal (live_spot_perp_basis_signal_20260827.py) needs both legs and
        # degrades to warmed_up=False/error="no_spot_market" for this symbol, permanently, by design
        # (see that module's NO_SPOT_MARKET_SYMBOLS). Liquidation map/direction/5m-signal are
        # unaffected -- they only need the perp.
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_hype.duckdb",
        "tail_risk_table": "tail_risk_1m_hype",
    },
    "sndk": {
        "binance_symbol": "SNDKUSDT",
        "equity": True,   # 2026-11-13 주식류 = CBOE 지연 옵션 체인 자동 수집·옵션 카드 세 칸(scripts/live_cboe_option_chain_collector_20261101.py)
        # 2026-11-01 Binance TradFi 주식 선물. 전용 워커가 없어 아래 파일은 없다(모듈이 «파일 없음 = 빈 응답»으로 처리) --
        # 풋프린트·1초 수급·OI 는 테이프/ctx 수집기 데이터로 나오고, 신호 모듈들은 비어 있다. 현물도 없다(NO_SPOT_MARKET_SYMBOLS 와 같은 사정).
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_sndk.duckdb",
        "microstructure_table": "microstructure_1m_sndk",
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_sndk.duckdb",
        "tail_risk_table": "tail_risk_1m_sndk",
    },
    "soxl": {
        "binance_symbol": "SOXLUSDT",
        "equity": True,   # 2026-11-13 주식류 = CBOE 지연 옵션 체인 자동 수집·옵션 카드 세 칸(scripts/live_cboe_option_chain_collector_20261101.py)
        # 2026-11-02 Binance TradFi ETF 선물(SOXL, 반도체 3배). sndk 와 같은 사정 -- 전용 워커 없음·현물 없음.
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_soxl.duckdb",
        "microstructure_table": "microstructure_1m_soxl",
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_soxl.duckdb",
        "tail_risk_table": "tail_risk_1m_soxl",
    },
    # 2026-11-13 사용자 «꾸준히 거래량 많은 주식 선물 2개 추가» -- 60일 평일 거래대금 중앙 SKHYNIX $11.6억(하위10% 5.3억) · MU $6.1억(하위10% 3.6억). soxl 과 같은 사정(전용 워커 없음·현물 없음).
    "skhynix": {
        "binance_symbol": "SKHYNIXUSDT",   # KR_EQUITY(SK하이닉스) -- 미국 상장 옵션이 없어 equity(CBOE 옵션) 아님
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_skhynix.duckdb",
        "microstructure_table": "microstructure_1m_skhynix",
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_skhynix.duckdb",
        "tail_risk_table": "tail_risk_1m_skhynix",
    },
    "mu": {
        "binance_symbol": "MUUSDT",   # 마이크론
        "equity": True,   # CBOE 옵션 체인 자동 수집·옵션 카드
        "microstructure_db_path": ROOT / "data" / "live" / "microstructure_mu.duckdb",
        "microstructure_table": "microstructure_1m_mu",
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_mu.duckdb",
        "tail_risk_table": "tail_risk_1m_mu",
    },
}
