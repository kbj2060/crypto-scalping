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
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk.duckdb",
        "tail_risk_table": "tail_risk_1m",
    },
    "btc": {
        "binance_symbol": "BTCUSDT",
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
        # SAME file+worker as btc (scripts/ops/supervisor_tail_risk_btc_sol_worker.sh's
        # BOT_SYMBOLS="BTCUSDT,SOLUSDT"), separate table -- confirmed live 2026-08-31: table has
        # 19769+ rows (longer history than xrp's, same worker start as btc's), comfortably past
        # TRAIL_WIN's 2-day warmup.
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_btc_sol.duckdb",
        "tail_risk_table": "tail_risk_1m_sol",
    },
    "xrp": {
        "binance_symbol": "XRPUSDT",
        # XRP gets its own fully dedicated worker + file (scripts/ops/supervisor_xrp_worker.sh,
        # QUANT_TAIL_DB_PATH=tail_risk_xrp.duckdb) -- unlike BTC/SOL it isn't sharing a combined
        # worker/file, so no single-writer contention concern here. Confirmed live 2026-08-31:
        # table has 5397+ rows spanning since 2026-08-27, comfortably past TRAIL_WIN's 2-day warmup.
        "tail_risk_db_path": ROOT / "data" / "live" / "tail_risk_xrp.duckdb",
        "tail_risk_table": "tail_risk_1m_xrp",
    },
    "hype": {
        "binance_symbol": "HYPEUSDT",
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
}
