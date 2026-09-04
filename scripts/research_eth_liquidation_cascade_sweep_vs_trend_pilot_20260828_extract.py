#!/usr/bin/env python3
"""Run ON THE SERVER (has the live duckdb files). Extracts fixed-window slices needed for
docs/experiments/eth_liquidation_cascade_sweep_vs_trend_pilot_design_20260828.md into compact
CSVs under data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828/, so the pilot can
be pulled once (point-in-time frozen snapshot -- avoids the load_hourly() re-fetch drift documented
in feedback_liquidation_map_load_hourly_data_drift_nondeterminism) and analyzed entirely on dev.

Window: 2026-07-18 12:00 UTC -> now (extended 2026-08-28, sample-size follow-up). tail_risk_1m's
valid epoch starts 2026-07-18 15:03 UTC (forceOrder WS fix, see
eth_liquidation_feed_epoch_defect_20260817 memory) -- everything before that is fake zeros, so this
is the earliest honest start, not an arbitrary "more data" choice. orderbook_decision_snapshots
(from 05-13) and microstructure_1m (from 05-03) both already cover this whole range. oi_lsratio_5m
only starts 2026-08-22 (collector's actual deploy date) -- extracted as-is, shorter by construction,
not a bug.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_KST = "2026-07-18 21:00:00+09:00"  # = 2026-07-18 12:00 UTC


def dump(db_rel: str, sql: str, out_name: str) -> None:
    path = ROOT / db_rel
    con = duckdb.connect(str(path), read_only=True)
    try:
        df = con.execute(sql).df()
    finally:
        con.close()
    out_path = OUT_DIR / out_name
    df.to_csv(out_path, index=False)
    print(f"{out_name}: {len(df)} rows -> {out_path}")


dump(
    "data/live/tail_risk.duckdb",
    f"""
    SELECT ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale
    FROM tail_risk_1m
    WHERE ts >= TIMESTAMPTZ '{START_KST}'
    ORDER BY ts
    """,
    "tail_risk_1m.csv",
)

dump(
    "data/live/oi_lsratio.duckdb",
    f"""
    SELECT ts, symbol, sum_open_interest, sum_open_interest_value,
           global_ls_ratio, global_ls_long_account, global_ls_short_account,
           top_pos_ls_ratio, top_pos_ls_long_account, top_pos_ls_short_account, sources_ok
    FROM oi_lsratio_5m
    WHERE ts >= TIMESTAMPTZ '{START_KST}' AND symbol = 'ETHUSDT'
    ORDER BY ts
    """,
    "oi_lsratio_5m.csv",
)

dump(
    "data/live/microstructure.duckdb",
    f"""
    SELECT recorded_at_kst, symbol, best_bid, best_ask, mid, spread_bps,
           bid_qty_1, ask_qty_1, bid_notional_1, ask_notional_1, imbalance_1,
           bid_qty_5, ask_qty_5, bid_notional_5, ask_notional_5, imbalance_5,
           bid_qty_10, ask_qty_10, bid_notional_10, ask_notional_10, imbalance_10,
           bid_qty_20, ask_qty_20, bid_notional_20, ask_notional_20, imbalance_20
    FROM orderbook_decision_snapshots
    WHERE recorded_at_kst >= TIMESTAMPTZ '{START_KST}' AND symbol = 'ETH/USDT:USDT'
    ORDER BY recorded_at_kst
    """,
    "orderbook_decision_snapshots.csv",
)

dump(
    "data/live/microstructure.duckdb",
    f"""
    SELECT ts, obi, taker_buy_ratio, oi_delta_pct, shadow_queue_collapse,
           shadow_absorption_score, shadow_queue_bias, shadow_toxicity_score,
           shadow_toxicity_regime, shadow_regime_tag, shadow_regime_conf,
           nif_whale, nif_retail, eai, funding_rate, spoofing_score,
           whale_position_score, valid_taker_flow, valid_nif, mark_price,
           recent_whale_count_5m, recent_trade_count_5m, recent_trade_notional_5m
    FROM microstructure_1m
    WHERE ts >= TIMESTAMPTZ '{START_KST}'
    ORDER BY ts
    """,
    "microstructure_1m.csv",
)

dump(
    "data/live/l2_anomaly_snapshots.duckdb",
    """
    SELECT event_id, symbol, triggered_at_kst, liq_burst_usd_60s, liq_z,
           price_move_pct_60s, price_z, liquidity_notional_usd, liquidity_z,
           liquidity_withdrawal_matched
    FROM l2_anomaly_events
    ORDER BY triggered_at_kst
    """,
    "l2_anomaly_events.csv",
)

dump(
    "data/live/l2_anomaly_snapshots.duckdb",
    """
    SELECT event_id, symbol, phase, ts_ms, side, qty_usd, price
    FROM l2_anomaly_liquidations
    ORDER BY event_id, ts_ms
    """,
    "l2_anomaly_liquidations.csv",
)

print("done")
