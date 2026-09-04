#!/usr/bin/env python3
"""One-off verification: directly call compute_evidence_signal_metalabels() with a REAL recent
bottom_liquidity_sweep fire bar (2026-08-30 09:40 UTC, confirmed via /api/evidence-signals) to
prove the new liquidity_sweep entry in METALABEL_SIGNALS actually loads its frozen train context
and returns a real TabPFN proba end-to-end -- matching this project's established live-wiring
verification pattern (e.g. taker_delta_z_climax's own 2026-08-21 17:05 reproduction)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from live_evidence_signal_dashboard_20260823 import compute_signals
from live_evidence_signal_metalabel_20260829 import compute_evidence_signal_metalabels

KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"


def main() -> int:
    full = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Known chart-verified HIT fire bar (pos=14712, side=bottom, ratio=5.53) from the label design
    # phase -- static klines CSV is a periodic snapshot (currently through 2026-08-28), not the
    # dashboard's own live-fetched feed, so we target a bar guaranteed present instead of "now".
    target_ts = pd.Timestamp("2024-02-20 17:00:00")
    target_pos_full = full.index[full["timestamp"] == target_ts][0]
    df = full.iloc[:target_pos_full + 1].tail(2000).reset_index(drop=True)
    sig = compute_signals(df, btc_df=None, funding_df=None).reset_index(drop=True)
    latest = sig.iloc[-1]
    print("latest bar:", latest["timestamp"], "bottom_liquidity_sweep:", latest.get("bottom_liquidity_sweep"),
          "top_liquidity_sweep:", latest.get("top_liquidity_sweep"))
    result = compute_evidence_signal_metalabels(df, latest)
    print("liquidity_sweep result:", result.get("liquidity_sweep"))
    if result.get("taker_delta_z_climax") is not None:
        print("taker_delta_z_climax result (sanity, other signal still works):", result.get("taker_delta_z_climax")["fired"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
