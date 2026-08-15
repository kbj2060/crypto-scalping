"""Pre-adoption checks flagged (not run) by eth_yush_orderflow_strategy_absorption_study_20260815.md
for the avg-print-size z-score leftover candidate (`quote_volume / trades`, rolling z-score):

  1. spearmanr(avg_print_z, vol_z) -- is it just a repaint of existing volume, or genuinely new?
  2. spearmanr(avg_print_z, price) / spearmanr(avg_print_z, forward+trailing return) -- price-trend
     contamination check (feedback_raw_feature_price_trend_contamination memory: disqualifying
     threshold ~0.5-0.6).

Reuses load_frame/add_flow_features/compute_indicators unmodified from the source analysis script --
no new feature-engineering logic, this is purely a diagnostic on already-computed columns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_yush_orderflow_component_evidence_20260815 import (  # noqa: E402
    add_flow_features,
    load_frame,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402


def main() -> None:
    raw = load_frame()
    frame = compute_indicators(raw).reset_index(drop=True)
    f = add_flow_features(frame)
    f = f.dropna(subset=["avg_print_z", "vol_z", "close"]).reset_index(drop=True)

    print(f"rows usable (post rolling-window warmup): {len(f)}")

    # 1. redundancy vs volume itself
    rho_vol, p_vol = spearmanr(f["avg_print_z"], f["vol_z"])
    print(f"\n1. spearmanr(avg_print_z, vol_z)          = {rho_vol:+.4f}  (p={p_vol:.2e})")

    # 2. contamination vs raw price level (non-stationary, included since the memory rule asks for
    #    it literally, but (3)/(4) below are the more meaningful trend checks)
    rho_price, p_price = spearmanr(f["avg_print_z"], f["close"])
    print(f"2. spearmanr(avg_print_z, close)           = {rho_price:+.4f}  (p={p_price:.2e})")

    # 3. contamination vs trailing return (was price already trending INTO this bar?)
    trail_ret_1h = f["close"].pct_change(12)
    m = f["avg_print_z"].notna() & trail_ret_1h.notna()
    rho_trail, p_trail = spearmanr(f.loc[m, "avg_print_z"], trail_ret_1h[m])
    print(f"3. spearmanr(avg_print_z, trailing_1h_ret) = {rho_trail:+.4f}  (p={p_trail:.2e})")

    # 4. contamination vs forward return (does the feature just proxy "a big move is happening/about
    #    to happen", i.e. is it already baked into the label the way price-trend features have been
    #    before -- feedback_raw_feature_price_trend_contamination memory)
    fwd_ret_1h = f["close"].pct_change(12).shift(-12)
    m = f["avg_print_z"].notna() & fwd_ret_1h.notna()
    rho_fwd, p_fwd = spearmanr(f.loc[m, "avg_print_z"], fwd_ret_1h[m])
    print(f"4. spearmanr(avg_print_z, forward_1h_ret)  = {rho_fwd:+.4f}  (p={p_fwd:.2e})")

    rho_absret_fwd, p_absret_fwd = spearmanr(f.loc[m, "avg_print_z"], fwd_ret_1h[m].abs())
    print(f"5. spearmanr(avg_print_z, |forward_1h_ret|)= {rho_absret_fwd:+.4f}  (p={p_absret_fwd:.2e})")

    print("\n--- verdict inputs ---")
    print(f"|rho(avg_print_z, vol_z)|        = {abs(rho_vol):.4f}  (disqualify-as-redundant if >~0.6)")
    print(f"max(|rho| vs price/trend checks) = "
          f"{max(abs(rho_price), abs(rho_trail), abs(rho_fwd), abs(rho_absret_fwd)):.4f}  "
          f"(disqualify-as-contaminated if >~0.5-0.6)")


if __name__ == "__main__":
    main()
