#!/usr/bin/env python3
"""Cross-family combination round: ICT geometry (sweep / SMT raw) x order-flow climax.

Pre-registered in docs/experiments/eth_ict2022_ob_smt_po3_component_evidence_20260824.md
(follow-up round section). Only 2 combos, same-bar AND, no grids. Parents measured in-frame
for same-scale comparison. delta_z reuses the original creative-signals formula verbatim
(delta = 2*taker_buy_base - volume, rolling 288 z, min_periods=288) for comparability with
the scorecard's 2.75x/3.51x reference numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815 import add_sweep  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from analyze_eth_ict2022_ob_smt_po3_component_evidence_20260824 import add_smt  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

ETH_PATH = ROOT / "data" / "eth_5m_1year.csv"
OUT_DIR = ROOT / "tmp" / "eth_ict_geometry_x_flow_climax_combo_20260824"


def components(f: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    if side == "bottom":
        flow = f["delta_z"] <= -2.0
        return {"P_sweep": f["sweep_low"], "P_smt_raw": f["smt_raw_bottom"], "P_flow_climax": flow,
                "C1_sweep_x_flow": f["sweep_low"] & flow, "C2_smt_x_flow": f["smt_raw_bottom"] & flow}
    flow = f["delta_z"] >= 2.0
    return {"P_sweep": f["sweep_high"], "P_smt_raw": f["smt_raw_top"], "P_flow_climax": flow,
            "C1_sweep_x_flow": f["sweep_high"] & flow, "C2_smt_x_flow": f["smt_raw_top"] & flow}


def run_side(f: pd.DataFrame, mask: np.ndarray, pivots: pd.DataFrame, side: str, window: str) -> pd.DataFrame:
    close = f["close"].to_numpy()
    all_pos = np.flatnonzero(mask)
    pivot_pos = f.index[f["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, sig in components(f, side).items():
        trigger_pos = np.flatnonzero(sig.fillna(False).to_numpy() & mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"window": window, "side": side, "signal": name, "horizon": k_name,
                         **stats, "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    raw = pd.read_csv(ETH_PATH, usecols=cols, parse_dates=["timestamp"])
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    f = compute_indicators(raw).reset_index(drop=True)
    f = add_sweep(f)
    f = add_smt(f)
    delta = 2.0 * f["taker_buy_base"] - f["volume"]
    f["delta_z"] = (delta - delta.rolling(288, min_periods=288).mean()) / \
        delta.rolling(288, min_periods=288).std().replace(0.0, np.nan)
    pivots = load_zigzag_pivots()

    ts = f["timestamp"]
    masks = {
        "POOLED": (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy(),
        "VAL": ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy(),
        "OOS": ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy(),
    }
    res = pd.concat([run_side(f, m, pivots, side, w) for w, m in masks.items() for side in ("bottom", "top")],
                    ignore_index=True)

    pd.set_option("display.width", 200)
    cols_show = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n########## {side.upper()} (POOLED, K12_1h) ##########")
        sub = res[(res["side"] == side) & (res["window"] == "POOLED") & (res["horizon"] == "K12_1h")]
        print(sub[cols_show].to_string(index=False))

    print("\n########## pre-registered verdict (K12_1h, pooled) ##########")
    for side in ("bottom", "top"):
        sub = res[(res["side"] == side) & (res["window"] == "POOLED") & (res["horizon"] == "K12_1h")]
        lifts = sub.set_index("signal")["lift"]
        ns = sub.set_index("signal")["n_triggers"]
        for combo, parents in (("C1_sweep_x_flow", ["P_sweep", "P_flow_climax"]),
                               ("C2_smt_x_flow", ["P_smt_raw", "P_flow_climax"])):
            best_parent = max(lifts[p] for p in parents)
            vo = res[(res["side"] == side) & (res["horizon"] == "K12_1h") & (res["signal"] == combo)]
            lv = vo[vo["window"] == "VAL"]["lift"].iloc[0]
            lo = vo[vo["window"] == "OOS"]["lift"].iloc[0]
            n = int(ns[combo])
            if n < 100:
                verdict = "INSUFFICIENT(n<100)"
            elif lifts[combo] > best_parent and lv > 1 and lo > 1:
                verdict = "PASS(beats best parent)"
            else:
                verdict = "FAIL(dilution or inconsistent)"
            print(f"  {side:<7}{combo:<17} lift={lifts[combo]:.2f} vs best_parent={best_parent:.2f} "
                  f"n={n} VAL/OOS={lv:.2f}/{lo:.2f} -> {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / "combo_evidence_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'combo_evidence_table.csv'}")


if __name__ == "__main__":
    main()
