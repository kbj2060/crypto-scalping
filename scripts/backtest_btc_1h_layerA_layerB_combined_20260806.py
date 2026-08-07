"""BTC 1h new-architecture, Layer A + Layer B combined at SERVE time (per user: train each label
independently, connect the two models at the model/serving level, not via a conditioned label).

Layer A: swing-transition detector (scripts/build_btc_1h_pivot_transition_labels_20260805.py,
scripts/eval_btc_1h_pivot_transition_predictability_20260805.py) -- OOS AUC 0.70, trained/evaluated
independently on ALL bars.
Layer B: zigzag direction classifier (scripts/build_btc_1h_zigzag_labels_20260805.py,
scripts/eval_btc_1h_zigzag_predictability_20260805.py) -- OOS acc 50.9%/macro-F1 0.489, also
trained/evaluated independently on ALL bars.

Serve-time combination: enter only on bars where Layer A's P(transition_soon) clears a threshold
AND Layer B's predicted action is LONG or SHORT (skip CASH). Exit via ATR-vol-sized TP/SL or
MAX_HOLD bars, same convention as the earlier breakout backtest. One position at a time.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIGZAG_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/zigzag_pred_full.parquet"
PIVOT_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/pivot_transition_pred_full.parquet"
VOLREGIME_LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"

VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
MAX_HOLD = 24
ROUND_TRIP_COST = 0.0010
TP_MULT, SL_MULT = 2.5, 1.2
PROB_THRESH_PCTL = 0.90  # top-decile, matches the VAL-fit threshold used in the Layer A cheap check


def load_combined() -> pd.DataFrame:
    zz = pd.read_parquet(ZIGZAG_PRED_PATH, columns=["timestamp", "close", "high", "low", "pred"])
    zz = zz.rename(columns={"pred": "layerB_action"})  # 0=CASH 1=LONG 2=SHORT
    piv = pd.read_parquet(PIVOT_PRED_PATH, columns=["timestamp", "prob"])
    piv = piv.rename(columns={"prob": "layerA_prob"})
    vol = pd.read_parquet(VOLREGIME_LABEL_PATH, columns=["timestamp", "trailing_vol_24h"])
    df = zz.merge(piv, on="timestamp", how="inner").merge(vol, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)


def run_backtest(df: pd.DataFrame, prob_thresh: float, use_gate: bool) -> dict:
    trades = []
    i, n = 0, len(df)
    while i < n:
        row = df.iloc[i]
        if pd.isna(row["trailing_vol_24h"]):
            i += 1
            continue
        action = int(row["layerB_action"])
        if action == 0:
            i += 1
            continue
        if use_gate and row["layerA_prob"] < prob_thresh:
            i += 1
            continue
        direction = 1 if action == 1 else -1

        entry_price = row["close"]
        vol = row["trailing_vol_24h"]
        tp_price = entry_price * (1 + direction * TP_MULT * vol)
        sl_price = entry_price * (1 - direction * SL_MULT * vol)

        exit_price, exit_reason = None, "time"
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            bar = df.iloc[j]
            hit_tp = bar["high"] >= tp_price if direction == 1 else bar["low"] <= tp_price
            hit_sl = bar["low"] <= sl_price if direction == 1 else bar["high"] >= sl_price
            if hit_tp and hit_sl:
                exit_price, exit_reason = sl_price, "sl"
                break
            if hit_tp:
                exit_price, exit_reason = tp_price, "tp"
                break
            if hit_sl:
                exit_price, exit_reason = sl_price, "sl"
                break
        if exit_price is None:
            exit_price, exit_reason = df.iloc[j_end]["close"], "time"

        ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        trades.append({"entry_ts": row["timestamp"], "direction": direction, "ret": ret, "exit_reason": exit_reason})
        i = j_end + 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"n_trades": 0}
    equity = tdf["ret"].cumsum()
    mdd = (equity - equity.cummax()).min()
    return {
        "n_trades": len(tdf),
        "sum_ret_pct": tdf["ret"].sum() * 100,
        "mean_ret_pct": tdf["ret"].mean() * 100,
        "win_rate": (tdf["ret"] > 0).mean(),
        "mdd_pct": mdd * 100,
        "exit_reasons": tdf["exit_reason"].value_counts().to_dict(),
    }


def main() -> int:
    df = load_combined()

    # fit the gate threshold on VAL only (causal: never peek at OOS to set it), apply the SAME
    # fixed threshold to both splits -- same convention as a real deployment would use.
    val_df = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)]
    thresh = val_df["layerA_prob"].quantile(PROB_THRESH_PCTL)
    print(f"gate threshold fit on VAL ({PROB_THRESH_PCTL:.0%}ile) = {thresh:.4f}")

    for split_name, start, end in [("VAL", VAL_START, OOS_START), ("OOS", OOS_START, OOS_END)]:
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        print(f"\n===== {split_name} ({start}..{end}, n_bars={len(sub)}) =====")
        res_b_only = run_backtest(sub, prob_thresh=0.0, use_gate=False)
        print(f"-- LAYER B ONLY (no gate): {res_b_only}")
        res_combined = run_backtest(sub, prob_thresh=thresh, use_gate=True)
        print(f"-- LAYER A+B COMBINED (gated): {res_combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
