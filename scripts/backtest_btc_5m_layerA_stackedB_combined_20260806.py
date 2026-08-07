"""BTC 5m: Layer A (transition gate, AUC 0.75/0.77) + Layer B v4 (3-class, CASH included) combined.
Untried combo -- 5m previously only tested binary Layer B (collapsed), and 1h's 3-class combo was
the best result of that timeframe (OOS ~breakeven). Sequential, non-overlapping positions, ATR/vol
TP/SL, same cost convention as the rest of this session.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LAYERA_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
LAYERB_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerB_stacked_pred.parquet"

VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
TRAIL_VOL_BARS = 288
MAX_HOLD = 288
ROUND_TRIP_COST = 0.0010
TP_MULT, SL_MULT = 2.5, 1.2


def load_combined() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "close", "high", "low"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    log_ret = np.log(panel["close"]).diff()
    panel["trailing_vol"] = log_ret.rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std()

    layerA = pd.read_parquet(LAYERA_PRED_PATH)
    layerB = pd.read_parquet(LAYERB_PRED_PATH)
    df = panel.merge(layerA, on="timestamp", how="inner").merge(layerB, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)


def run_backtest(df: pd.DataFrame, layerA_thresh: float, use_gate: bool) -> dict:
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol = df["trailing_vol"].to_numpy()
    probA = df["probA"].to_numpy()
    pred = df["pred"].to_numpy()
    ts = df["timestamp"].to_numpy()
    n = len(df)

    trades = []
    i = 0
    while i < n:
        if not np.isfinite(vol[i]) or vol[i] <= 0:
            i += 1
            continue
        action = int(pred[i])
        if action == 0:
            i += 1
            continue
        if use_gate and probA[i] < layerA_thresh:
            i += 1
            continue
        direction = 1 if action == 1 else -1

        entry_price = close[i]
        v = vol[i]
        tp_price = entry_price * (1 + direction * TP_MULT * v)
        sl_price = entry_price * (1 - direction * SL_MULT * v)

        exit_price, exit_reason = None, "time"
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            hit_tp = high[j] >= tp_price if direction == 1 else low[j] <= tp_price
            hit_sl = low[j] <= sl_price if direction == 1 else high[j] >= sl_price
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
            exit_price, exit_reason = close[j_end], "time"

        ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        trades.append({"entry_ts": ts[i], "direction": direction, "ret": ret, "exit_reason": exit_reason})
        i = j_end + 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"n_trades": 0}
    equity = tdf["ret"].cumsum()
    mdd = (equity - equity.cummax()).min()
    return {
        "n_trades": len(tdf),
        "sum_ret_pct": round(tdf["ret"].sum() * 100, 3),
        "mean_ret_pct": round(tdf["ret"].mean() * 100, 4),
        "win_rate": round((tdf["ret"] > 0).mean(), 4),
        "mdd_pct": round(mdd * 100, 3),
        "exit_reasons": tdf["exit_reason"].value_counts().to_dict(),
    }


def main() -> int:
    df = load_combined()
    val_df = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)]

    for pctl in (0.90, 0.95, 0.97, 0.99):
        layerA_thresh = val_df["probA"].quantile(pctl)
        print(f"\n########## gate percentile={pctl:.0%} (threshold={layerA_thresh:.4f}) ##########")
        for split_name, start, end in [("VAL", VAL_START, OOS_START), ("OOS", OOS_START, OOS_END)]:
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
            res_combined = run_backtest(sub, layerA_thresh=layerA_thresh, use_gate=True)
            print(f"{split_name}: {res_combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
