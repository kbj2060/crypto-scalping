"""BTC 1h new-architecture, Step 4: does gating a plain 24h-breakout entry with the vol-regime
"expansion" prediction (scripts/eval_btc_1h_volregime_predictability_20260805.py) improve on the
same breakout ungated? Fresh-Forward bar-by-bar walk, VAL then OOS, gated vs ungated compared
side by side (cheap falsification of the B2 hypothesis as an actual PnL source).

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.

Entry: close crosses above/below the prior LOOKBACK-bar high/low (causal, shift(1) excludes the
current bar). Exit: TP/SL sized off trailing_vol_24h (known causally at entry) or MAX_HOLD bars,
whichever first. One position at a time. ROUND_TRIP_COST applied flat per trade (assumption,
noted below -- not tuned).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"

LOOKBACK = 24
TP_MULT, SL_MULT = 2.5, 1.2
MAX_HOLD = 24
ROUND_TRIP_COST = 0.0010  # 10bps flat assumption, not tuned -- treat headline numbers as indicative


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_csv(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "label_3class", "trailing_vol_24h"])
    dvol = build_dvol_features()

    df = panel.merge(labels, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.dropna(subset=["label_3class"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h", "trailing_vol_24h",
    ]
    return df, feature_cols


def add_predictions(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    train_mask = df["timestamp"] < VAL_START
    clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    clf.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, "label_3class"].astype(int))
    df = df.copy()
    df["pred_label"] = clf.predict(df[feature_cols])
    return df


def add_breakout_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roll_high"] = df["high"].shift(1).rolling(LOOKBACK).max()
    df["roll_low"] = df["low"].shift(1).rolling(LOOKBACK).min()
    df["long_trig"] = df["close"] > df["roll_high"]
    df["short_trig"] = df["close"] < df["roll_low"]
    return df


SIZE_MULT = {1: 1.5, 0: 1.0, -1: 0.5}  # expansion=full+, stable=normal, contraction=de-risked


def run_backtest_sized(df: pd.DataFrame) -> dict:
    """Risk-overlay variant: take EVERY breakout trade (no entry gate), but scale position size
    by the vol-regime prediction at entry (expansion=1.5x, stable=1.0x, contraction=0.5x). Tests
    whether the vol-regime signal adds value as a sizing/risk overlay rather than an entry filter,
    on the same (edge-less) breakout entries."""
    trades = []
    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        if pd.isna(row["roll_high"]) or pd.isna(row["trailing_vol_24h"]):
            i += 1
            continue
        direction = 0
        if row["long_trig"] and not row["short_trig"]:
            direction = 1
        elif row["short_trig"] and not row["long_trig"]:
            direction = -1
        if direction == 0:
            i += 1
            continue

        entry_price = row["close"]
        vol = row["trailing_vol_24h"]
        size = SIZE_MULT[int(row["pred_label"])]
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

        raw_ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        trades.append({"entry_ts": row["timestamp"], "direction": direction, "size": size,
                        "ret": raw_ret * size, "raw_ret": raw_ret, "exit_reason": exit_reason})
        i = j_end + 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"n_trades": 0}
    equity = tdf["ret"].cumsum()
    mdd = (equity - equity.cummax()).min()
    equity_unsized = tdf["raw_ret"].cumsum()
    mdd_unsized = (equity_unsized - equity_unsized.cummax()).min()
    return {
        "n_trades": len(tdf),
        "sum_ret_pct_sized": tdf["ret"].sum() * 100,
        "sum_ret_pct_unsized": tdf["raw_ret"].sum() * 100,
        "mdd_pct_sized": mdd * 100,
        "mdd_pct_unsized": mdd_unsized * 100,
        "win_rate": (tdf["ret"] > 0).mean(),
        "avg_size": tdf["size"].mean(),
    }


def run_backtest(df: pd.DataFrame, gated: bool, return_trades: bool = False):
    trades = []
    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        if pd.isna(row["roll_high"]) or pd.isna(row["trailing_vol_24h"]):
            i += 1
            continue
        direction = 0
        if row["long_trig"] and not row["short_trig"]:
            direction = 1
        elif row["short_trig"] and not row["long_trig"]:
            direction = -1
        if direction == 0:
            i += 1
            continue
        if gated and row["pred_label"] != 1:
            i += 1
            continue

        entry_price = row["close"]
        vol = row["trailing_vol_24h"]
        tp_price = entry_price * (1 + direction * TP_MULT * vol)
        sl_price = entry_price * (1 - direction * SL_MULT * vol)

        exit_price, exit_reason, exit_ts = None, "time", None
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            bar = df.iloc[j]
            hit_tp = bar["high"] >= tp_price if direction == 1 else bar["low"] <= tp_price
            hit_sl = bar["low"] <= sl_price if direction == 1 else bar["high"] >= sl_price
            if hit_tp and hit_sl:
                exit_price, exit_reason, exit_ts = sl_price, "sl", bar["timestamp"]  # conservative: assume SL hit first if both touched same bar
                break
            if hit_tp:
                exit_price, exit_reason, exit_ts = tp_price, "tp", bar["timestamp"]
                break
            if hit_sl:
                exit_price, exit_reason, exit_ts = sl_price, "sl", bar["timestamp"]
                break
        if exit_price is None:
            exit_price, exit_reason, exit_ts = df.iloc[j_end]["close"], "time", df.iloc[j_end]["timestamp"]

        ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        trades.append({"entry_ts": row["timestamp"], "exit_ts": exit_ts, "direction": direction,
                        "entry_price": entry_price, "exit_price": exit_price, "tp_price": tp_price,
                        "sl_price": sl_price, "pred_label": int(row["pred_label"]),
                        "ret": ret, "exit_reason": exit_reason})
        i = j_end + 1  # no overlapping positions

    tdf = pd.DataFrame(trades)
    if return_trades:
        return tdf
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
    df, feature_cols = load_frame()
    df = add_predictions(df, feature_cols)
    df = add_breakout_signal(df)

    for split_name, start, end in [("VAL", VAL_START, OOS_START), ("OOS", OOS_START, OOS_END)]:
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        print(f"\n===== {split_name} ({start} .. {end}, n_bars={len(sub)}) =====")
        for gated in (False, True):
            res = run_backtest(sub, gated=gated)
            tag = "GATED (expansion only)" if gated else "UNGATED (plain breakout)"
            print(f"-- {tag}: {res}")
        res_sized = run_backtest_sized(sub)
        print(f"-- SIZED (risk overlay, all breakouts, size by regime): {res_sized}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
