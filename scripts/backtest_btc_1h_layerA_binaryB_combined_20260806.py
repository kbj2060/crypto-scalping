"""BTC 1h new-architecture, final combined backtest: Layer A (transition detector, OOS AUC 0.70)
gates entry timing, Layer B v3 (binary LONG/SHORT direction, OOS acc 70.6%/AUC 0.80 restricted to
active-wave bars) provides direction. Sequential, non-overlapping position management, same ATR/vol
TP/SL and cost convention used throughout this session.

Caveat this backtest is designed to surface: Layer B was trained/evaluated ONLY on oracle-active
(non-CASH) bars. Layer A's real-world gate is imperfect (AUC 0.70, not 1.0), so some bars it passes
through will actually be CASH/transition bars where Layer B's binary call is meaningless noise --
this combined test is what actually measures whether that degrades the standalone Layer B numbers.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BINARY_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/zigzag_binary_pred_full.parquet"
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"
VOLREGIME_LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
MAX_HOLD = 24
ROUND_TRIP_COST = 0.0010
TP_MULT, SL_MULT = 2.5, 1.2
CONF_PCTL = 0.70  # Layer B confidence gate (top-30% by |prob-0.5|, fit on VAL)


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def score_layerB_on_all_bars():
    """Layer B was trained on oracle-active bars only; here we score EVERY bar (since at serve
    time we don't know in advance which bars are active -- that's what Layer A is for)."""
    from lightgbm import LGBMClassifier

    panel = pd.read_csv(PANEL_PATH)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    labels = pd.read_parquet(ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet",
                              columns=["timestamp", "zigzag_action"])
    active_labels = labels[labels["zigzag_action"].isin([1, 2])]
    dvol = build_dvol_features()

    train_frame = panel.merge(active_labels, on="timestamp", how="inner")
    train_frame = pd.merge_asof(train_frame.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    train_frame = train_frame.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    train_mask = train_frame["timestamp"] < VAL_START
    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=50, verbosity=-1)
    clf.fit(train_frame.loc[train_mask, feature_cols], (train_frame.loc[train_mask, "zigzag_action"] == 1).astype(int))

    full = pd.merge_asof(panel.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    full["pred_prob_long"] = clf.predict_proba(full[feature_cols])[:, 1]
    return full[["timestamp", "pred_prob_long"]]


def load_combined() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "close", "high", "low"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    layerB = score_layerB_on_all_bars()
    piv_prob = pd.read_parquet(ROOT / "tmp/btc_1h_volregime_20260805/pivot_transition_pred_full.parquet",
                                columns=["timestamp", "prob"]).rename(columns={"prob": "layerA_prob"})
    vol = pd.read_parquet(VOLREGIME_LABEL_PATH, columns=["timestamp", "trailing_vol_24h"])
    df = panel.merge(layerB, on="timestamp", how="inner").merge(piv_prob, on="timestamp", how="inner").merge(vol, on="timestamp", how="inner")
    return df.sort_values("timestamp").reset_index(drop=True)


def run_backtest(df: pd.DataFrame, layerA_thresh: float, conf_thresh: float, use_gate: bool) -> dict:
    trades = []
    i, n = 0, len(df)
    while i < n:
        row = df.iloc[i]
        if pd.isna(row["trailing_vol_24h"]):
            i += 1
            continue
        conf = abs(row["pred_prob_long"] - 0.5)
        if use_gate:
            if row["layerA_prob"] < layerA_thresh or conf < conf_thresh:
                i += 1
                continue
        else:
            if conf < conf_thresh:
                i += 1
                continue
        direction = 1 if row["pred_prob_long"] >= 0.5 else -1

        entry_price = row["close"]
        v = row["trailing_vol_24h"]
        tp_price = entry_price * (1 + direction * TP_MULT * v)
        sl_price = entry_price * (1 - direction * SL_MULT * v)

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
        "sum_ret_pct": round(tdf["ret"].sum() * 100, 3),
        "mean_ret_pct": round(tdf["ret"].mean() * 100, 4),
        "win_rate": round((tdf["ret"] > 0).mean(), 4),
        "mdd_pct": round(mdd * 100, 3),
        "exit_reasons": tdf["exit_reason"].value_counts().to_dict(),
    }


def main() -> int:
    df = load_combined()
    val_df = df[(df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)]
    layerA_thresh = val_df["layerA_prob"].quantile(0.90)
    conf_thresh = (val_df["pred_prob_long"] - 0.5).abs().quantile(CONF_PCTL)
    print(f"gate thresholds fit on VAL: layerA_prob>={layerA_thresh:.4f}, layerB_conf>={conf_thresh:.4f}")

    for split_name, start, end in [("VAL", VAL_START, OOS_START), ("OOS", OOS_START, OOS_END)]:
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
        print(f"\n===== {split_name} ({start}..{end}, n_bars={len(sub)}) =====")
        res_conf_only = run_backtest(sub, layerA_thresh=0.0, conf_thresh=conf_thresh, use_gate=False)
        print(f"-- LAYER B confidence-gate ONLY (no Layer A): {res_conf_only}")
        res_combined = run_backtest(sub, layerA_thresh=layerA_thresh, conf_thresh=conf_thresh, use_gate=True)
        print(f"-- LAYER A + LAYER B combined: {res_combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
