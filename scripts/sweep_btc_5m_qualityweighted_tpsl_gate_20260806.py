"""BTC 5m: full hyperparameter sweep on top of the quality-weighted LightGBM Layer B (best result
so far this session, OOS -3.98% at default TP/SL/gate). Two-stage grid:
  1. quality-weight scale (retrains Layer B LightGBM, the expensive step)
  2. TP_MULT x SL_MULT x gate_pctl (cheap, backtest-only, no retraining)

Selection is VAL-only (never peek at OOS while choosing hyperparameters) -- the single winning
config is then evaluated on OOS exactly once, to avoid multiple-comparison overfitting on the
already-thin trade counts this session has seen throughout (19-79 trades per config).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
QUALITY_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_quality_oracle_20260806.parquet"
LAYERA_PRED_PATH = ROOT / "tmp/btc_1h_volregime_20260805/btc5m_layerA_pred.parquet"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"

DROP_RAW = {"timestamp", "open", "high", "low", "close", "close_btc", "volume_btc", "quote_volume_btc"}
VAL_START, OOS_START, OOS_END = "2025-09-01", "2026-01-01", "2026-04-01"
MAX_HOLD = 288
ROUND_TRIP_COST = 0.0010
TRAIL_VOL_BARS = 288
MIN_TRADES = 15  # ignore configs with too few trades to trust

WEIGHT_SCALES = [0, 20, 40, 60, 80, 120]
TP_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
SL_GRID = [0.8, 1.0, 1.2, 1.5]
GATE_PCTL_GRID = [0.80, 0.85, 0.90, 0.93, 0.95]


def build_dvol_features() -> pd.DataFrame:
    df = pd.read_csv(DVOL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
    df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
    df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
    df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
    df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    return df


def load_frame():
    panel = pd.read_parquet(PANEL_PATH)
    labels = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "zigzag_action"])
    qual = pd.read_parquet(QUALITY_PATH, columns=["timestamp", "net_ret_sim"])
    dvol = build_dvol_features()
    layerA = pd.read_parquet(LAYERA_PRED_PATH)

    df = panel.merge(labels, on="timestamp", how="inner").merge(qual, on="timestamp", how="inner")
    df = pd.merge_asof(df.sort_values("timestamp"), dvol, on="timestamp", direction="backward")
    df = df.merge(layerA, on="timestamp", how="inner")
    df = df.dropna(subset=["zigzag_action"]).reset_index(drop=True)

    log_ret = np.log(df["close"]).diff()
    df["trailing_vol"] = log_ret.rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std()

    feature_cols = [c for c in panel.columns if c not in DROP_RAW] + [
        "dvol_btc", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "dvol_btc_pctrank_720h",
    ]
    return df, feature_cols


def train_layerB(df: pd.DataFrame, feature_cols: list[str], weight_scale: float) -> np.ndarray:
    X = df[feature_cols]
    y = df["zigzag_action"].astype(int)
    train_mask = (df["timestamp"] < VAL_START).to_numpy()

    if weight_scale == 0:
        sample_weight = None
    else:
        sw = 1.0 + df["net_ret_sim"].fillna(0.0).to_numpy() * weight_scale
        sample_weight = np.clip(sw, 0.2, 3.0)[train_mask]

    clf = LGBMClassifier(n_estimators=400, num_leaves=31, learning_rate=0.05, min_child_samples=100, verbosity=-1)
    clf.fit(X[train_mask], y[train_mask], sample_weight=sample_weight)
    return clf.predict(X)


def run_backtest(sub: pd.DataFrame, tp_mult: float, sl_mult: float, layerA_thresh: float) -> dict:
    close = sub["close"].to_numpy()
    high = sub["high"].to_numpy()
    low = sub["low"].to_numpy()
    vol = sub["trailing_vol"].to_numpy()
    probA = sub["probA"].to_numpy()
    pred = sub["pred"].to_numpy()
    n = len(sub)

    trades = []
    i = 0
    while i < n:
        if not np.isfinite(vol[i]) or vol[i] <= 0:
            i += 1
            continue
        action = int(pred[i])
        if action == 0 or probA[i] < layerA_thresh:
            i += 1
            continue
        direction = 1 if action == 1 else -1
        entry_price = close[i]
        v = vol[i]
        tp_price = entry_price * (1 + direction * tp_mult * v)
        sl_price = entry_price * (1 - direction * sl_mult * v)
        exit_price = None
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            hit_tp = high[j] >= tp_price if direction == 1 else low[j] <= tp_price
            hit_sl = low[j] <= sl_price if direction == 1 else high[j] >= sl_price
            if hit_tp and hit_sl:
                exit_price = sl_price
                break
            if hit_tp:
                exit_price = tp_price
                break
            if hit_sl:
                exit_price = sl_price
                break
        if exit_price is None:
            exit_price = close[j_end]
        ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        trades.append(ret)
        i = j_end + 1

    if not trades:
        return {"n_trades": 0, "sum_ret_pct": -999.0, "mdd_pct": -999.0}
    trades = np.array(trades)
    equity = np.cumsum(trades)
    mdd = (equity - np.maximum.accumulate(equity)).min()
    return {"n_trades": len(trades), "sum_ret_pct": trades.sum() * 100, "win_rate": (trades > 0).mean(), "mdd_pct": mdd * 100}


def main() -> int:
    df, feature_cols = load_frame()
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] < OOS_END)

    all_results = []
    for ws in WEIGHT_SCALES:
        print(f"\n=== training Layer B with weight_scale={ws} ===")
        pred = train_layerB(df, feature_cols, ws)
        df_ws = df.copy()
        df_ws["pred"] = pred
        val_df = df_ws[val_mask].reset_index(drop=True)

        for gate_pctl in GATE_PCTL_GRID:
            thresh = val_df["probA"].quantile(gate_pctl)
            for tp in TP_GRID:
                for sl in SL_GRID:
                    res = run_backtest(val_df, tp, sl, thresh)
                    if res["n_trades"] < MIN_TRADES:
                        continue
                    all_results.append({
                        "weight_scale": ws, "gate_pctl": gate_pctl, "tp_mult": tp, "sl_mult": sl,
                        "gate_thresh": thresh, "val_n_trades": res["n_trades"],
                        "val_sum_ret_pct": res["sum_ret_pct"], "val_win_rate": res["win_rate"],
                        "val_mdd_pct": res["mdd_pct"],
                    })

    results_df = pd.DataFrame(all_results)
    print(f"\ntotal valid configs (n_trades>={MIN_TRADES}): {len(results_df)}")
    results_df = results_df.sort_values("val_sum_ret_pct", ascending=False)
    print("\ntop 15 by VAL sum_ret_pct:")
    print(results_df.head(15).to_string())

    best = results_df.iloc[0]
    print(f"\n### BEST CONFIG (chosen on VAL only): {best.to_dict()}")

    # confirm on OOS exactly once
    pred = train_layerB(df, feature_cols, best["weight_scale"])
    df_best = df.copy()
    df_best["pred"] = pred
    oos_df = df_best[oos_mask].reset_index(drop=True)
    oos_res = run_backtest(oos_df, best["tp_mult"], best["sl_mult"], best["gate_thresh"])
    print(f"\n### OOS confirmation of VAL-selected config: {oos_res}")

    results_df.to_csv(ROOT / "tmp/btc_1h_volregime_20260805/sweep_results_20260806.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
