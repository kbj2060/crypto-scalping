"""Barrier/horizon calibration sweep for BTC -- the remaining untried lever per
docs/btc_3way_tpfirst_label_closed_20260804.md ("남은, 아직 시도 안 한 레버는 라벨
패러다임이 아니라 barrier/horizon 캘리브레이션 자체"): both label paradigms
(magnitude-regression and TP-first classification) are closed on the current best
feature set at two fixed calibrations (h48qual_shape, longhold_shape). This sweeps
horizon and TP/SL multiplier directly, holding the label paradigm (dense-nogate
quality regression, the project's standard) fixed, to check whether the wall found
so far is calibration-specific or paradigm-general.

Same pipeline as scripts/train_eval_btc_dense_nogate_quality_unified_raw_20260804.py
(same _reason_and_return barrier-touch function, same VAL/OOS split, same cost
model, same n_trades>=15 pass bar) -- only horizon_bars/tp_mult/sl_mult vary.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move, _reason_and_return  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
OUT_CSV = ROOT / "tmp/btc_barrier_horizon_calibration_sweep_20260805.csv"

TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
FEE_COST = 0.0007
VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
}

HORIZONS = [24, 48, 96, 192, 384, 576]  # 2h/4h/8h/16h/32h/48h at 5m bars
TP_SL_PAIRS = [(1.0, 0.6), (1.2, 0.8), (1.5, 1.0), (2.0, 1.2), (2.5, 1.5)]
THRESHOLDS = [0.0, 0.002, 0.004, 0.006, 0.010]
STRIDE = 5


def build_dense_labels(frame: pd.DataFrame, idxs: np.ndarray, horizon: int, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = frame["timestamp"]

    rows = []
    for i in idxs:
        entry_i = i + 1
        end_i = entry_i + horizon
        if end_i + 1 > n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(TB_MIN_TP, tp_mult * vol)
        sl_move = max(TB_MIN_SL, sl_mult * vol)
        fh, fl, fc = high[entry_i:end_i + 1], low[entry_i:end_i + 1], close[entry_i:end_i + 1]
        long_ret, long_reason, _, _, _ = _reason_and_return(
            side=1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        short_ret, short_reason, _, _, _ = _reason_and_return(
            side=-1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        long_q = long_ret - FEE_COST - 0.003 * int(long_reason == "sl")
        short_q = short_ret - FEE_COST - 0.003 * int(short_reason == "sl")
        rows.append({"i": i, "timestamp": ts.iloc[i], "long_ret": long_ret, "short_ret": short_ret,
                      "long_q": long_q, "short_q": short_q})
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n = len(frame)
    print(f"unified raw panel: {n} rows, {len(feat_cols)} feature cols")
    print(f"sweep: {len(HORIZONS)} horizons x {len(TP_SL_PAIRS)} tp/sl pairs = {len(HORIZONS)*len(TP_SL_PAIRS)} configs")

    all_results = []
    t_start = time.time()
    for horizon in HORIZONS:
        for tp_mult, sl_mult in TP_SL_PAIRS:
            name = f"h{horizon}_tp{tp_mult}_sl{sl_mult}"
            idxs = np.arange(0, n - horizon - 2, STRIDE)
            t0 = time.time()
            labels = build_dense_labels(frame, idxs, horizon, tp_mult, sl_mult)

            event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
            data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

            train = data[data["timestamp"] < VAL_START]
            val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
            oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
            if len(train) < 500 or len(val) < 30 or len(oos) < 30:
                print(f"{name}: skipped, too few rows (train={len(train)} val={len(val)} oos={len(oos)})")
                continue

            models = {}
            for side, target in [("long", "long_q"), ("short", "short_q")]:
                model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                           subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
                model.fit(train[feat_cols], train[target])
                models[side] = model

            best_val_mean, best_oos_mean, n_thresh_pass = None, None, 0
            for split_name, split in [("VAL", val), ("OOS", oos)]:
                pred_long = models["long"].predict(split[feat_cols])
                pred_short = models["short"].predict(split[feat_cols])
                realized_long = split["long_ret"].to_numpy() - COST_CONSERVATIVE
                realized_short = split["short_ret"].to_numpy() - COST_CONSERVATIVE
                for thresh in THRESHOLDS:
                    take_long = pred_long >= thresh
                    take_short = (pred_short >= thresh) & (pred_short > pred_long)
                    take_long = take_long & ~take_short
                    n_trades = int(take_long.sum() + take_short.sum())
                    if n_trades == 0:
                        continue
                    net = np.concatenate([realized_long[take_long], realized_short[take_short]])
                    win = (net > 0).sum()
                    all_results.append({
                        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult, "config": name,
                        "split": split_name, "thresh": thresh, "n_trades": n_trades,
                        "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                        "sum_net_pct": 100 * net.sum(),
                    })
            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            print(f"{name}: n={len(labels)} train={len(train)} val={len(val)} oos={len(oos)} "
                  f"({elapsed:.1f}s, total {total_elapsed:.0f}s)")

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    val_pos = out[(out["split"] == "VAL") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    oos_pos = out[(out["split"] == "OOS") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    both = val_pos.merge(oos_pos, on=["config", "horizon", "tp_mult", "sl_mult", "thresh"], suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)} ===")
    if len(both):
        print(both[["config", "thresh", "n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]]
              .sort_values("mean_net_pct_oos", ascending=False).to_string(index=False))
    else:
        print("(none) -- barrier/horizon calibration sweep gate FAILS at every point tested.")

    # Even without a pass, report the closest OOS results (least-bad / most-promising)
    # to see if any region of the grid trends toward positive.
    oos_all = out[(out["split"] == "OOS") & (out["n_trades"] >= 15)].sort_values("mean_net_pct", ascending=False)
    print(f"\n=== Top 15 OOS results by mean_net_pct (n>=15), regardless of pass/fail ===")
    print(oos_all.head(15)[["config", "thresh", "n_trades", "win_pct", "mean_net_pct", "sum_net_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
