"""
CUSUM-filtered triple-barrier quality classifier for BTC (h48qual-style
architecture: rule-based event detector (CUSUM) + learned quality/direction
classifier), on the new 5m+1h+metrics4 feature frame.

Pipeline:
  1. CUSUM events (mult=2.0, same as the label-scheme comparison) -> h48_conservative
     triple-barrier outcome at each event bar (long_quality, short_quality).
  2. Train two LightGBM regressors (long_quality, short_quality) on event-bar
     features, train-only rows (entry_ts < 2025-09-01).
  3. Predict on VAL (2025-09-01..2025-12-31) and OOS (2026-01-01..2026-03-31)
     event bars; sweep an action threshold on predicted quality; at each
     threshold, compute REALIZED (not predicted) net return using the same
     conservative cost basis as the earlier hybrid check.

Diagnostic/dev-score only, not a Fresh-Forward validated promotion claim
(features are already causal per the existing pipeline's shift conventions,
but this script does one shot in-sample->OOS evaluation, not a full walk-forward).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move, _reason_and_return  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
OUT_DIR = ROOT / "tmp/btc_cusum_tb_quality_20260803"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TB_HORIZON = 48
TB_TP_MULT, TB_SL_MULT = 1.2, 0.8
TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
FEE_COST = 0.0007  # same per-side quality-formula cost h48qual's own build script uses

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0  # 0.42%, same buffer used earlier

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
}


def build_event_labels(frame: pd.DataFrame, events: np.ndarray) -> pd.DataFrame:
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = frame["timestamp"]
    n = len(frame)

    rows = []
    for i in events:
        entry_i = i + 1
        end_i = entry_i + TB_HORIZON
        if end_i + 1 > n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(TB_MIN_TP, TB_TP_MULT * vol)
        sl_move = max(TB_MIN_SL, TB_SL_MULT * vol)
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
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < len(frame) - TB_HORIZON - 2]

    labels = build_event_labels(frame, events)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"events: total={len(data)} train={len(train)} val={len(val)} oos={len(oos)}")

    models = {}
    for side, target in [("long", "long_q"), ("short", "short_q")]:
        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        model.fit(train[feat_cols], train[target])
        models[side] = model

    for split_name, split in [("VAL", val), ("OOS", oos)]:
        pred_long = models["long"].predict(split[feat_cols])
        pred_short = models["short"].predict(split[feat_cols])
        realized_long = split["long_ret"].to_numpy() - COST_CONSERVATIVE
        realized_short = split["short_ret"].to_numpy() - COST_CONSERVATIVE
        print(f"\n=== {split_name} (n={len(split)}) — threshold sweep on predicted quality ===")
        for thresh in [0.0, 0.001, 0.002, 0.004, 0.006, 0.010]:
            take_long = pred_long >= thresh
            take_short = (pred_short >= thresh) & (pred_short > pred_long)
            take_long = take_long & ~take_short
            n_trades = int(take_long.sum() + take_short.sum())
            if n_trades == 0:
                print(f"  thresh={thresh:.3f}  n=0")
                continue
            net = np.concatenate([realized_long[take_long], realized_short[take_short]])
            win = (net > 0).sum()
            print(f"  thresh={thresh:.3f}  n={n_trades:4d}  win%={100*win/n_trades:5.1f}  "
                  f"mean_net={100*net.mean():6.3f}%  sum_net={100*net.sum():8.2f}%")


if __name__ == "__main__":
    main()
