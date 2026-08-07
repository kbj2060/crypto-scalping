"""Re-examine the labeling/architecture PARADIGM itself, not another feature axis (per user
request 2026-08-04, after DVOL became the third independent data source to fail the identical
cheap-falsification test at the identical -0.35%..-0.55%/trade band -- see
project-btc-deribit-dvol-data-acquired-20260804.md's closing note: this narrow, repeated negative
band across unrelated data sources is itself a clue that the bottleneck may be the labeling/
evaluation architecture, not missing information).

Two concrete, checkable hypotheses about the triple-barrier quality-classifier paradigm used by
every closed line this session (train_eval_btc_dense_nogate_quality_20260804.py and its variants):

H_cost: train-time target assumes a LIGHT cost (long_q = long_ret - FEE_COST(0.07%) -
0.3%*is_sl_hit), but eval-time backtests subtract a FLAT, HEAVIER cost (COST_CONSERVATIVE = 0.42%)
regardless of exit reason. On a TP/timeout trade this is a ~0.35% mismatch the model was never
trained to anticipate -- big relative to typical per-trade moves. This decomposes gross vs net
returns to see how much of the "-0.4%/trade" negative result is actually pure cost drag on an
otherwise-near-breakeven or mildly positive gross signal, vs a genuinely negative gross edge.

H_selection: entry selection takes whichever side (long_q or short_q) has the higher PREDICTED
quality above a threshold. Since long_q and short_q are two independently-noisy regression outputs
on the same underlying price path, always taking the argmax-above-threshold side has a structural
"winner's curse" flavor (favoring whichever prediction has the larger positive ERROR, not
necessarily the larger true edge) -- this checks whether predicted-quality quintiles actually
rank-order REALIZED gross returns monotonically (a real, well-calibrated signal should), which is
a more informative diagnostic than a single win/loss threshold sweep.
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

BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"

TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
FEE_COST = 0.0007
VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
}

HORIZON, TP_MULT, SL_MULT = 576, 2.0, 1.2  # "longhold_shape" -- the config with more OOS trades
STRIDE = 3


def build_dense_labels(frame: pd.DataFrame, idxs: np.ndarray) -> pd.DataFrame:
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
        end_i = entry_i + HORIZON
        if end_i + 1 > n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        tp_move = max(TB_MIN_TP, TP_MULT * vol)
        sl_move = max(TB_MIN_SL, SL_MULT * vol)
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
                      "long_q": long_q, "short_q": short_q,
                      "long_reason": long_reason, "short_reason": short_reason})
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(BTC_FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n = len(frame)

    idxs = np.arange(0, n - HORIZON - 2, STRIDE)
    t0 = time.time()
    labels = build_dense_labels(frame, idxs)
    print(f"built {len(labels)} dense samples in {time.time()-t0:.1f}s")

    event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

    train = data[data["timestamp"] < VAL_START]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"train={len(train)} oos={len(oos)}")

    models = {}
    for side, target in [("long", "long_q"), ("short", "short_q")]:
        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        model.fit(train[feat_cols], train[target])
        models[side] = model

    pred_long = models["long"].predict(oos[feat_cols])
    pred_short = models["short"].predict(oos[feat_cols])

    # ================= H_selection: does predicted quality rank-order realized gross return? =================
    print("\n=== H_selection: predicted-quality quintiles vs realized GROSS return (OOS, longhold config) ===")
    for side_name, pred, gross in [("long", pred_long, oos["long_ret"].to_numpy()),
                                     ("short", pred_short, oos["short_ret"].to_numpy())]:
        qbins = pd.qcut(pred, 5, labels=False, duplicates="drop")
        df = pd.DataFrame({"qbin": qbins, "gross_ret": gross})
        summary = df.groupby("qbin")["gross_ret"].agg(["mean", "count"])
        summary["mean_pct"] = summary["mean"] * 100
        print(f"\n[{side_name}] predicted-quality quintile -> mean realized GROSS return:")
        print(summary[["count", "mean_pct"]].to_string())
        spearman = pd.Series(pred).corr(pd.Series(gross), method="spearman")
        print(f"  Spearman(predicted_quality, realized_gross_return) = {spearman:.4f}")

    # ================= H_cost: gross vs net decomposition at the entry rule actually used =================
    print("\n=== H_cost: gross vs net decomposition, entry = argmax(long_q,short_q) above threshold ===")
    realized_long_gross = oos["long_ret"].to_numpy()
    realized_short_gross = oos["short_ret"].to_numpy()
    train_consistent_cost_long = FEE_COST + 0.003 * (oos["long_reason"] == "sl").to_numpy()
    train_consistent_cost_short = FEE_COST + 0.003 * (oos["short_reason"] == "sl").to_numpy()

    for thresh in [0.0, 0.002, 0.004, 0.006, 0.010]:
        take_long = pred_long >= thresh
        take_short = (pred_short >= thresh) & (pred_short > pred_long)
        take_long = take_long & ~take_short
        n_trades = int(take_long.sum() + take_short.sum())
        if n_trades == 0:
            continue
        gross = np.concatenate([realized_long_gross[take_long], realized_short_gross[take_short]])
        net_eval_cost = gross - COST_CONSERVATIVE
        net_train_cost = np.concatenate([
            realized_long_gross[take_long] - train_consistent_cost_long[take_long],
            realized_short_gross[take_short] - train_consistent_cost_short[take_short],
        ])
        print(f"  thresh={thresh:.3f} n={n_trades:5d}  "
              f"gross_mean={100*gross.mean():6.3f}%  "
              f"net_train_cost_mean={100*net_train_cost.mean():6.3f}%  "
              f"net_eval_cost(0.42%)_mean={100*net_eval_cost.mean():6.3f}%  "
              f"gap_from_cost_mismatch={100*(net_train_cost.mean()-net_eval_cost.mean()):6.3f}pp")


if __name__ == "__main__":
    main()
