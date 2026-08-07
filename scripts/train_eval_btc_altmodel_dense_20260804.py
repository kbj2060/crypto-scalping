"""
Per user direction: microstructure/liquidation data (the other identified
next step) is deferred to next year, so try a genuinely different MODEL
FAMILY on the same causal feature set instead of more label/gating variants.
See project-btc-cusum-architecture-structural-redesign-closed-20260804 --
LightGBM (gradient-boosted trees) found zero signal across 4 event-gating
strategies and 20+ label geometries on the causalfix_final 114-col frame.

This reuses the dense (no event gate, stride-3, h48qual-shaped horizon/TP-SL)
label construction from train_eval_btc_dense_nogate_quality_20260804.py --
already the largest, most decisive sample (n~90k) -- and swaps the learner:
  - LightGBM (repeated here as the same-day reference point)
  - RandomForestRegressor (bagging, very different bias/variance than boosting)
  - ExtraTreesRegressor (more randomized splits, less prone to overfitting on
    weakly-informative tabular features)
  - MLPRegressor (neural net, different inductive bias entirely -- smooth
    function approximation vs axis-aligned tree splits)

If NONE of these find a real edge either, that's strong evidence the failure
is in the feature set / prediction target relationship, not the learner
architecture (already suggested by TabM failing on a different BTC line, see
project-btc-ceiling-and-eth-vs-others-structural-20260720).

Diagnostic/dev-score only (single in-sample->OOS split), not Fresh-Forward
validated.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move, _reason_and_return  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_altmodel_dense_20260804.csv"

TB_HORIZON, TB_TP_MULT, TB_SL_MULT = 48, 1.2, 0.8
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
THRESHOLDS = [0.0, 0.002, 0.004, 0.006, 0.010]
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


def make_model(kind: str):
    if kind == "lightgbm":
        return lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                      max_features=0.5, n_jobs=-1, random_state=0)
    if kind == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                    max_features=0.5, n_jobs=-1, random_state=0)
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                          n_iter_no_change=10, max_iter=200, random_state=0))
    raise ValueError(kind)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n = len(frame)

    idxs = np.arange(0, n - TB_HORIZON - 2, STRIDE)
    labels = build_dense_labels(frame, idxs)
    event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)
    data[feat_cols] = data[feat_cols].fillna(0.0)

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"train={len(train)} val={len(val)} oos={len(oos)}")

    all_results = []
    for kind in ["lightgbm", "random_forest", "extra_trees", "mlp"]:
        t0 = time.time()
        models = {}
        for side, target in [("long", "long_q"), ("short", "short_q")]:
            model = make_model(kind)
            model.fit(train[feat_cols], train[target])
            models[side] = model
        print(f"{kind}: trained both sides in {time.time()-t0:.1f}s")

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
                    "model": kind, "split": split_name, "thresh": thresh, "n_trades": n_trades,
                    "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                    "sum_net_pct": 100 * net.sum(),
                })
                print(f"  [{split_name}] thresh={thresh:.3f} n={n_trades:5d} win%={100*win/n_trades:5.1f} "
                      f"mean_net={100*net.mean():6.3f}% sum_net={100*net.sum():8.2f}%")

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    val_pos = out[(out["split"] == "VAL") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    oos_pos = out[(out["split"] == "OOS") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    both = val_pos.merge(oos_pos, on=["model", "thresh"], suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)} ===")
    if len(both):
        print(both[["model", "thresh", "n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]]
              .sort_values("mean_net_pct_oos", ascending=False).to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
