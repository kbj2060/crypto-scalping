"""
Re-do Stage 1 selection for the BTC long-hold trend-following CUSUM+LightGBM
architecture (see train_eval_btc_trendpersist_longhold_quality_20260804.py,
which tried one hand-picked config -- cusum_mult=3.0/horizon=576/tp_sl=2.0,1.2
-- and found no edge, VAL -0.51%/OOS -0.35% per trade).

Rather than pick one config off research_btc_trendpersist_longhold_labels_
20260804.py's hindsight-label ranking (which is not a real signal -- it always
shows ~100% win rate because it oracle-selects the winning side after the
fact), this trains the ACTUAL LightGBM quality classifier and measures
REALIZED VAL/OOS return from the model's own out-of-sample predictions across
a small grid of (cusum_mult, tp_mult, sl_mult) at fixed horizon=576 (Stage 1
already showed horizon 288/576/864 barely differ -- hindsight-optimal hold
converges to ~20-25 bars regardless of horizon budget). This is the real
Stage-1-redo: search across configs on the metric that matters (predicted-
quality-driven realized OOS return), not the hindsight-label metric.

Same architecture, features (trend-scan excluded, causal hurst/OU/regime-
persistence context), same causalfix_final frame as the single-config script.
Diagnostic/dev-score only, not Fresh-Forward validated.
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

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_trendpersist_longhold_quality_grid_20260804.csv"

TB_HORIZON = 576  # 2 days; Stage 1 showed this doesn't bind vs 288/864
TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
FEE_COST = 0.0007

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0  # 0.42%

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
}

CUSUM_MULTS = [1.5, 2.0, 2.5, 3.0, 4.0]
TP_SL_MULTS = [(1.2, 0.8), (1.5, 1.0), (2.0, 1.2), (2.5, 1.5)]
THRESHOLDS = [0.0, 0.002, 0.004, 0.006, 0.010]


def build_event_labels(frame: pd.DataFrame, events: np.ndarray, tp_mult: float, sl_mult: float) -> pd.DataFrame:
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
    atr = _atr_price_move(frame)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]

    events_by_mult = {}
    for cusum_mult in CUSUM_MULTS:
        ev = cusum_events(frame, atr, mult=cusum_mult)
        ev = ev[ev < len(frame) - TB_HORIZON - 2]
        events_by_mult[cusum_mult] = ev
        print(f"cusum_mult={cusum_mult}: {len(ev)} events total")

    results = []
    for cusum_mult in CUSUM_MULTS:
        events = events_by_mult[cusum_mult]
        for tp_mult, sl_mult in TP_SL_MULTS:
            labels = build_event_labels(frame, events, tp_mult, sl_mult)
            event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
            data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

            train = data[data["timestamp"] < VAL_START]
            val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
            oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
            if len(train) < 200 or len(val) < 20 or len(oos) < 20:
                continue

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
                for thresh in THRESHOLDS:
                    take_long = pred_long >= thresh
                    take_short = (pred_short >= thresh) & (pred_short > pred_long)
                    take_long = take_long & ~take_short
                    n_trades = int(take_long.sum() + take_short.sum())
                    if n_trades == 0:
                        continue
                    net = np.concatenate([realized_long[take_long], realized_short[take_short]])
                    win = (net > 0).sum()
                    results.append({
                        "cusum_mult": cusum_mult, "tp_mult": tp_mult, "sl_mult": sl_mult,
                        "split": split_name, "thresh": thresh, "n_trades": n_trades,
                        "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                        "sum_net_pct": 100 * net.sum(),
                    })
            print(f"done cusum_mult={cusum_mult} tp/sl={tp_mult}/{sl_mult} "
                  f"(train={len(train)} val={len(val)} oos={len(oos)})")

    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    # Find configs where BOTH VAL and OOS mean_net_pct are positive at the same threshold,
    # with a minimally meaningful sample size on both.
    val_pos = out[(out["split"] == "VAL") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    oos_pos = out[(out["split"] == "OOS") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    key = ["cusum_mult", "tp_mult", "sl_mult", "thresh"]
    both = val_pos.merge(oos_pos, on=key, suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)} ===")
    if len(both):
        cols = key + ["n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]
        print(both[cols].sort_values("mean_net_pct_oos", ascending=False).to_string(index=False))
    else:
        print("(none)")

    print("\n=== Best 15 OOS rows overall by mean_net_pct (n>=15) ===")
    oos_all = out[(out["split"] == "OOS") & (out["n_trades"] >= 15)].sort_values("mean_net_pct", ascending=False)
    print(oos_all.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
