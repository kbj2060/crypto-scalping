"""
Stage 2 of the BTC long-hold trend-following redesign (see
research_btc_trendpersist_longhold_labels_20260804.py for the label-scheme
grid search this config was picked from). That script's own "top" numbers are
NOT a trading-edge claim -- its build_tb_at_events() picks the hindsight-best
side (long_q>0 and long_q>=short_q, else short if short_q>0), so ~100% win
rate there is a labeling artifact (oracle side selection), same convention as
compare_btc_label_schemes_20260803.py. It only tells you the label definition
is well-formed and that horizon >~1 day rarely binds (median hindsight-optimal
hold ~20-25 bars even with a 2-3 day horizon budget).

This script trains an actual LightGBM quality classifier on causal features
(mtf1h_ts_t_value/ts_opt_L excluded -- closed line, see
project-trendscan-lookahead-bug-found-fixed-20260804) to PREDICT long_q/short_q
from information available at the event bar, then measures REALIZED return on
VAL/OOS using the model's own out-of-sample predictions (not hindsight labels)
-- this is the actual edge check.

Config chosen from the grid: cusum_mult=3.0 (rarer events than the original
2.0, matching BTC's ~half-frequency reversal structure), horizon=576 bars (2
days, generous vs the original 48-bar/4h), tp_mult/sl_mult=2.0/1.2 (wider than
original 1.2/0.8, matching BTC's ~30% lower ATR / slower breach speed).

Diagnostic/dev-score only (single in-sample->OOS split), not Fresh-Forward
validated per CLAUDE.md policy -- if this shows promise, next step is a full
bar-by-bar Fresh-Forward walk-forward before any promotion claim.
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
OUT_DIR = ROOT / "tmp/btc_trendpersist_longhold_quality_20260804"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUSUM_MULT = 3.0
TB_HORIZON = 576
TB_TP_MULT, TB_SL_MULT = 2.0, 1.2
TB_MIN_TP, TB_MIN_SL = 0.006, 0.004
FEE_COST = 0.0007

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0  # 0.42%

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    # closed line: even causal-fixed, trend-scan showed no edge, see
    # project-trendscan-lookahead-bug-found-fixed-20260804
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
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
        long_ret, long_reason, _, _, long_bars = _reason_and_return(
            side=1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        short_ret, short_reason, _, _, short_bars = _reason_and_return(
            side=-1, entry=entry, future_high=fh, future_low=fl, future_close=fc,
            tp_move=tp_move, sl_move=sl_move)
        long_q = long_ret - FEE_COST - 0.003 * int(long_reason == "sl")
        short_q = short_ret - FEE_COST - 0.003 * int(short_reason == "sl")
        rows.append({"i": i, "timestamp": ts.iloc[i], "long_ret": long_ret, "short_ret": short_ret,
                      "long_q": long_q, "short_q": short_q,
                      "long_bars": long_bars, "short_bars": short_bars})
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=CUSUM_MULT)
    events = events[events < len(frame) - TB_HORIZON - 2]

    labels = build_event_labels(frame, events)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"events: total={len(data)} train={len(train)} val={len(val)} oos={len(oos)}")
    print(f"mean hindsight hold (train): long={train['long_bars'].mean():.1f} bars short={train['short_bars'].mean():.1f} bars")

    models = {}
    for side, target in [("long", "long_q"), ("short", "short_q")]:
        model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        model.fit(train[feat_cols], train[target])
        models[side] = model
        model.booster_.save_model(str(OUT_DIR / f"btc_trendpersist_longhold_{side}.txt"))

    imp = pd.Series(models["long"].feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n=== Top 15 feature importances (long model) ===")
    print(imp.head(15).to_string())

    for split_name, split in [("VAL", val), ("OOS", oos)]:
        pred_long = models["long"].predict(split[feat_cols])
        pred_short = models["short"].predict(split[feat_cols])
        realized_long = split["long_ret"].to_numpy() - COST_CONSERVATIVE
        realized_short = split["short_ret"].to_numpy() - COST_CONSERVATIVE
        print(f"\n=== {split_name} (n={len(split)}) -- threshold sweep on PREDICTED quality (real OOS check) ===")
        for thresh in [0.0, 0.002, 0.004, 0.006, 0.010, 0.015, 0.020]:
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
