"""
Same quality-classifier architecture as train_eval_btc_trendpersist_longhold_
quality_grid_20260804.py (LightGBM predicts long_q/short_q from causal
features at an event bar, threshold-swept on REAL out-of-sample predictions),
but swaps the event SELECTOR: instead of CUSUM cumulative-return crossing,
test zigzag pivots and Directional-Change(1h) confirmations as candidate entry
events. Outcome/quality computation (triple-barrier _reason_and_return, same
horizon/TP-SL grid, same cost model) stays identical across all three event
sources for a fair comparison -- only "where do we look for an entry" changes.

Context: zig075 (zigzag-based entry head) was already extensively tested
elsewhere and closed (project-btc-zigzag-dual-component-already-failed-
20260802) -- but that was a different quality-model/Leg-B architecture. This
retests zigzag pivots specifically as the EVENT SELECTOR feeding this
session's causal-feature LightGBM classifier (which CUSUM events already
failed on, 0/92 configs), per user request to re-widen the label-scheme
search rather than accept CUSUM as the only event definition.

Event sources:
  A) zigzag_pivot: first zigzag_quality_gate==1 bar of each accepted wave
     segment (v2 production defaults, same as compare_btc_label_schemes).
  B) directional_change_1h: intrinsic-time DC confirmations on the 1h close
     series (mult=1.5, same as compare_btc_label_schemes), mapped back to 5m.

Diagnostic/dev-score only (single in-sample->OOS split), not Fresh-Forward
validated.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move, _reason_and_return  # noqa: E402
from compare_btc_label_schemes_20260803 import directional_change_1h  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_altlabel_quality_grid_20260804.csv"

TB_HORIZON = 576
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

TP_SL_MULTS = [(1.2, 0.8), (1.5, 1.0), (2.0, 1.2), (2.5, 1.5)]
THRESHOLDS = [0.0, 0.002, 0.004, 0.006, 0.010]


def zigzag_confirmation_events(frame: pd.DataFrame, *, min_reversal_pct: float = 0.010,
                                max_reversal_pct: float = 0.018, atr_window: int = 14,
                                atr_multiplier: float = 1.0) -> np.ndarray:
    """
    CAUSAL zigzag trigger: fires at the bar a NEW pivot is actually confirmed
    (price crosses the reversal threshold from the running extreme), using
    only information available up to that bar. This deliberately does NOT use
    build_zigzag_action_labels()'s zigzag_quality_gate as the event selector --
    that gate is computed per-bar-inside-a-segment using high/low/close up to
    the segment's END pivot (idx_e), which for any bar i < idx_e is FUTURE
    information relative to i (idx_e is itself only confirmable later, at
    exactly this function's trigger bar). Using that gate to pick "is now a
    good time to enter" bars was found to inflate VAL/OOS win rate to ~80-90%
    (see run before this fix) -- a selection-bias lookahead, same failure
    category as the trend-scan bug (project-trendscan-lookahead-bug-found-
    fixed-20260804), just in the event-selector instead of a feature column.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr_pct = _atr_pct_local(frame, atr_window)
    n = len(close)

    def _threshold(i: int) -> float:
        atr = float(atr_pct[min(max(int(i), 0), n - 1)])
        return float(np.clip(max(min_reversal_pct, atr * atr_multiplier), min_reversal_pct, max_reversal_pct))

    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(close[0])
    confirm_events = []  # bar j where a pivot just got confirmed

    for i in range(1, n):
        price = float(close[i])
        if not np.isfinite(price):
            continue
        if trend == 0:
            if price < low_price:
                low_idx, low_price = i, price
            if price > high_price:
                high_idx, high_price = i, price
            thr = _threshold(i)
            if high_price / max(low_price, 1e-12) - 1.0 >= thr:
                if low_idx < high_idx:
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    trend = -1
                    low_idx, low_price = i, price
                confirm_events.append(i)
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            if high_price / max(price, 1e-12) - 1.0 >= _threshold(i):
                trend = -1
                low_idx, low_price = i, price
                confirm_events.append(i)
        else:
            if price < low_price:
                low_idx, low_price = i, price
            if price / max(low_price, 1e-12) - 1.0 >= _threshold(i):
                trend = 1
                high_idx, high_price = i, price
                confirm_events.append(i)
    return np.array(sorted(set(confirm_events)), dtype=np.int64)


def _atr_pct_local(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = pd.Series(tr).ewm(span=int(window), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


def dc_1h_events(frame: pd.DataFrame) -> np.ndarray:
    close = frame["close"].to_numpy(dtype=np.float64)
    overlay_1h = frame[["mtf1h_atr_pct"]].copy()
    overlay_1h["close"] = close
    hourly = overlay_1h.iloc[::12].reset_index(drop=True)
    dc = directional_change_1h(hourly, mult=1.5)
    idx_5m = (dc["i"].to_numpy() * 12)
    idx_5m = idx_5m[idx_5m < len(frame)]
    return np.array(sorted(set(idx_5m.tolist())), dtype=np.int64)


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


def run_source(name: str, frame: pd.DataFrame, events: np.ndarray, feat_cols: list[str]) -> list[dict]:
    print(f"\n### event source={name}: {len(events)} events total")
    results = []
    for tp_mult, sl_mult in TP_SL_MULTS:
        labels = build_event_labels(frame, events, tp_mult, sl_mult)
        if labels.empty:
            continue
        event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
        data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

        train = data[data["timestamp"] < VAL_START]
        val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
        oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
        if len(train) < 100 or len(val) < 15 or len(oos) < 15:
            print(f"  tp/sl={tp_mult}/{sl_mult}: too few events (train={len(train)} val={len(val)} oos={len(oos)}), skip")
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
                    "source": name, "tp_mult": tp_mult, "sl_mult": sl_mult,
                    "split": split_name, "thresh": thresh, "n_trades": n_trades,
                    "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                    "sum_net_pct": 100 * net.sum(),
                })
        print(f"  tp/sl={tp_mult}/{sl_mult} done (train={len(train)} val={len(val)} oos={len(oos)})")
    return results


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]

    all_results = []
    all_results += run_source("zigzag_confirmation", frame, zigzag_confirmation_events(frame), feat_cols)
    all_results += run_source("directional_change_1h", frame, dc_1h_events(frame), feat_cols)

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    val_pos = out[(out["split"] == "VAL") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    oos_pos = out[(out["split"] == "OOS") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    key = ["source", "tp_mult", "sl_mult", "thresh"]
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
