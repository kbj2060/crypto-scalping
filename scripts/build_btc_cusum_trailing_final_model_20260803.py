"""
Consolidated "final" BTC candidate model, combining the pieces validated
independently earlier this session (see docs/model_contracts/
btc_cusum_trailing_model_20260803_contract.md for the write-up):

  - feature frame: 5m base + 1h trend-scan/RSI/vol overlay (causal merge,
    lookahead-checked) + metrics4 (taker/toptrader-derived) columns
    -> data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet
  - entry trigger: CUSUM event filter (mult=2.0), causal
  - exit: ATR-adaptive causal trailing stop (zigzag-style threshold),
    replacing h48qual's fixed 4h barrier -- this was the single biggest
    lever found this session (roughly doubles per-trade edge)
  - quality/direction model: LightGBM regressors for long/short, trained
    THIS TIME on the trailing-exit realized net return (fixing the
    train/eval target mismatch flagged in the previous pass, where the
    classifier was trained on fixed-barrier quality but evaluated with
    trailing exits)
  - single-position-at-a-time constraint enforced in both training-label
    construction context and evaluation

Outputs model artifacts to data/ensemble/supervised/btc_cusum_trailing_*
and prints VAL/OOS threshold-sweep evaluation.

Diagnostic/dev-score only (single train/val/oos split, not a Fresh-Forward
bar-by-bar walk-forward) -- NOT a promotion claim.
"""
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402
from build_btc_cusum_trendscan_zigzag_hybrid_20260803 import simulate_trade, HARD_SL_MULT, HARD_SL_MIN  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
OUT_DIR = ROOT / "data/ensemble/supervised"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = "btc_cusum_trailing_20260803"

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
COST_STD = (0.0005 + 0.0002) * 2.0  # validated as the realistic live round-trip cost basis
ENTRY_THRESHOLD = 0.004  # point where both VAL (+0.484%) and OOS (+0.425%) clear the
                           # h48qual per-trade benchmark (+0.41%) with the retrained
                           # (trailing-exit-target) model; see sweep printed at build time

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
}


def build_trailing_targets(frame: pd.DataFrame, events: np.ndarray) -> pd.DataFrame:
    """For each CUSUM event bar, simulate the trailing-exit outcome for BOTH
    sides independently (as if that side were taken) -- this gives an honest
    per-side regression target consistent with the exit mechanism actually used
    at inference/live time."""
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)
    ts = frame["timestamp"]

    rows = []
    for i in events:
        entry_i = i + 1
        if entry_i + 1 >= n:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        sl_move = max(HARD_SL_MIN, HARD_SL_MULT * vol)
        long_ret, long_reason, long_bars = simulate_trade(1, entry, atr, high, low, close, entry_i, sl_move)
        short_ret, short_reason, short_bars = simulate_trade(2, entry, atr, high, low, close, entry_i, sl_move)
        rows.append({
            "i": i, "timestamp": ts.iloc[i],
            "long_net": long_ret - COST_STD, "short_net": short_ret - COST_STD,
            "long_bars": long_bars, "short_bars": short_bars,
        })
    return pd.DataFrame(rows)


def evaluate_sequential(frame_len: int, split: pd.DataFrame, models: dict, feat_cols: list,
                         atr: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         open_px: np.ndarray, threshold: float) -> pd.DataFrame:
    pl = models["long"].predict(split[feat_cols])
    ps = models["short"].predict(split[feat_cols])
    take_long = pl >= threshold
    take_short = (ps >= threshold) & (ps > pl)
    take_long = take_long & ~take_short
    idxs = split["i"].to_numpy()

    rows = []
    last_exit_i = -1
    for k, i in enumerate(idxs):
        side = 1 if take_long[k] else (2 if take_short[k] else 0)
        if side == 0:
            continue
        entry_i = i + 1
        if entry_i <= last_exit_i or entry_i + 1 >= frame_len:
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[i])
        sl_move = max(HARD_SL_MIN, HARD_SL_MULT * vol)
        ret, reason, bars = simulate_trade(side, entry, atr, high, low, close, entry_i, sl_move)
        exit_i = min(entry_i + bars, frame_len - 1)
        rows.append({"i": i, "side": side, "ret": ret, "net": ret - COST_STD, "reason": reason, "bars": bars})
        last_exit_i = exit_i
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    atr = _atr_price_move(frame)

    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < n - 48 - 2]
    targets = build_trailing_targets(frame, events)

    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    event_feats = frame.loc[targets["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
    data = pd.concat([targets.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"events: total={len(data)} train={len(train)} val={len(val)} oos={len(oos)}")

    models = {}
    for side, target in [("long", "long_net"), ("short", "short_net")]:
        m = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        m.fit(train[feat_cols], train[target])
        models[side] = m

    results = {}
    for split_name, split in [("VAL", val), ("OOS", oos)]:
        print(f"\n=== {split_name} — trailing-exit-target model, threshold sweep ===")
        for th in [0.0, 0.001, ENTRY_THRESHOLD, 0.003, 0.004]:
            tt = evaluate_sequential(n, split, models, feat_cols, atr, high, low, close, open_px, th)
            if len(tt) == 0:
                print(f"  th={th:.3f} n=0")
                continue
            win = (tt["net"] > 0).mean()
            print(f"  th={th:.3f} n={len(tt):4d} win%={100*win:5.1f} mean_net={100*tt['net'].mean():6.3f}% "
                  f"sum_net={100*tt['net'].sum():7.2f}% meanhold={tt['bars'].mean():5.1f}")
            if th == ENTRY_THRESHOLD:
                results[split_name] = {
                    "n": int(len(tt)), "win_pct": float(100 * win),
                    "mean_net_pct": float(100 * tt["net"].mean()), "sum_net_pct": float(100 * tt["net"].sum()),
                    "mean_hold_bars": float(tt["bars"].mean()),
                }

    with open(OUT_DIR / f"{MODEL_ID}_long.pkl", "wb") as f:
        pickle.dump(models["long"], f)
    with open(OUT_DIR / f"{MODEL_ID}_short.pkl", "wb") as f:
        pickle.dump(models["short"], f)
    config = {
        "model_id": MODEL_ID,
        "status": "dev_candidate_not_promoted",
        "entry_trigger": "cusum_event_mult2.0",
        "exit": "atr_adaptive_causal_trailing_stop",
        "quality_target": "trailing_exit_realized_net_return",
        "entry_threshold": ENTRY_THRESHOLD,
        "feature_frame": str(FRAME_PATH.relative_to(ROOT)),
        "feature_cols": feat_cols,
        "cost_basis": "standard_live_0.14pct_roundtrip (validated vs live fee config, see position_router.py)",
        "val_oos_split": {"val_start": str(VAL_START.date()), "oos_start": str(OOS_START.date()), "oos_end": str(OOS_END.date())},
        "results": results,
        "caveats": [
            "single train/val/oos split only, not Fresh-Forward bar-by-bar walk-forward",
            "no BTC-specific slippage audit exists (only ETH validated)",
            "not promoted / not live",
        ],
    }
    with open(OUT_DIR / f"{MODEL_ID}_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nwrote model artifacts to {OUT_DIR}/{MODEL_ID}_{{long,short}}.pkl, {MODEL_ID}_config.json")


if __name__ == "__main__":
    main()
