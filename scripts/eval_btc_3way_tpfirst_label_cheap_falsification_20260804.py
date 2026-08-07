"""Cheapest falsification of a 3-way TP/SL/timeout classification label (the
label design behind the user's proposed 2-branch fusion net with Long/Short
TP-first heads), on the unified raw panel
(scripts/build_btc_unified_raw_panel_20260804.py).

Per user decision: timeout (neither barrier touched before horizon) gets its own
explicit class rather than being dropped (avoids the survivorship-bias failure mode
that the prior BTC v2 meta-labeling search's "terminal target" family hit -- see
docs/model_reports/btc_v2_direction_meta_20260716.md, 0/12544 candidates passed).

Per side, label in {sl, tp, timeout} via the SAME _reason_and_return function used
throughout this project's triple-barrier labeling (no new barrier-touch logic).
Trained as a 3-class LightGBM classifier per side; trading score =
P(tp) - P(sl) (timeout is treated as a genuine third outcome, not folded into
either side, so a model uncertain between tp/timeout naturally gets a smaller edge
than one confident in tp specifically). Same VAL/OOS split, cost model, and
pass/fail gate as every prior stage in this line.
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
OUT_CSV = ROOT / "tmp/btc_3way_tpfirst_cheap_falsification_20260804.csv"

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

CONFIGS = [
    ("h48qual_shape", 48, 1.2, 0.8),
    ("longhold_shape", 576, 2.0, 1.2),
]
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
STRIDE = 3
CLASS_TO_ID = {"sl": 0, "tp": 1, "timeout": 2}


def build_3way_labels(frame: pd.DataFrame, idxs: np.ndarray, horizon: int, tp_mult: float, sl_mult: float) -> pd.DataFrame:
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
        if long_reason not in CLASS_TO_ID or short_reason not in CLASS_TO_ID:
            continue
        rows.append({"i": i, "timestamp": ts.iloc[i], "long_ret": long_ret, "short_ret": short_ret,
                      "long_cls": CLASS_TO_ID[long_reason], "short_cls": CLASS_TO_ID[short_reason]})
    return pd.DataFrame(rows)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n = len(frame)
    print(f"unified raw panel: {n} rows, {len(feat_cols)} feature cols")

    all_results = []
    for name, horizon, tp_mult, sl_mult in CONFIGS:
        idxs = np.arange(0, n - horizon - 2, STRIDE)
        t0 = time.time()
        labels = build_3way_labels(frame, idxs, horizon, tp_mult, sl_mult)
        print(f"{name}: built {len(labels)} dense samples in {time.time()-t0:.1f}s")
        print(f"  long class dist: {labels['long_cls'].value_counts(normalize=True).to_dict()}")
        print(f"  short class dist: {labels['short_cls'].value_counts(normalize=True).to_dict()}")

        event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
        data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

        train = data[data["timestamp"] < VAL_START]
        val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
        oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
        print(f"  train={len(train)} val={len(val)} oos={len(oos)}")

        models = {}
        for side, target in [("long", "long_cls"), ("short", "short_cls")]:
            t0 = time.time()
            model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                        subsample=0.8, colsample_bytree=0.8, random_state=0,
                                        objective="multiclass", num_class=3, verbosity=-1)
            model.fit(train[feat_cols], train[target])
            models[side] = model
            print(f"  trained {side} in {time.time()-t0:.1f}s")

        for split_name, split in [("VAL", val), ("OOS", oos)]:
            proba_long = models["long"].predict_proba(split[feat_cols])
            proba_short = models["short"].predict_proba(split[feat_cols])
            score_long = proba_long[:, CLASS_TO_ID["tp"]] - proba_long[:, CLASS_TO_ID["sl"]]
            score_short = proba_short[:, CLASS_TO_ID["tp"]] - proba_short[:, CLASS_TO_ID["sl"]]
            realized_long = split["long_ret"].to_numpy() - COST_CONSERVATIVE
            realized_short = split["short_ret"].to_numpy() - COST_CONSERVATIVE
            for thresh in THRESHOLDS:
                take_long = score_long >= thresh
                take_short = (score_short >= thresh) & (score_short > score_long)
                take_long = take_long & ~take_short
                n_trades = int(take_long.sum() + take_short.sum())
                if n_trades == 0:
                    continue
                net = np.concatenate([realized_long[take_long], realized_short[take_short]])
                win = (net > 0).sum()
                all_results.append({
                    "config": name, "split": split_name, "thresh": thresh, "n_trades": n_trades,
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
    both = val_pos.merge(oos_pos, on=["config", "thresh"], suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)} ===")
    if len(both):
        print(both[["config", "thresh", "n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]]
              .sort_values("mean_net_pct_oos", ascending=False).to_string(index=False))
    else:
        print("(none) -- 3-way TP-first label gate FAILS. Do not build the proposed fusion architecture on this label.")


if __name__ == "__main__":
    main()
