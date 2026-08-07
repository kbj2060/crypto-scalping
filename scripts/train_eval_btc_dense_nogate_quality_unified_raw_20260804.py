"""Stage A of the BTC deep-feature-encoder plan: cheapest-falsification test of the
unified raw panel (causalfix_final + regime3-wide24 raw inputs + DVOL + on-chain,
see scripts/build_btc_unified_raw_panel_20260804.py).

Identical to scripts/train_eval_btc_dense_nogate_quality_20260804.py (same labels,
split dates, cost model, threshold sweep) except FRAME_PATH points at the unified
panel and EXCLUDE_COLS is extended to drop the on-chain raw `time` passthrough col
if present. This isolates "does the raw union carry tree-usable signal" from "does a
learned encoder help" -- see the approved plan at
C:\\Users\\kbj20\\.claude\\plans\\unified-napping-lighthouse.md.
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
OUT_CSV = ROOT / "tmp/btc_dense_nogate_quality_unified_raw_20260804.csv"

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
THRESHOLDS = [0.0, 0.002, 0.004, 0.006, 0.010]
STRIDE = 3


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

    all_results = []
    for name, horizon, tp_mult, sl_mult in CONFIGS:
        idxs = np.arange(0, n - horizon - 2, STRIDE)
        t0 = time.time()
        labels = build_dense_labels(frame, idxs, horizon, tp_mult, sl_mult)
        print(f"{name}: built {len(labels)} dense samples in {time.time()-t0:.1f}s")

        event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
        data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

        train = data[data["timestamp"] < VAL_START]
        val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
        oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
        print(f"  train={len(train)} val={len(val)} oos={len(oos)}")

        models = {}
        for side, target in [("long", "long_q"), ("short", "short_q")]:
            t0 = time.time()
            model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                       subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
            model.fit(train[feat_cols], train[target])
            models[side] = model
            print(f"  trained {side} in {time.time()-t0:.1f}s")

        imp = pd.Series(models["long"].feature_importances_, index=feat_cols).sort_values(ascending=False)
        print(f"  top 15 features (long): {list(imp.head(15).index)}")
        new_cols = ["state7_trend_score", "state7_trend_efficiency_48", "state7_directional_return_48",
                    "state7_volatility_state", "state7_sign_flip_rate_24", "state7_range_compression",
                    "state7_flow_alignment", "state12_log_return", "state12_garman_klass_vol",
                    "state12_net_taker_ratio", "state12_oi_change_rate", "state12_chop_index",
                    "breakout_strength", "dvol_btc", "dvol_eth", "dvol_btc_eth_spread",
                    "dvol_btc_pctrank_720h", "dvol_btc_roc_24h", "dvol_btc_roc_168h", "vol_risk_premium",
                    "mvrv", "mvrv_pctrank_90d", "net_exchange_flow_pct_supply", "sply_ex_roc_7d",
                    "active_addr_roc_7d", "active_addr_pctrank_90d"]
        new_col_rank = {c: int(imp.rank(ascending=False)[c]) for c in new_cols if c in imp.index}
        print(f"  rank of new (non-causalfix_final) cols out of {len(feat_cols)}: {sorted(new_col_rank.items(), key=lambda kv: kv[1])}")

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
        print("(none)")


if __name__ == "__main__":
    main()
