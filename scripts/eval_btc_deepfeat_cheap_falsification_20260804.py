"""Stage B (gate) of the BTC deep-feature-encoder plan: cheapest-falsification test
of the JEPA embeddings (scripts/pretrain_btc_deep_feature_encoder_20260804.py)
concatenated onto the unified raw panel, run through the SAME dense-nogate LightGBM
pipeline as Stage A. Pass bar (matching the project's existing cheap-gate
convention): >=1 config with VAL AND OOS both mean_net_pct>0, n_trades>=15 each side,
and a meaningful OOS improvement over the Stage A (raw-union-only) baseline.
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

PANEL_PATH = ROOT / "data/splits/year_oos/btc_unified_raw_panel_20260804.parquet"
EMB_PATH = ROOT / "data/splits/year_oos/btc_deepfeat_embeddings_20260804.parquet"
OUT_CSV = ROOT / "tmp/btc_deepfeat_cheap_falsification_20260804.csv"

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


def run(frame: pd.DataFrame, feat_cols: list[str], tag: str) -> pd.DataFrame:
    n = len(frame)
    all_results = []
    for name, horizon, tp_mult, sl_mult in CONFIGS:
        idxs = np.arange(0, n - horizon - 2, STRIDE)
        t0 = time.time()
        labels = build_dense_labels(frame, idxs, horizon, tp_mult, sl_mult)
        print(f"[{tag}] {name}: built {len(labels)} dense samples in {time.time()-t0:.1f}s")

        event_feats = frame.loc[labels["i"], ["timestamp"] + feat_cols].reset_index(drop=True)
        data = pd.concat([labels.drop(columns=["timestamp"]).reset_index(drop=True), event_feats], axis=1)

        train = data[data["timestamp"] < VAL_START]
        val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
        oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]

        models = {}
        for side, target in [("long", "long_q"), ("short", "short_q")]:
            model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                                       subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
            model.fit(train[feat_cols], train[target])
            models[side] = model

        imp = pd.Series(models["long"].feature_importances_, index=feat_cols).sort_values(ascending=False)
        deepfeat_ranks = {c: int(imp.rank(ascending=False)[c]) for c in feat_cols if c.startswith("deepfeat_")}
        print(f"  [{tag}/{name}] deepfeat_* ranks out of {len(feat_cols)}: {sorted(deepfeat_ranks.items(), key=lambda kv: kv[1])}")

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
                    "tag": tag, "config": name, "split": split_name, "thresh": thresh, "n_trades": n_trades,
                    "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                    "sum_net_pct": 100 * net.sum(),
                })
                print(f"  [{tag}/{split_name}] thresh={thresh:.3f} n={n_trades:5d} win%={100*win/n_trades:5.1f} "
                      f"mean_net={100*net.mean():6.3f}% sum_net={100*net.sum():8.2f}%")
    return pd.DataFrame(all_results)


def main():
    panel = pd.read_parquet(PANEL_PATH).sort_values("timestamp").reset_index(drop=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    emb = pd.read_parquet(EMB_PATH)
    emb["timestamp"] = pd.to_datetime(emb["timestamp"], utc=True)
    emb_cols = [c for c in emb.columns if c != "timestamp"]

    frame = panel.merge(emb, on="timestamp", how="inner")
    print(f"panel rows={len(panel)}, embeddings rows={len(emb)}, after inner-join={len(frame)}")

    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    results = run(frame, feat_cols, "with_deepfeat")

    results.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(results)} rows -> {OUT_CSV}")

    val_pos = results[(results["split"] == "VAL") & (results["mean_net_pct"] > 0) & (results["n_trades"] >= 15)]
    oos_pos = results[(results["split"] == "OOS") & (results["mean_net_pct"] > 0) & (results["n_trades"] >= 15)]
    both = val_pos.merge(oos_pos, on=["config", "thresh"], suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)} ===")
    if len(both):
        print(both[["config", "thresh", "n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]]
              .sort_values("mean_net_pct_oos", ascending=False).to_string(index=False))
    else:
        print("(none) -- Stage B gate FAILS. Do not proceed to Stage C (TabM integration / Fresh-Forward).")


if __name__ == "__main__":
    main()
