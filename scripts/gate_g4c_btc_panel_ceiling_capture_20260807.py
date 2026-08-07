"""Gate G4c -- does a model capture the same FRACTION of the oracle ceiling on high-ceiling alts
as it does on BTC?

G4b found BTC ranks 58/60 by oracle ceiling (44.3x OOS equity vs a panel median of 633x) purely
because barriers scale with each asset's own volatility while cost is fixed at 10bps. That makes
asset selection look like a free version of the "widen the barriers" lever -- but only under an
unverified assumption: that the model extracts the same proportion of a noisier alt's ceiling as it
does of BTC's. If alts are correspondingly less predictable, the bigger ceiling buys nothing.

Design: identical model, identical hyperparameters, identical barrier machinery, trained SEPARATELY
per asset over 5 seeds, on all 60 panel symbols. Per-asset training also sidesteps cross-asset
feature normalisation entirely (macd_hist and friends are price-scaled), so the only thing varying
between runs is the asset.

The metric is the capture ratio:

    capture = model gross bps per trade / oracle gross bps per trade

both measured on the same asset, same split, same fresh-entry backtest. G4c passes if capture on
high-ceiling assets is at least as large as on BTC -- in which case the bigger ceiling translates
into a bigger net edge -- and fails if capture falls in proportion to the ceiling.

Capacity is deliberately small (depth-4 XGBoost, early stopping) because G2 measured only ~4,076
effective samples per asset; a d_model=96 transformer is not a defensible choice at that sample
size and is not what this gate is testing.

IMPORTANT on interpreting agreement across the 60 assets: G4b measured 1.6-3.7 independent assets
in this panel, so a consistent sign across all 60 is worth roughly 2-4 independent confirmations,
not 60.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_DIR = ROOT / "data/panel/features"
LABEL_DIR = ROOT / "data/panel/tripbarrier"
OUT_DIR = ROOT / "tmp/btc_gate_g4c_ceiling_capture_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
ACCOUNT_COST = ROUNDTRIP_COST_RATE * MARGIN_FRACTION * LEVERAGE
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
TRAIN_STRIDE = 4
SEEDS = [11, 137, 2029, 40507, 918273]

# raw OHLC excluded: non-stationary and price-scaled, useless to a tree on a single asset and
# actively misleading if this feature set is later reused for a pooled model
DROP_COLS = ["timestamp", "symbol", "open", "high", "low", "close"]

XGB_PARAMS = dict(objective="multi:softprob", num_class=3, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=2.0,
                  tree_method="hist", eval_metric="mlogloss")
NUM_ROUNDS, EARLY_STOP = 400, 30


def _uniqueness(sample_idx, label_end, n_bars):
    starts = sample_idx + 1
    ends = np.minimum(label_end[sample_idx], n_bars - 1)
    keep = ends >= starts
    starts, ends = starts[keep], ends[keep]
    conc = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(conc, starts, 1.0)
    np.add.at(conc, ends + 1, -1.0)
    conc = np.cumsum(conc)[:n_bars]
    inv_c = np.where(conc > 0, 1.0 / np.maximum(conc, 1e-12), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv_c)])
    u = (prefix[ends + 1] - prefix[starts]) / np.maximum(ends - starts + 1, 1)
    out = np.zeros(len(sample_idx), dtype=np.float64)
    out[keep] = u
    return out


def _fresh_entry_mask(side_state):
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def _backtest(row_idx, side_state, tp_all, sl_all, fr):
    fresh = _fresh_entry_mask(side_state)
    idx, side = row_idx[fresh], side_state[fresh]
    tp, sl = tp_all[idx], sl_all[idx]
    ok = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[ok], side[ok], tp[ok], sl[ok]
    if len(idx) == 0:
        return None
    r = simulate_single_position(
        timestamps=fr["timestamp"], open_px=fr["open"].to_numpy(dtype=np.float64),
        high=fr["high"].to_numpy(dtype=np.float64), low=fr["low"].to_numpy(dtype=np.float64),
        close=fr["close"].to_numpy(dtype=np.float64), decision_indices=idx,
        scores=side.astype(np.float64), tp_moves=tp, sl_moves=sl, upper_threshold=0.0,
        lower_threshold=0.0, horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION,
        leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    if len(r.ledger) == 0:
        return None
    rets = r.ledger["trade_return"].to_numpy(dtype=np.float64)
    gross = rets + ACCOUNT_COST
    return {"n_trades": int(len(rets)), "win_rate": float((rets > 0).mean()),
            "gross_bps": float(gross.mean() * 10000.0),
            "gross_std_bps": float(gross.std(ddof=1) * 10000.0) if len(rets) > 1 else float("nan"),
            "sum_ret_pct": float(rets.sum() * 100.0), "final_equity": float(r.equity[-1])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(PANEL_DIR.glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    rows = []

    for k, f in enumerate(files, 1):
        sym = f.stem
        fr = pd.read_parquet(f).sort_values("timestamp").reset_index(drop=True)
        lb = pd.read_parquet(LABEL_DIR / f"{sym}.parquet").sort_values("timestamp").reset_index(drop=True)
        if not (fr["timestamp"].to_numpy() == lb["timestamp"].to_numpy()).all():
            raise RuntimeError(f"{sym}: feature/label timestamp misalignment")

        ts = fr["timestamp"].to_numpy()
        close = fr["close"].to_numpy(dtype=np.float64)
        n = len(fr)
        feat_cols = [c for c in fr.columns if c not in DROP_COLS]
        X = fr[feat_cols].to_numpy(dtype=np.float32)
        y = lb["trade_outcome_action"].to_numpy(dtype=np.int64)
        span = lb["label_span_bars"].to_numpy(dtype=np.int64)
        valid = lb["label_valid"].to_numpy(dtype=bool)
        label_end = np.arange(n, dtype=np.int64) + span

        log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(max(close[0], 1e-9)))
        cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
        vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
        tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

        val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
        oos_start_i = int(np.searchsorted(ts, np.datetime64(OOS_START)))
        base = np.flatnonzero(valid & np.isfinite(tp_all) & np.isfinite(sl_all))
        tr = base[(base < val_start_i)]
        tr = tr[label_end[tr] < val_start_i]          # purge (G2)
        tr = tr[::TRAIN_STRIDE]
        va = base[(ts[base] >= np.datetime64(VAL_START)) & (ts[base] <= np.datetime64(VAL_END))]
        va = va[label_end[va] < oos_start_i]
        oo = base[(ts[base] >= np.datetime64(OOS_START)) & (ts[base] <= np.datetime64(OOS_END))]
        if len(tr) < 500 or len(va) < 500 or len(oo) < 500:
            print(f"[{k}/{len(files)}] {sym}: SKIP (tr={len(tr)} va={len(va)} oo={len(oo)})", flush=True)
            continue

        w = _uniqueness(tr, label_end, n)
        w = w / max(w.mean(), 1e-12)
        dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w)
        dva = xgb.DMatrix(X[va], label=y[va])
        doo = xgb.DMatrix(X[oo], label=y[oo])

        side_true = np.where(y == 1, 1, np.where(y == 2, -1, 0))
        oracle = {s: _backtest(ix, side_true[ix], tp_all, sl_all, fr) for s, ix in (("val", va), ("oos", oo))}

        for seed in SEEDS:
            params = dict(XGB_PARAMS, seed=seed)
            bst = xgb.train(params, dtr, NUM_ROUNDS, evals=[(dva, "val")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            row = {"symbol": sym, "seed": seed, "best_iter": int(bst.best_iteration),
                   "n_train": int(len(tr)), "n_features": len(feat_cols)}
            for split, ix, dm in (("val", va, dva), ("oos", oo, doo)):
                pred = bst.predict(dm, iteration_range=(0, bst.best_iteration + 1)).argmax(axis=1)
                side = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
                bt = _backtest(ix, side, tp_all, sl_all, fr)
                orc = oracle[split]
                row[f"{split}_acc"] = float((pred == y[ix]).mean())
                if bt and orc and orc["gross_bps"] > 0:
                    row.update({f"{split}_{m}": bt[m] for m in
                                ("n_trades", "win_rate", "gross_bps", "sum_ret_pct", "final_equity")})
                    row[f"{split}_oracle_gross_bps"] = orc["gross_bps"]
                    row[f"{split}_capture"] = bt["gross_bps"] / orc["gross_bps"]
                    row[f"{split}_t_gross"] = bt["gross_bps"] / (bt["gross_std_bps"] / np.sqrt(bt["n_trades"]))
            rows.append(row)

        seed_rows = [r for r in rows if r["symbol"] == sym]
        cap = np.mean([r.get("oos_capture", np.nan) for r in seed_rows])
        print(f"[{k}/{len(files)}] {sym}: oracle_oos_gross={oracle['oos']['gross_bps']:.0f}bps "
              f"model_capture={cap:.4f} n_train={len(tr)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "g4c_per_seed.csv", index=False)

    agg = df.groupby("symbol").agg(
        oracle_gross=("oos_oracle_gross_bps", "first"),
        capture_mean=("oos_capture", "mean"), capture_std=("oos_capture", "std"),
        model_gross_mean=("oos_gross_bps", "mean"), model_gross_std=("oos_gross_bps", "std"),
        sum_ret_mean=("oos_sum_ret_pct", "mean"), t_gross_mean=("oos_t_gross", "mean"),
        val_capture_mean=("val_capture", "mean"), acc=("oos_acc", "mean"),
    ).reset_index().sort_values("oracle_gross", ascending=False)
    agg.to_csv(OUT_DIR / "g4c_per_asset.csv", index=False)

    agg["ceiling_tercile"] = pd.qcut(agg["oracle_gross"], 3, labels=["low", "mid", "high"])
    print("\n=== capture ratio by oracle-ceiling tercile (OOS, seed-mean per asset) ===")
    print(agg.groupby("ceiling_tercile", observed=True).agg(
        n=("symbol", "size"), oracle_gross=("oracle_gross", "median"),
        capture=("capture_mean", "median"), model_gross=("model_gross_mean", "median"),
        sum_ret=("sum_ret_mean", "median"), t=("t_gross_mean", "median")).round(4).to_string())

    print("\n=== reference assets ===")
    ref = agg[agg["symbol"].isin(["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
    print(ref[["symbol", "oracle_gross", "capture_mean", "capture_std", "model_gross_mean",
               "sum_ret_mean", "t_gross_mean"]].round(4).to_string(index=False))

    print("\n=== panel-wide ===")
    print(f"  assets: {len(agg)}")
    print(f"  capture ratio: median {agg['capture_mean'].median():.4f}  "
          f"mean {agg['capture_mean'].mean():.4f}  min {agg['capture_mean'].min():.4f}  "
          f"max {agg['capture_mean'].max():.4f}")
    print(f"  assets with positive seed-mean model gross edge: "
          f"{int((agg['model_gross_mean'] > 0).sum())}/{len(agg)}")
    print(f"  assets with positive seed-mean OOS sum_ret:      "
          f"{int((agg['sum_ret_mean'] > 0).sum())}/{len(agg)}")
    corr = agg[["oracle_gross", "capture_mean"]].corr(method="spearman").iloc[0, 1]
    print(f"  Spearman(oracle ceiling, capture ratio) = {corr:.3f}")
    print("  NOTE: G4b measured 1.6-3.7 independent assets here, so panel-wide agreement is worth")
    print("        ~2-4 independent confirmations, not 60.")

    (OUT_DIR / "g4c_summary.json").write_text(json.dumps({
        "xgb_params": XGB_PARAMS, "seeds": SEEDS, "train_stride": TRAIN_STRIDE,
        "n_assets": int(len(agg)),
        "capture_median": float(agg["capture_mean"].median()),
        "capture_mean": float(agg["capture_mean"].mean()),
        "assets_positive_gross": int((agg["model_gross_mean"] > 0).sum()),
        "assets_positive_sumret": int((agg["sum_ret_mean"] > 0).sum()),
        "spearman_ceiling_vs_capture": float(corr),
        "per_asset": agg.drop(columns=["ceiling_tercile"]).to_dict(orient="records"),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
