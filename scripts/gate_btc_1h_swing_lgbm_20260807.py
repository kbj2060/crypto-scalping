"""Cheap-falsification gate for the btc_1h_native_swing_entry line (contract
docs/experiments/btc_1h_swing_native_20260807.json).

ATR-scaled triple-barrier labels on 1h bars (96-bar horizon, explicit timeout class),
LightGBM over 5 genuinely random seeds with purge+embargo+uniqueness weighting,
candidate selection on VAL only, then ONE OOS read for the selected config.

Backtest: one position at a time, entry next bar open, TP/SL from entry using the
decision bar's ATR%, same-bar TP+SL touch resolved pessimistically as SL-first,
10bps round-trip cost. No saved ledgers, no future rows joined to decisions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/experiments/btc_1h_swing_native_20260807.json"
PREFLIGHT = ROOT / "docs/experiments/btc_1h_swing_native_20260807.preflight.json"
DATASET = ROOT / "data/splits/year_oos/btc_features_1h_swing_20260807.parquet"
OUT = ROOT / "docs/experiments/btc_1h_swing_cheap_gate_20260807_results.json"

HORIZON = 96
EMBARGO = 96  # extra bars dropped after the purge window at each split boundary
COST_ROUNDTRIP = 0.0010  # 10bps
SEEDS = [914237, 60481, 7754321, 283009, 51418]
BARRIER_GRID = [(2.0, 1.0), (2.0, 1.5), (3.0, 1.0), (3.0, 1.5)]
THRESHOLDS = [0.05, 0.10, 0.20]
SPLITS = {"train_end": "2025-08-31 23:00", "val_start": "2025-09-01", "val_end": "2025-12-31 23:00",
          "oos_start": "2026-01-01", "oos_end": "2026-03-31 23:00"}
NON_FEATURE_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_labels(frame: pd.DataFrame, tp_mult: float, sl_mult: float):
    """Long-side triple barrier: +1 up barrier first, -1 down barrier first, 0 timeout.
    Decision at close of bar i, entry at open of bar i+1, barriers scan bars i+1..i+HORIZON."""
    n = len(frame)
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    open_ = frame["open"].to_numpy()
    atr = frame["atr_pct_14h"].to_numpy()
    label = np.full(n, -9, dtype=np.int8)
    end_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n - 1):
        j_max = min(i + HORIZON, n - 1)
        if j_max <= i:
            continue
        entry = open_[i + 1]
        up = entry * (1 + tp_mult * atr[i])
        down = entry * (1 - sl_mult * atr[i])
        result, endj = 0, j_max
        for j in range(i + 1, j_max + 1):
            hit_up = high[j] >= up
            hit_down = low[j] <= down
            if hit_down:  # pessimistic: same-bar double touch counts as down-first
                result, endj = -1, j
                break
            if hit_up:
                result, endj = 1, j
                break
        label[i] = result
        end_idx[i] = endj
    valid = label != -9
    return label, end_idx, valid


def uniqueness_weights(end_idx: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = len(end_idx)
    concurrency = np.zeros(n + 1)
    for i in np.flatnonzero(valid):
        concurrency[i + 1] += 1
        concurrency[end_idx[i] + 1] -= 1 if end_idx[i] + 1 <= n else 0
    active = np.cumsum(concurrency[:-1])
    active = np.maximum(active, 1)
    weights = np.zeros(n)
    for i in np.flatnonzero(valid):
        weights[i] = np.mean(1.0 / active[i + 1:end_idx[i] + 1]) if end_idx[i] > i else 0.0
    return weights


def backtest(frame: pd.DataFrame, signal: np.ndarray, mask: np.ndarray,
             tp_mult: float, sl_mult: float, threshold: float) -> dict:
    """One position at a time. signal = P(up) - P(down) at each decision bar."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    open_ = frame["open"].to_numpy()
    close = frame["close"].to_numpy()
    atr = frame["atr_pct_14h"].to_numpy()
    n = len(frame)
    trades = []
    i = 0
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"trades": 0, "net_pnl": 0.0, "win_rate": float("nan"), "mdd": 0.0}
    pos = idx[0]
    busy_until = -1
    for i in idx:
        if i <= busy_until or i + 1 >= n:
            continue
        s = signal[i]
        if not np.isfinite(s) or abs(s) < threshold:
            continue
        side = 1 if s > 0 else -1
        entry = open_[i + 1]
        a = atr[i]
        if side == 1:
            tp_price, sl_price = entry * (1 + tp_mult * a), entry * (1 - sl_mult * a)
        else:
            tp_price, sl_price = entry * (1 - tp_mult * a), entry * (1 + sl_mult * a)
        j_max = min(i + HORIZON, n - 1)
        exit_ret, endj = None, j_max
        for j in range(i + 1, j_max + 1):
            if side == 1:
                hit_sl = low[j] <= sl_price
                hit_tp = high[j] >= tp_price
            else:
                hit_sl = high[j] >= sl_price
                hit_tp = low[j] <= tp_price
            if hit_sl:  # pessimistic ordering
                exit_ret, endj = -sl_mult * a, j
                break
            if hit_tp:
                exit_ret, endj = tp_mult * a, j
                break
        if exit_ret is None:
            exit_ret = side * (close[j_max] / entry - 1)
        trades.append(exit_ret - COST_ROUNDTRIP)
        busy_until = endj
    if not trades:
        return {"trades": 0, "net_pnl": 0.0, "win_rate": float("nan"), "mdd": 0.0}
    arr = np.array(trades)
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    mdd = float(np.min(equity - peak))
    return {"trades": int(len(arr)), "net_pnl": float(arr.sum()),
            "win_rate": float((arr > 0).mean()), "mdd": mdd}


def main() -> None:
    preflight = json.loads(PREFLIGHT.read_text())
    digest = sha256_file(DATASET)
    assert digest == preflight["dataset"]["sha256"], "dataset drifted from preflight; rerun preflight"
    frame = pd.read_parquet(DATASET)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    ts = frame["timestamp"]
    feature_cols = [c for c in frame.columns if c not in NON_FEATURE_COLS]
    X = frame[feature_cols].to_numpy(dtype=np.float64)

    train_end = pd.Timestamp(SPLITS["train_end"])
    val_mask = (ts >= SPLITS["val_start"]) & (ts <= SPLITS["val_end"])
    oos_mask = (ts >= SPLITS["oos_start"]) & (ts <= SPLITS["oos_end"])

    val_rows, per_config_signals = [], {}
    for tp_mult, sl_mult in BARRIER_GRID:
        label, end_idx, valid = build_labels(frame, tp_mult, sl_mult)
        weights = uniqueness_weights(end_idx, valid)
        # purge: training label windows must resolve before train_end; embargo on top
        purge_cut = train_end - pd.Timedelta(hours=HORIZON + EMBARGO)
        train_mask = (ts <= purge_cut).to_numpy() & valid
        y = (label + 1).astype(int)  # {-1,0,1} -> {0,1,2}
        signals = {}
        for seed in SEEDS:
            model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.05,
                num_leaves=31, min_child_samples=50, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, random_state=seed, verbose=-1)
            model.fit(X[train_mask], y[train_mask], sample_weight=weights[train_mask])
            proba = model.predict_proba(X)
            signals[seed] = proba[:, 2] - proba[:, 0]  # P(up) - P(down)
        per_config_signals[(tp_mult, sl_mult)] = signals
        for threshold in THRESHOLDS:
            seed_stats = {seed: backtest(frame, signals[seed], val_mask.to_numpy(), tp_mult, sl_mult, threshold)
                          for seed in SEEDS}
            nets = [s["net_pnl"] for s in seed_stats.values()]
            trades = [s["trades"] for s in seed_stats.values()]
            val_rows.append({
                "tp_mult": tp_mult, "sl_mult": sl_mult, "threshold": threshold,
                "val_seed_median_net": float(np.median(nets)),
                "val_seeds_positive": int(sum(v > 0 for v in nets)),
                "val_min_trades": int(min(trades)), "val_median_trades": int(np.median(trades)),
                "val_per_seed": {str(seed): seed_stats[seed] for seed in SEEDS}})
            print(f"VAL tp={tp_mult} sl={sl_mult} thr={threshold}: median_net={np.median(nets):+.4f} "
                  f"pos_seeds={sum(v > 0 for v in nets)}/5 min_trades={min(trades)}")

    passing = [r for r in val_rows
               if r["val_seed_median_net"] > 0 and r["val_seeds_positive"] >= 4 and r["val_min_trades"] >= 15]
    result = {
        "contract": "docs/experiments/btc_1h_swing_native_20260807.json",
        "dataset_sha256": digest,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "seeds": SEEDS, "cost_model": "10bps_roundtrip_conservative",
        "val_grid": val_rows, "val_passing_configs": len(passing),
    }
    if passing:
        best = max(passing, key=lambda r: r["val_seed_median_net"])
        signals = per_config_signals[(best["tp_mult"], best["sl_mult"])]
        oos_stats = {seed: backtest(frame, signals[seed], oos_mask.to_numpy(),
                                    best["tp_mult"], best["sl_mult"], best["threshold"])
                     for seed in SEEDS}
        oos_nets = [s["net_pnl"] for s in oos_stats.values()]
        result["selected_config"] = {k: best[k] for k in ("tp_mult", "sl_mult", "threshold",
                                                          "val_seed_median_net", "val_seeds_positive")}
        result["oos_read"] = {
            "per_seed": {str(seed): oos_stats[seed] for seed in SEEDS},
            "seed_median_net": float(np.median(oos_nets)),
            "seeds_positive": int(sum(v > 0 for v in oos_nets)),
            "gate_pass": bool(np.median(oos_nets) > 0 and sum(v > 0 for v in oos_nets) >= 4)}
        print(f"\nSELECTED tp={best['tp_mult']} sl={best['sl_mult']} thr={best['threshold']} "
              f"(VAL median {best['val_seed_median_net']:+.4f})")
        print(f"OOS: median_net={np.median(oos_nets):+.4f} pos_seeds={sum(v > 0 for v in oos_nets)}/5 "
              f"-> gate_pass={result['oos_read']['gate_pass']}")
    else:
        result["oos_read"] = None
        print("\nNo config passed VAL criteria; OOS was NOT read. Gate FAIL at VAL stage.")

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
