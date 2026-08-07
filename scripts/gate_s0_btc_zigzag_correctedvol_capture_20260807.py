"""Step 0 of the ETH-Omega4.6.1-on-BTC-with-corrected-labels retrain plan (see
docs/model_contracts/ -- this gate precedes any TabM retrain): does a model trained on the
corrected-vol-basis zigzag label capture a positive fraction of ITS oracle ceiling, with real
VAL->OOS consistency across genuinely random seeds?

Cheap screen before the expensive TabM+risk-sidecar+router retrain cycle. Uses the exact same
XGBoost depth-4 recipe, purge/uniqueness-weighting logic, and capture-ratio metric as
scripts/gate_g4c_btc_panel_ceiling_capture_20260807.py, but single-asset (BTC only, causalfix_final
113-feature panel -- the live TabM feature set, not G4c's 60-asset 18-col panel) and trained against
data/splits/year_oos/btc_5m_zigzag_correctedvol_labels_20260806.parquet instead of the triple-barrier
trade_outcome label.

Label-span note: label spans are NOT re-derived here. The per-bar span in
data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet is a pure function of price +
the 12-bar/288-bar cumret vol basis + TP_MULT/SL_MULT/HORIZON_BARS (it races both a hypothetical
long and short entry at every bar regardless of which label file is consulted) -- since the zigzag
corrected-vol label uses the IDENTICAL TP_MULT=2.5/SL_MULT=1.2/HORIZON_BARS=288/vol basis, the same
span file applies unchanged. This also means effective sample size for zigzag training rows is
mechanically identical to G2's triple-barrier finding (~4,058 effective / 43,798 nominal at
stride-4) -- confirmed, not re-measured, by this script's printed effective_n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
SPAN_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_correctedvol_labels_20260806.parquet"
OUT_DIR = ROOT / "tmp/btc_gate_s0_zigzag_correctedvol_capture_20260807"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
ACCOUNT_COST = ROUNDTRIP_COST_RATE * MARGIN_FRACTION * LEVERAGE
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")
TRAIN_STRIDE = 4
# genuinely random draw, not a fixed-increment cluster -- see CLAUDE.md Seed-Diversity gate
SEEDS = [11, 137, 2029, 40507, 918273]

DROP_COLS = ["timestamp"]

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH).sort_values("timestamp").reset_index(drop=True)
    span_df = pd.read_parquet(SPAN_PATH).sort_values("timestamp").reset_index(drop=True)
    lab = pd.read_parquet(LABEL_PATH).sort_values("timestamp").reset_index(drop=True)
    ts = panel["timestamp"].to_numpy()
    if not ((span_df["timestamp"].to_numpy() == ts).all() and (lab["timestamp"].to_numpy() == ts).all()):
        raise RuntimeError("panel / span / zigzag-label timestamp misalignment")

    n = len(panel)
    close = panel["close"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c not in DROP_COLS]
    X = panel[feat_cols].to_numpy(dtype=np.float32)
    y = lab["zigzag_correctedvol_action"].to_numpy(dtype=np.int64)
    span = span_df["label_span_bars"].to_numpy(dtype=np.int64)
    label_end = np.arange(n, dtype=np.int64) + span

    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(max(close[0], 1e-9)))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    val_start_i = int(np.searchsorted(ts, np.datetime64(VAL_START)))
    oos_start_i = int(np.searchsorted(ts, np.datetime64(OOS_START)))
    base = np.flatnonzero(np.isfinite(tp_all) & np.isfinite(sl_all))
    tr = base[base < val_start_i]
    tr = tr[label_end[tr] < val_start_i]  # purge, matching G2/G4c
    tr_full = tr.copy()
    tr = tr[::TRAIN_STRIDE]
    va = base[(ts[base] >= np.datetime64(VAL_START)) & (ts[base] <= np.datetime64(VAL_END))]
    va = va[label_end[va] < oos_start_i]
    oo = base[(ts[base] >= np.datetime64(OOS_START)) & (ts[base] <= np.datetime64(OOS_END))]

    w = _uniqueness(tr, label_end, n)
    eff_n_stride4 = float(w.sum())
    w_full = _uniqueness(tr_full, label_end, n)
    eff_n_stride1 = float(w_full.sum())
    w = w / max(w.mean(), 1e-12)

    print(json.dumps({
        "n_train_nominal_stride4": int(len(tr)), "effective_n_stride4": eff_n_stride4,
        "n_train_nominal_stride1": int(len(tr_full)), "effective_n_stride1": eff_n_stride1,
        "span_median": float(np.median(span[tr_full])), "span_mean": float(span[tr_full].mean()),
        "note": "span is TP/SL-mechanics-derived, identical basis to G2's triple-barrier span -- "
                "expect effective_n close to G2's 4,058 (stride4) / 4,224 (stride1) regardless of label",
    }, indent=2))

    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w)
    dva = xgb.DMatrix(X[va], label=y[va])
    doo = xgb.DMatrix(X[oo], label=y[oo])

    side_true = np.where(y == 1, 1, np.where(y == 2, -1, 0))
    oracle = {s: _backtest(ix, side_true[ix], tp_all, sl_all, panel) for s, ix in (("val", va), ("oos", oo))}
    print("oracle:", json.dumps(oracle, indent=2))

    rows = []
    for seed in SEEDS:
        params = dict(XGB_PARAMS, seed=seed)
        bst = xgb.train(params, dtr, NUM_ROUNDS, evals=[(dva, "val")],
                        early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        row = {"seed": seed, "best_iter": int(bst.best_iteration), "n_train": int(len(tr))}
        for split, ix, dm in (("val", va, dva), ("oos", oo, doo)):
            pred = bst.predict(dm, iteration_range=(0, bst.best_iteration + 1)).argmax(axis=1)
            side = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
            bt = _backtest(ix, side, tp_all, sl_all, panel)
            orc = oracle[split]
            row[f"{split}_acc"] = float((pred == y[ix]).mean())
            if bt and orc and orc["gross_bps"] > 0:
                row.update({f"{split}_{m}": bt[m] for m in
                            ("n_trades", "win_rate", "gross_bps", "sum_ret_pct", "final_equity")})
                row[f"{split}_oracle_gross_bps"] = orc["gross_bps"]
                row[f"{split}_capture"] = bt["gross_bps"] / orc["gross_bps"]
                if bt["n_trades"] > 1 and np.isfinite(bt["gross_std_bps"]) and bt["gross_std_bps"] > 0:
                    row[f"{split}_t_gross"] = bt["gross_bps"] / (bt["gross_std_bps"] / np.sqrt(bt["n_trades"]))
        rows.append(row)
        print(f"seed={seed} best_iter={bst.best_iteration} "
              f"val_capture={row.get('val_capture', float('nan')):.4f} "
              f"oos_capture={row.get('oos_capture', float('nan')):.4f} "
              f"oos_gross_bps={row.get('oos_gross_bps', float('nan')):.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "s0_per_seed.csv", index=False)

    corr = df[["val_capture", "oos_capture"]].corr(method="spearman").iloc[0, 1] if "val_capture" in df and "oos_capture" in df else float("nan")
    summary = {
        "label": "zigzag_correctedvol_action", "panel": "causalfix_final_113col", "n_seeds": len(SEEDS),
        "seeds": SEEDS, "effective_n_stride4": eff_n_stride4, "effective_n_stride1": eff_n_stride1,
        "oracle_val_gross_bps": oracle["val"]["gross_bps"] if oracle["val"] else None,
        "oracle_oos_gross_bps": oracle["oos"]["gross_bps"] if oracle["oos"] else None,
        "oos_gross_bps_mean": float(df["oos_gross_bps"].mean()) if "oos_gross_bps" in df else None,
        "oos_gross_bps_std": float(df["oos_gross_bps"].std()) if "oos_gross_bps" in df else None,
        "oos_positive_seeds": int((df["oos_gross_bps"] > 0).sum()) if "oos_gross_bps" in df else None,
        "oos_capture_mean": float(df["oos_capture"].mean()) if "oos_capture" in df else None,
        "spearman_val_oos_capture": float(corr) if pd.notna(corr) else None,
    }
    (OUT_DIR / "s0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
