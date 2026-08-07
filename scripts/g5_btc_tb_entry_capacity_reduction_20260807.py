"""G5 Layer 2 -- capacity-reduction test for the BTC triple-barrier entry model.

Design doc (docs/btc_tripbarrier_zigzag_architecture_design_20260807.md, Layer 2): with effective
sample size ~4,058 (not the 43,798 nominal), the shipped transformer's d_model=96/3-layer capacity
is almost certainly oversized. Start from a shallow/near-linear model and raise capacity, watching
where seed dispersion explodes, instead of assuming the transformer's capacity was ever validated
against this label's real information content.

Prior GBDT-vs-transformer comparisons on this repo (54-config sweep, 2026-08-06; G4c's 60-asset
depth-4 XGBoost, 2026-08-07) never combined a GBDT with G2's purge+embargo+uniqueness-weighting
hygiene -- they either predate that hygiene or apply it to the transformer only. This script is the
first GBDT run under that exact hygiene.

Uses build_dataset(window=1, ...) so purge/embargo/uniqueness weighting are IDENTICAL in
implementation to the transformer gates (G2/G5) -- only window collapses to a flat per-bar feature
vector, since XGBoost has no sequence structure. Backtest reuses the transformer gates' own
_fresh_entry_mask/_backtest/_summarize (same TP/SL vol basis, same account cost, same simulator)
so results are directly comparable to G2's D_purge_uniq baseline and G5's condition A/B.

Per design doc: model SELECTION across capacity levels uses seed-mean OOS trading metrics, not
VAL loss -- VAL loss at effective_n~4,058 is not a reliable ranking signal for this label
(see docs/btc_tripbarrier_zigzag_architecture_design_20260807.md Layer 2). Early stopping WITHIN a
single model's boosting rounds (against VAL logloss) is still ordinary regularization, not the
cross-config selection criterion the design doc warns about.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from g5_btc_tb_entry_zigzag_feature_retest_20260807 import (  # noqa: E402
    _backtest, _summarize, ACCOUNT_COST, TP_MULT, SL_MULT, CUMRET_BARS, VOL_LOOKBACK,
    HORIZON_BARS, ALWAYS_SHORT_OOS_GROSS_BPS,
)

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_flatsmooth_20260806.parquet"
SPAN_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_label_span_20260807.parquet"
OUT_DIR = ROOT / "tmp/btc_g5_capacity_reduction_20260807"
SOFT_COLS = ["trade_outcome_soft_cash", "trade_outcome_soft_long", "trade_outcome_soft_short"]
EMBARGO_BARS = 288
CASH_WEIGHT = 0.9  # matches G2/G5's TRAIN_CFG cash_weight, applied as a hard-label sample down-weight here


@dataclass(frozen=True)
class Capacity:
    name: str
    max_depth: int
    n_estimators: int  # upper cap; early stopping on VAL logloss picks the actual round


CAPACITY_LEVELS = (
    Capacity("L1_stumps_nearlinear", max_depth=1, n_estimators=300),
    Capacity("L2_shallow", max_depth=3, n_estimators=400),
    Capacity("L3_medium_g4c_depth4_ref", max_depth=4, n_estimators=500),
    Capacity("L4_deeper", max_depth=8, n_estimators=600),
)


def _flat_features(ds, split: str) -> np.ndarray:
    idx = ds.end_idx[split]
    return ds.feat_std[idx]  # window=1 -> feat_std IS the flat per-bar feature matrix


def _train_predict(cap: Capacity, seed: int, x_train, y_train, w_train, x_val, y_val, x_oos, device: str):
    model = xgb.XGBClassifier(
        max_depth=cap.max_depth, n_estimators=cap.n_estimators, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob", num_class=3,
        random_state=seed, tree_method="hist", device=device,
        early_stopping_rounds=50, eval_metric="mlogloss", n_jobs=0,
    )
    model.fit(x_train, y_train, sample_weight=w_train, eval_set=[(x_val, y_val)], verbose=False)
    best_iteration = getattr(model, "best_iteration", None)
    val_proba = model.predict_proba(x_val)
    oos_proba = model.predict_proba(x_oos)
    return model, val_proba, oos_proba, best_iteration


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="481937,1029384756,6271,88420193,3305577")
    p.add_argument("--levels", type=str, default=",".join(c.name for c in CAPACITY_LEVELS))
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < 5:
        raise ValueError("seed-diversity gate requires >=5 seeds")
    level_names = [s.strip() for s in args.levels.split(",")]
    levels = [c for c in CAPACITY_LEVELS if c.name in level_names]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset(
        window=1, train_stride=4, label_path=LABEL_PATH, hard_col="trade_outcome_action",
        soft_cols=SOFT_COLS, label_span_path=SPAN_PATH, purge=True,
        embargo_bars=EMBARGO_BARS, uniqueness_weights=True,
    )
    print(f"n_features={len(ds.feature_columns)} effective_sample_size={ds.hygiene['effective_sample_size']:.1f} "
          f"n_train={len(ds.end_idx['train'])} n_val={len(ds.end_idx['val'])} n_oos={len(ds.end_idx['oos'])}")

    x_train = _flat_features(ds, "train")
    x_val = _flat_features(ds, "val")
    x_oos = _flat_features(ds, "oos")
    y_train = ds.y_hard_all[ds.end_idx["train"]]
    y_val = ds.y_hard_all[ds.end_idx["val"]]
    w_uniq = ds.train_weight if ds.train_weight is not None else np.ones(len(y_train), dtype=np.float32)
    w_cash = np.where(y_train == 0, CASH_WEIGHT, 1.0).astype(np.float32)
    w_train = (w_uniq * w_cash).astype(np.float32)

    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_all, sl_all = TP_MULT * vol, SL_MULT * vol

    rows = []
    for cap in levels:
        for seed in seeds:
            model, val_proba, oos_proba, best_iter = _train_predict(
                cap, seed, x_train, y_train, w_train, x_val, y_val, x_oos, device
            )
            row = {"capacity": cap.name, "max_depth": cap.max_depth, "seed": seed, "best_iteration": best_iter}
            for split, proba in (("val", val_proba), ("oos", oos_proba)):
                pred = proba.argmax(axis=1)
                side_state = np.where(pred == 1, 1, np.where(pred == 2, -1, 0))
                bt = _summarize(_backtest(ds.end_idx[split], side_state, tp_all, sl_all, panel))
                row.update({f"{split}_{k}": v for k, v in bt.items()})
            rows.append(row)
            print(json.dumps(row, default=str))

    df = pd.DataFrame(rows)
    agg_cols = ["oos_win_rate", "oos_sum_ret_pct", "oos_gross_mean_ret_bps", "oos_n_trades", "oos_mdd_pct", "val_sum_ret_pct"]
    agg = df.groupby("capacity")[agg_cols].agg(["mean", "std"])

    print("\n=== capacity-level means (std), N=%d seeds each ===" % len(seeds))
    hdr = f"{'capacity':<28}{'oos_win%':>18}{'oos_gross_bps':>20}{'oos_sum%':>20}{'oos_trades_mean':>18}"
    print(hdr)
    print("-" * len(hdr))
    for cap in [c.name for c in levels]:
        if cap not in agg.index:
            continue
        def cell(col):
            return f"{agg.loc[cap, (col, 'mean')]:.2f} ({agg.loc[cap, (col, 'std')]:.2f})"
        print(f"{cap:<28}{cell('oos_win_rate'):>18}{cell('oos_gross_mean_ret_bps'):>20}"
              f"{cell('oos_sum_ret_pct'):>20}{agg.loc[cap, ('oos_n_trades', 'mean')]:>18.0f}")

    best_cap, best_t = None, float("-inf")
    for cap in [c.name for c in levels]:
        if cap not in agg.index:
            continue
        mean_v = agg.loc[cap, ("oos_gross_mean_ret_bps", "mean")]
        sd_v = agg.loc[cap, ("oos_gross_mean_ret_bps", "std")]
        t = (mean_v - ALWAYS_SHORT_OOS_GROSS_BPS) / (sd_v / np.sqrt(len(seeds))) if sd_v else float("nan")
        print(f"{cap}: seed-mean OOS gross={mean_v:.2f}bps sd={sd_v:.2f} vs always_short -> t={t:.2f} "
              f"({'PASS' if t >= 2 else 'FAIL'})")
        if not np.isnan(t) and t > best_t:
            best_cap, best_t = cap, t

    print(f"\nbest capacity level by seed-mean OOS gross vs always_short: {best_cap} (t={best_t:.2f})")

    payload = {"capacity_levels": [c.__dict__ for c in levels], "seeds": seeds,
               "effective_sample_size": ds.hygiene["effective_sample_size"],
               "cash_weight": CASH_WEIGHT, "per_seed": rows, "aggregate": json.loads(agg.to_json()),
               "always_short_oos_gross_bps_reference": ALWAYS_SHORT_OOS_GROSS_BPS,
               "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
               "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "capacity_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    df.to_csv(OUT_DIR / "capacity_per_seed.csv", index=False)
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
