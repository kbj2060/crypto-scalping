#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH conformal veto step 4: train the two conformal HGB regressors (full return, adverse
MAE) per component on the uniqueness-weighted episode labels (docs/experiments/
eth_candidate_conformal_veto_uniqueness_weights_20260816.md), calibrate a weighted validation-
residual quantile, and report N>=5-seed consistency before trusting any single run (this repo's
seed-diversity gate, [[tabm_hp_low_signal_pattern]]).

Train pool = 2025q1+q2+q3 pooled (per component, independent models -- contract Open Issue 3).
Calibration = val (weighted residual quantiles). OOS-Q1/OOS-Q2 not touched.

5 genuinely random seeds drawn from a documented meta-seed (np.random.default_rng(12345)), not a
fixed-increment cluster (this repo's seed-diversity policy, pipeline/architecture_workbench.py
validate_contract: seeds must not be base/base+5/base+10/...).

IMPORTANT correction made while building this: sklearn's HistGradientBoostingRegressor is
DETERMINISTIC given fixed data/hyperparameters when n_samples < 10000 (early_stopping defaults to
off below that size, and HGB does no row/feature subsampling unless configured) -- varying only
`random_state` therefore produced 5 byte-identical fits, a vacuous seed check. Fixed by having each
seed draw a WEIGHTED BOOTSTRAP resample of the training rows (probability proportional to
uniqueness_weight, sampled with replacement, same size as the pool) before fitting -- this directly
tests what we actually care about (how much do the correlation/quantile numbers move under resampling
given the thin effective N from the uniqueness-weighting step), which a deterministic-algorithm
random_state sweep would not have tested even if it had "worked".

Reports, per component: seed-to-seed mean/std of (a) weighted VAL correlation between predicted and
actual `full`/`adverse` (does the regressor have genuine out-of-window predictive power at all,
given this repo's exhaustive history of near-zero direction/quality skill from the same 102
features), (b) the calibration residual-quantile values themselves at a conservative grid
({0.50, 0.60, 0.70} for both components -- zig075's VAL effective N (~40) argues against BTC's
original 0.80 upper end; h48qual's (~50) isn't much more comfortable, so kept symmetric rather than
over-differentiating).

fresh_forward_bar_by_bar=true (labels/features are already causal per prior steps; this script only
fits/evaluates, no new simulation). No GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_hgb_train_20260816"

TRAIN_WINDOWS = ("2025q1", "2025q2", "2025q3")
CALIBRATION_WINDOW = "val"
COMPONENTS = ("h48qual", "zig075")
N_FEATURES = 102
QUANTILE_GRID = (0.50, 0.60, 0.70)
SEEDS = [int(s) for s in np.random.default_rng(12345).integers(1, 1_000_000, size=5)]


def log(msg: str) -> None:
    print(f"[candidate_conformal_veto_hgb_train] {msg}", flush=True)


def _load(window: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(LABEL_DIR / f"episode_labels_{window}_{name}.parquet")


def _pool(windows: tuple[str, ...], name: str) -> pd.DataFrame:
    return pd.concat([_load(w, name) for w in windows], ignore_index=True)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


def _weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    wa = np.average(a, weights=w)
    wb = np.average(b, weights=w)
    cov = np.average((a - wa) * (b - wb), weights=w)
    va = np.average((a - wa) ** 2, weights=w)
    vb = np.average((b - wb) ** 2, weights=w)
    denom = np.sqrt(va * vb)
    return float(cov / denom) if denom > 0 else 0.0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_cols = [f"f{i}" for i in range(N_FEATURES)]
    report: dict[str, Any] = {
        "design": "ETH conformal-downside-veto candidate HGB training -- pooled 2025q1-q3 train, val calibration, uniqueness sample_weight, N>=5 seeds.",
        "seeds": SEEDS,
        "seed_meta_rng": 12345,
        "quantile_grid": list(QUANTILE_GRID),
        "components": {},
    }

    for name in COMPONENTS:
        log(f"=== stage=train component={name} ===")
        train_df = _pool(TRAIN_WINDOWS, name)
        val_df = _load(CALIBRATION_WINDOW, name)
        x_train = train_df[feat_cols].to_numpy(dtype=np.float64)
        w_train = train_df["uniqueness_weight"].to_numpy(dtype=np.float64)
        y_full_train = train_df["full"].to_numpy(dtype=np.float64)
        y_adv_train = train_df["adverse"].to_numpy(dtype=np.float64)
        x_val = val_df[feat_cols].to_numpy(dtype=np.float64)
        w_val = val_df["uniqueness_weight"].to_numpy(dtype=np.float64)
        y_full_val = val_df["full"].to_numpy(dtype=np.float64)
        y_adv_val = val_df["adverse"].to_numpy(dtype=np.float64)

        per_seed: list[dict[str, Any]] = []
        for seed in SEEDS:
            rng = np.random.default_rng(int(seed))
            boot_idx = rng.choice(len(x_train), size=len(x_train), replace=True, p=w_train / w_train.sum())
            x_boot, y_full_boot, y_adv_boot = x_train[boot_idx], y_full_train[boot_idx], y_adv_train[boot_idx]

            model_full = HistGradientBoostingRegressor(loss="squared_error", max_iter=160, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.08, random_state=int(seed))
            model_adv = HistGradientBoostingRegressor(loss="squared_error", max_iter=160, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.08, random_state=int(seed) + 1)
            model_full.fit(x_boot, y_full_boot)
            model_adv.fit(x_boot, y_adv_boot)

            pred_full_val = model_full.predict(x_val)
            pred_adv_val = model_adv.predict(x_val)
            residual = np.abs(y_full_val - pred_full_val)

            corr_full = _weighted_corr(pred_full_val, y_full_val, w_val)
            corr_adv = _weighted_corr(pred_adv_val, y_adv_val, w_val)
            quantiles = {f"q{int(q*100)}": _weighted_quantile(residual, w_val, q) for q in QUANTILE_GRID}

            per_seed.append({
                "seed": int(seed),
                "val_weighted_corr_full": corr_full,
                "val_weighted_corr_adverse": corr_adv,
                "residual_quantiles": quantiles,
                "pred_full_val_wmean": float(np.average(pred_full_val, weights=w_val)),
                "pred_adverse_val_wmean": float(np.average(pred_adv_val, weights=w_val)),
            })
            log(f"  {name} seed={seed}: corr_full={corr_full:+.4f} corr_adverse={corr_adv:+.4f} "
                f"q50={quantiles['q50']:.4f} q60={quantiles['q60']:.4f} q70={quantiles['q70']:.4f}")

        corr_full_arr = np.array([r["val_weighted_corr_full"] for r in per_seed])
        corr_adv_arr = np.array([r["val_weighted_corr_adverse"] for r in per_seed])
        q_arrs = {f"q{int(q*100)}": np.array([r["residual_quantiles"][f"q{int(q*100)}"] for r in per_seed]) for q in QUANTILE_GRID}
        summary = {
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "weighted_n_train": float(w_train.sum()),
            "weighted_n_val": float(w_val.sum()),
            "corr_full_mean": float(corr_full_arr.mean()), "corr_full_std": float(corr_full_arr.std()),
            "corr_adverse_mean": float(corr_adv_arr.mean()), "corr_adverse_std": float(corr_adv_arr.std()),
            "residual_quantile_mean": {k: float(v.mean()) for k, v in q_arrs.items()},
            "residual_quantile_std": {k: float(v.std()) for k, v in q_arrs.items()},
        }
        log(f"  {name} SUMMARY: corr_full={summary['corr_full_mean']:+.4f}+-{summary['corr_full_std']:.4f} "
            f"corr_adverse={summary['corr_adverse_mean']:+.4f}+-{summary['corr_adverse_std']:.4f} "
            f"q60={summary['residual_quantile_mean']['q60']:.4f}+-{summary['residual_quantile_std']['q60']:.4f}")
        report["components"][name] = {"per_seed": per_seed, "summary": summary}

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
