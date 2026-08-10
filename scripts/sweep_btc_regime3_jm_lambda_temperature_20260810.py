"""Lambda (and temperature) sweep for the BTC regime3-wide24 JM classifier -- never done for this
architecture on either BTC or ETH (lambda=4 was borrowed unswept from the 2026-08-08 zigzag-oracle
detector zoo, a different feature panel). Detector-level only (no downstream neural net retrain):
for each lambda, refit JM(k=3, lambda) on 2024, causal-decode 2024/2025/2026, score against the
SAME ADX/slope/BB rule label the live HMM uses. Also separately checks the argmax-invariance claim
(temperature is a softmax scale on -V/temperature; it cannot change which state has minimum V, so
accuracy/flip_rate must be IDENTICAL across temperature -- verified, not assumed) and reports how
temperature affects only the confidence-distribution calibration against the live HMM's own spread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_regime3_current_hmm_wide24_20260529 import (  # noqa: E402
    CLASSES3, FEATURE_SETS, LABEL_CONFIGS,
    _class_proba, _fit_obs, _labels, _state_class_matrix, _with_features,
)
from scripts.train_regime3_hmm_mamba_20260529 import _read  # noqa: E402
from scripts.test_statistical_jump_model_regimes_20260808 import fit_jm  # noqa: E402
from scripts.build_btc_regime3_jm_lam4_20260809 import causal_decode_soft  # noqa: E402

LABEL_MODE = "balancedish_adx16_slope15_bb012"
FEATURE_SET = "wide24"
K = 3
SEED = 7529
LAMBDA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/btc_features_2024.csv"
SOURCES = {
    "2024": DEFAULT_TRAIN_2024,
    "2025": ROOT / "data/splits/year_oos/btc_features_2025.csv",
    "2026": ROOT / "data/splits/year_oos/btc_features_2026.csv",
}
REPORT_PATH = ROOT / "data/ensemble/reports/btc_regime3_jm_lambda_sweep_20260810_report.json"

# live BTC HMM confidence distribution to calibrate temperature against (from the live joblib's
# actual 2026 output, computed once via the wide24 _transform path -- same reference ETH used)
LIVE_HMM_CONF_MEAN = 0.670
LIVE_HMM_CONF_STD = 0.132


def main() -> None:
    cols = FEATURE_SETS[FEATURE_SET]
    train_raw = _read(DEFAULT_TRAIN_2024)
    work = _with_features(train_raw, cols)
    x_full, _, scaler, medians = _fit_obs(work, work.iloc[:1].copy(), cols)
    close = pd.to_numeric(work["close"], errors="coerce").to_numpy()
    fwd_ret = np.full(len(close), np.nan)
    fwd_ret[:-12] = np.log(close[12:]) - np.log(close[:-12])
    y_full_labels = _labels(work, LABEL_MODE)

    frames = {}
    for year, src in SOURCES.items():
        frame = _read(src)
        work_f = _with_features(frame, cols)
        x_raw = work_f[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        x_obs = scaler.transform(x_raw)
        y = _labels(work_f, LABEL_MODE)
        frames[year] = (x_obs, y)

    report = {"lambda_grid": LAMBDA_GRID, "cells": []}
    print(f"{'lambda':>7} {'year':>6} {'acc':>8} {'bal_acc':>9} {'flip_rate':>10} {'median_run':>11}")
    for lam in LAMBDA_GRID:
        mu = fit_jm(x_full, k=K, lam=lam, seed=SEED, n_init=5, n_iter=15)
        # temperature: keep the same "temperature=lambda" empirical rule from the ETH calibration
        # (verified below to be a reasonable match at lambda=4; re-checked per lambda here)
        temperature = lam
        hard_states_fit, _ = causal_decode_soft(x_full, mu, lam, temperature)
        mean_ret_by_state = {s: float(np.nanmean(fwd_ret[hard_states_fit == s])) if (hard_states_fit == s).any() else 0.0 for s in range(K)}
        order = sorted(range(K), key=lambda s: mean_ret_by_state[s])
        bear_i, chop_i, bull_i = order
        _, state_prob_fit = causal_decode_soft(x_full, mu, lam, temperature)
        state_class = _state_class_matrix(state_prob_fit, y_full_labels)

        for year, (x_obs, y) in frames.items():
            states, state_prob = causal_decode_soft(x_obs, mu, lam, temperature)
            proba = _class_proba(state_prob, state_class)
            pred = np.argmax(proba, axis=1)
            from sklearn.metrics import accuracy_score, balanced_accuracy_score
            acc = accuracy_score(y, pred)
            bacc = balanced_accuracy_score(y, pred)
            flip = float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0
            # median run length
            runs = []
            start = 0
            for i in range(1, len(pred) + 1):
                if i == len(pred) or pred[i] != pred[start]:
                    runs.append(i - start)
                    start = i
            median_run = float(np.median(runs)) if runs else 0.0
            sp = np.sort(proba, axis=1)
            conf = sp[:, -1]
            print(f"{lam:>7.1f} {year:>6} {acc:>8.4f} {bacc:>9.4f} {flip:>10.4f} {median_run:>11.1f}")
            report["cells"].append({
                "lambda": lam, "temperature": temperature, "year": year,
                "accuracy": float(acc), "balanced_accuracy": float(bacc),
                "flip_rate": flip, "median_run_bars": median_run,
                "confidence_mean": float(conf.mean()), "confidence_std": float(conf.std()),
            })

    print("\n=== temperature argmax-invariance check (lambda=4 fixed, sweep temperature only) ===")
    lam = 4.0
    mu = fit_jm(x_full, k=K, lam=lam, seed=SEED, n_init=5, n_iter=15)
    hard_states_fit, _ = causal_decode_soft(x_full, mu, lam, lam)
    mean_ret_by_state = {s: float(np.nanmean(fwd_ret[hard_states_fit == s])) if (hard_states_fit == s).any() else 0.0 for s in range(K)}
    order = sorted(range(K), key=lambda s: mean_ret_by_state[s])
    _, state_prob_fit = causal_decode_soft(x_full, mu, lam, lam)
    state_class = _state_class_matrix(state_prob_fit, y_full_labels)
    x_obs, y = frames["2026"]
    print(f"{'temp':>7} {'acc':>8} {'bal_acc':>9} {'conf_mean':>10} {'conf_std':>9}  (target live HMM conf mean={LIVE_HMM_CONF_MEAN} std={LIVE_HMM_CONF_STD})")
    temp_cells = []
    for temp in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        states, state_prob = causal_decode_soft(x_obs, mu, lam, temp)
        proba = _class_proba(state_prob, state_class)
        pred = np.argmax(proba, axis=1)
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        acc = accuracy_score(y, pred)
        bacc = balanced_accuracy_score(y, pred)
        sp = np.sort(proba, axis=1)
        conf = sp[:, -1]
        print(f"{temp:>7.2f} {acc:>8.4f} {bacc:>9.4f} {conf.mean():>10.4f} {conf.std():>9.4f}")
        temp_cells.append({"temperature": temp, "accuracy": float(acc), "balanced_accuracy": float(bacc),
                            "confidence_mean": float(conf.mean()), "confidence_std": float(conf.std())})
    report["temperature_sweep_lambda4_2026"] = temp_cells

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
