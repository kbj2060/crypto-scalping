"""BTC regime3-current classifier built with a Statistical Jump Model (k=3, lambda=4) instead of the
live 12-state sticky HMM, using the EXACT SAME wide24 feature panel and ADX/slope/BB "sensitive"
label mode as the live btc_regime3_current_hmm_sensitive_wide24_20260708 artifact (confirmed via its
joblib payload: feature_cols/label_mode/prefix_stem all match scripts/experiment_regime3_current_hmm_wide24_20260529.py's
wide24 + balancedish_adx16_slope15_bb012 config), so the swap isolates the model architecture
(HMM -> JM) and holds the feature engineering fixed.

Direct BTC mirror of scripts/build_eth_regime3_jm_lam4_20260809.py -- only the data source paths
differ (BTC's own year_oos split files instead of ETH's funding_clean_splits files); every modeling
choice (k, lambda, seed, temperature calibration, state-sort convention, soft-decode contract) is
copied unchanged so the HMM->JM swap is the only isolated variable, per the user's request to test
JM "with exactly the same logic as was done for ETH."

Mirrors the fit protocol: fit ONCE on the 2024 file only (train), then causally transform
2024/2025/2026 forward-only with the frozen jump-model centroids -- no future leakage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
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

LABEL_MODE = "balancedish_adx16_slope15_bb012"  # == live btc joblib's label_mode
PREFIX_STEM = "regime3_current_sensitive"
FEATURE_SET = "wide24"
K = 3
LAMBDA = 4.0
SEED = 7529  # identical to the ETH build -- only the data source changes
OUT_DIR = ROOT / "data/ensemble/supervised"
REPORT_PATH = ROOT / "data/ensemble/reports/btc_regime3_current_jm_lam4_20260809_report.json"
TAG = "jmlam4_20260809"

DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/btc_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "data/splits/year_oos/btc_features_2024.csv",
    ROOT / "data/splits/year_oos/btc_features_2025.csv",
    ROOT / "data/splits/year_oos/btc_features_2026.csv",
)


def causal_decode_soft(x: np.ndarray, mu: np.ndarray, lam: float, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    n, k = len(x), len(mu)
    cost = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    states = np.zeros(n, dtype=np.int8)
    probs = np.zeros((n, k), dtype=np.float64)
    V = cost[0].copy()
    states[0] = int(V.argmin())
    rel0 = V - V.min()
    probs[0] = np.exp(-rel0 / temperature)
    probs[0] /= probs[0].sum()
    for t in range(1, n):
        switch = V.min() + lam
        V = cost[t] + np.minimum(V, switch)
        V -= V.min()
        states[t] = int(V.argmin())
        probs[t] = np.exp(-V / temperature)
        probs[t] /= probs[t].sum()
    return states, probs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols = FEATURE_SETS[FEATURE_SET]
    # Same calibration as ETH: temperature=lambda matches the live HMM's confidence spread closely
    # enough at lambda=4 (verified there via direct sweep); reused unchanged here to isolate HMM->JM
    # as the only variable rather than re-deriving a BTC-specific temperature.
    temperature = LAMBDA

    print(f"[1/4] loading + feature-prep train source {DEFAULT_TRAIN_2024}")
    train_raw = _read(DEFAULT_TRAIN_2024)
    work = _with_features(train_raw, cols)

    print("[2/4] fitting JM (k=3, lambda=4) on 2024 only (RobustScaler fit on 2024, matches wide24 protocol)")
    x_full, _, scaler, medians = _fit_obs(work, work.iloc[:1].copy(), cols)
    mu = fit_jm(x_full, k=K, lam=LAMBDA, seed=SEED, n_init=5, n_iter=15)

    close = pd.to_numeric(work["close"], errors="coerce").to_numpy()
    fwd_ret = np.full(len(close), np.nan)
    fwd_ret[:-12] = np.log(close[12:]) - np.log(close[:-12])
    hard_states_fit, _ = causal_decode_soft(x_full, mu, LAMBDA, temperature)
    mean_ret_by_state = {}
    for s in range(K):
        m = hard_states_fit == s
        mean_ret_by_state[s] = float(np.nanmean(fwd_ret[m])) if m.any() else 0.0
    order = sorted(range(K), key=lambda s: mean_ret_by_state[s])  # bear, chop, bull
    bear_i, chop_i, bull_i = order
    print(f"  state mean fwd-12bar log-return by state: {mean_ret_by_state}  -> bear={bear_i} chop={chop_i} bull={bull_i}")

    y_full_labels = _labels(work, LABEL_MODE)
    _, state_prob_fit_soft = causal_decode_soft(x_full, mu, LAMBDA, temperature)
    state_class = _state_class_matrix(state_prob_fit_soft, y_full_labels)

    fit_eval_pred = np.argmax(_class_proba(state_prob_fit_soft, state_class), axis=1)
    from sklearn.metrics import balanced_accuracy_score
    fit_bal_acc = balanced_accuracy_score(y_full_labels, fit_eval_pred)
    print(f"[report] in-sample (2024 fit window) balanced_accuracy vs ADX/slope/BB label: {fit_bal_acc:.4f}")

    payload = {
        "model_id": f"btc_regime3_current_jm_{TAG}",
        "classes": CLASSES3,
        "label_mode": LABEL_MODE,
        "label_config": LABEL_CONFIGS[LABEL_MODE],
        "prefix_stem": PREFIX_STEM,
        "feature_set": FEATURE_SET,
        "feature_cols": cols,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "jm_mu": mu,
        "jm_lambda": LAMBDA,
        "jm_k": K,
        "jm_temperature": temperature,
        "jm_seed": SEED,
        "state_order_bear_chop_bull": [bear_i, chop_i, bull_i],
        "state_mean_fwd12bar_logret": mean_ret_by_state,
        "state_class_matrix": state_class,
        "fit_in_sample_balanced_accuracy": float(fit_bal_acc),
    }
    model_path = OUT_DIR / f"btc_regime3_current_jm_{TAG}_2024.joblib"
    joblib.dump(payload, model_path)
    print(f"[3/4] saved fitted JM payload -> {model_path}")

    report = {
        "model_id": payload["model_id"],
        "label_mode": LABEL_MODE,
        "prefix_stem": PREFIX_STEM,
        "feature_set": FEATURE_SET,
        "feature_cols": cols,
        "fit_source": str(DEFAULT_TRAIN_2024),
        "jm_lambda": LAMBDA,
        "jm_k": K,
        "jm_temperature": temperature,
        "state_order_bear_chop_bull": [bear_i, chop_i, bull_i],
        "state_mean_fwd12bar_logret": mean_ret_by_state,
        "fit_in_sample_balanced_accuracy": float(fit_bal_acc),
        "outputs": {},
    }

    print("[4/4] causal transform of 2024/2025/2026 with frozen JM centroids")
    for src in DEFAULT_TRANSFORMS:
        frame = _read(src)
        work_f = _with_features(frame, cols)
        med = medians
        x_raw = work_f[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        x_obs = scaler.transform(x_raw)
        _, state_prob = causal_decode_soft(x_obs, mu, LAMBDA, temperature)
        proba = _class_proba(state_prob, state_class)
        y = _labels(work_f, LABEL_MODE)

        out = pd.DataFrame({"timestamp": work_f["timestamp"].reset_index(drop=True)})
        prefix = f"{PREFIX_STEM}_wide24_"
        for i, name in enumerate(CLASSES3):
            out[f"{prefix}{name}_prob"] = proba[:, i]
        sp = np.sort(proba, axis=1)
        out[f"{prefix}confidence"] = sp[:, -1]
        out[f"{prefix}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / np.log(len(CLASSES3))
        out[f"{prefix}margin"] = sp[:, -1] - sp[:, -2]

        pred = np.argmax(proba, axis=1)
        from sklearn.metrics import balanced_accuracy_score as bas, accuracy_score as acs
        ev = {
            "rows": int(len(y)),
            "accuracy": float(acs(y, pred)),
            "balanced_accuracy": float(bas(y, pred)),
            "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        }
        print(f"  {src.name}: rows={ev['rows']} acc={ev['accuracy']:.4f} bal_acc={ev['balanced_accuracy']:.4f} flip_rate={ev['flip_rate']:.4f}")

        year = "".join(ch for ch in src.stem if ch.isdigit())[:4]
        out_path = OUT_DIR / f"btc_regime3_current_hmm_{TAG}_{year}_maskedname.csv"
        out.to_csv(out_path, index=False)
        print(f"    -> {out_path}")
        report["outputs"][src.name] = {"year": year, "eval_vs_adx_label": ev, "out_path": str(out_path)}

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
