#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime_pred_moe_20260517 import (  # noqa: E402
    CLEAN_PREFIX,
    CLASSES,
    CLASS_TO_ID,
    DEFAULT_CLEAN_2024,
    DEFAULT_CLEAN_2025,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    PRED_PREFIX,
    SELECTED_CLEAN_FEATURES,
    _eval_report,
    _future_path_frame,
    _json_default,
    _label_thresholds,
    _labels,
    _merge_clean,
    _output_frame,
    _predicted_path_diagnostics,
    _read,
)


MODEL_ID = "regime_pred_moe_hmm_20260517"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime_pred_moe_hmm_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime_pred_moe_hmm_20260517_report.json"

NON_FEATURES = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "m7_entry_long_price",
    "m7_entry_short_price",
    "m7_tp_price",
    "m7_sl_price",
}
FORBIDDEN_EXACT = {
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
    "regime_trending",
    "regime_break",
    "cvp_regime",
    f"{CLEAN_PREFIX}risk_off_prob",
    f"{CLEAN_PREFIX}transition_risk",
    f"{CLEAN_PREFIX}bull_prob",
    f"{CLEAN_PREFIX}bear_prob",
    f"{CLEAN_PREFIX}chop_prob",
    f"{CLEAN_PREFIX}whipsaw_prob",
    f"{CLEAN_PREFIX}normal_prob",
    f"{CLEAN_PREFIX}state_code",
    f"{CLEAN_PREFIX}cluster",
}
FORBIDDEN_FRAGMENTS = (
    "future",
    "target",
    "label",
    "realized",
    "trade_pnl",
    "cash_after",
    "legacy",
    "hdb",
    "hmm_",
)


@dataclass
class GaussianHMMDiag:
    n_states: int
    n_iter: int
    seed: int
    min_var: float = 1e-4
    sticky: float = 0.92

    def __post_init__(self) -> None:
        self.pi_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.log_likelihood_: list[float] = []

    @staticmethod
    def _logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
        m = np.max(a, axis=axis, keepdims=True)
        out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out

    def _init_params(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n = len(x)
        q = np.linspace(0.05, 0.95, self.n_states)
        score = x[:, 0] if x.shape[1] else np.arange(n, dtype=float)
        order = np.argsort(score)
        centers = []
        for frac in q:
            centers.append(x[order[int(np.clip(frac * (n - 1), 0, n - 1))]])
        self.mu_ = np.asarray(centers, dtype=np.float64)
        global_var = np.var(x, axis=0).clip(min=self.min_var)
        self.var_ = np.tile(global_var[None, :], (self.n_states, 1))
        self.pi_ = np.ones(self.n_states, dtype=np.float64) / self.n_states
        self.A_ = np.full((self.n_states, self.n_states), (1.0 - self.sticky) / max(self.n_states - 1, 1), dtype=np.float64)
        np.fill_diagonal(self.A_, self.sticky)
        self.A_ += rng.random(self.A_.shape) * 1e-4
        self.A_ /= self.A_.sum(axis=1, keepdims=True)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        assert self.mu_ is not None and self.var_ is not None
        diff = x[:, None, :] - self.mu_[None, :, :]
        var = np.maximum(self.var_, self.min_var)
        return -0.5 * np.sum((diff * diff) / var[None, :, :] + np.log(2.0 * np.pi * var[None, :, :]), axis=2)

    def _forward_backward(self, log_emit: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        assert self.pi_ is not None and self.A_ is not None
        n = log_emit.shape[0]
        log_a = np.zeros_like(log_emit)
        log_b = np.zeros_like(log_emit)
        log_trans = np.log(self.A_ + 1e-300)
        log_a[0] = np.log(self.pi_ + 1e-300) + log_emit[0]
        for t in range(1, n):
            log_a[t] = log_emit[t] + self._logsumexp(log_a[t - 1][:, None] + log_trans, axis=0)
        for t in range(n - 2, -1, -1):
            log_b[t] = self._logsumexp(log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :], axis=1)
        ll = float(self._logsumexp(log_a[-1], axis=0))
        log_gamma = log_a + log_b - ll
        gamma = np.exp(log_gamma)
        return gamma, log_a, ll

    def fit(self, x: np.ndarray) -> "GaussianHMMDiag":
        x = np.asarray(x, dtype=np.float64)
        self._init_params(x)
        assert self.pi_ is not None and self.A_ is not None
        prev_ll = -np.inf
        for _ in range(int(self.n_iter)):
            log_emit = self._log_emission(x)
            gamma, log_a, ll = self._forward_backward(log_emit)
            log_b = np.zeros_like(log_emit)
            log_trans = np.log(self.A_ + 1e-300)
            for t in range(len(x) - 2, -1, -1):
                log_b[t] = self._logsumexp(log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :], axis=1)
            log_xi_sum = np.full((self.n_states, self.n_states), -np.inf, dtype=np.float64)
            for t in range(len(x) - 1):
                lx = log_a[t][:, None] + log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :] - ll
                log_xi_sum = np.logaddexp(log_xi_sum, lx)
            xi_sum = np.exp(log_xi_sum)
            self.pi_ = gamma[0] / np.clip(gamma[0].sum(), 1e-300, None)
            self.A_ = xi_sum / np.clip(xi_sum.sum(axis=1, keepdims=True), 1e-300, None)
            self.A_ = 0.02 / max(self.n_states - 1, 1) + 0.98 * self.A_
            self.A_ /= self.A_.sum(axis=1, keepdims=True)
            wsum = gamma.sum(axis=0) + 1e-300
            self.mu_ = (gamma.T @ x) / wsum[:, None]
            for s in range(self.n_states):
                diff = x - self.mu_[s]
                self.var_[s] = ((gamma[:, s][:, None] * diff * diff).sum(axis=0) / wsum[s]).clip(min=self.min_var)
            self.log_likelihood_.append(ll)
            if abs(ll - prev_ll) < 1e-4 * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        return self

    def posterior(self, x: np.ndarray) -> np.ndarray:
        log_emit = self._log_emission(np.asarray(x, dtype=np.float64))
        gamma, _, _ = self._forward_backward(log_emit)
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)
        return gamma

    def one_step_posterior(self, x: np.ndarray) -> np.ndarray:
        assert self.A_ is not None
        gamma = self.posterior(x)
        pred = gamma @ self.A_
        pred /= np.clip(pred.sum(axis=1, keepdims=True), 1e-300, None)
        return pred


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _add_router_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    trend1 = _num(out, "mtf_trend_1h").fillna(0.0)
    trend4 = _num(out, "mtf_trend_4h").fillna(0.0)
    factor_flow = _num(out, f"{CLEAN_PREFIX}factor_flow").fillna(0.0)
    factor_vol = _num(out, f"{CLEAN_PREFIX}factor_vol").fillna(0.0)
    factor_liq = _num(out, f"{CLEAN_PREFIX}factor_liquidity").fillna(0.0)
    trend_bias = _num(out, f"{CLEAN_PREFIX}trend_bias").fillna(0.0)
    cluster_entropy = _num(out, f"{CLEAN_PREFIX}cluster_entropy").fillna(0.0)
    net_taker = _num(out, "net_taker_ratio").fillna(0.0)
    smart_flow = _num(out, "smart_money_flow").fillna(0.0)
    breakout = _num(out, "breakout_strength").fillna(0.0)
    out["router_trend_abs"] = trend1.abs() + trend4.abs()
    out["router_trend_agreement"] = ((np.sign(trend1) * np.sign(trend4)) > 0).astype(float)
    out["router_flow_alignment"] = np.sign(trend_bias) * np.sign(net_taker + smart_flow)
    out["router_vol_liq_stress"] = factor_vol * factor_liq
    out["router_breakout_flow"] = breakout * factor_flow
    out["router_whipsaw_pressure"] = factor_vol * cluster_entropy
    out["router_trend_flow"] = trend_bias * factor_flow
    return out


def _is_feature(col: str) -> bool:
    lower = col.lower()
    if col in NON_FEATURES or col in FORBIDDEN_EXACT:
        return False
    if lower.startswith("_") or any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
        return False
    if lower.startswith(CLEAN_PREFIX) and col not in SELECTED_CLEAN_FEATURES:
        return False
    if lower.startswith(PRED_PREFIX):
        return False
    if "regime" in lower and not lower.startswith(CLEAN_PREFIX):
        return False
    return True


def _feature_cols(train: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    common = set(train.columns) & set(pred.columns)
    cols: list[str] = []
    for col in sorted(common):
        if not _is_feature(col):
            continue
        try:
            if pd.to_numeric(train[col], errors="coerce").notna().any() or pd.to_numeric(pred[col], errors="coerce").notna().any():
                cols.append(str(col))
        except Exception:
            continue
    return cols


def _matrix(frame: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> pd.DataFrame:
    out = pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)
    if medians is not None:
        out = out.fillna(medians).fillna(0.0)
    return out


def _state_class_matrix(state_prob: np.ndarray, y: np.ndarray, smoothing: float = 0.05) -> np.ndarray:
    n_states = state_prob.shape[1]
    mat = np.full((n_states, len(CLASSES)), float(smoothing), dtype=np.float64)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    for cls in range(len(CLASSES)):
        # Balance the hidden-state to class map so majority labels do not force
        # every state to route to the largest class.
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(counts[cls], 1.0)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba_from_states(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    out = state_prob @ state_class
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out


def _fit_transform_obs(
    train_x: pd.DataFrame,
    pred_x: pd.DataFrame,
    *,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, RobustScaler, PCA]:
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    x_train_s = scaler.fit_transform(train_x)
    pca = PCA(n_components=min(int(n_components), train_x.shape[1]), whiten=True, random_state=int(seed))
    train_obs = pca.fit_transform(x_train_s)
    pred_obs = pca.transform(scaler.transform(pred_x))
    return train_obs, pred_obs, scaler, pca


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 5-class MoE regime features with a Gaussian HMM.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--predict-2025", type=Path, default=DEFAULT_PREDICT_2025)
    parser.add_argument("--clean-2024", type=Path, default=DEFAULT_CLEAN_2024)
    parser.add_argument("--clean-2025", type=Path, default=DEFAULT_CLEAN_2025)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--n-iter", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7517)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train_raw = _add_router_features(_merge_clean(_read(args.train_2024), args.clean_2024))
    pred_raw = _add_router_features(_merge_clean(_read(args.predict_2025), args.clean_2025))

    val_start = pd.Timestamp(args.val_start)
    raw_ts = pd.to_datetime(train_raw["timestamp"])
    raw_train_mask = raw_ts < val_start
    threshold_path = _future_path_frame(train_raw.loc[raw_train_mask].copy(), int(args.horizon))
    train_only_thresholds = _label_thresholds(threshold_path)
    val_label_frame, val_label_meta = _labels(train_raw, int(args.horizon), thresholds=train_only_thresholds)
    val_labeled = train_raw.loc[val_label_frame.index].copy().join(val_label_frame[["_label_name", "_label_id"]])
    cols = _feature_cols(val_labeled, pred_raw)
    if len(cols) < 10:
        raise ValueError(f"not enough feature columns: {len(cols)}")

    ts = pd.to_datetime(val_labeled["timestamp"])
    train_mask = ts < val_start
    val_mask = ts >= val_start
    x_train_raw = _matrix(val_labeled.loc[train_mask], cols)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_val = _matrix(val_labeled.loc[val_mask], cols, medians)
    y_train = val_labeled.loc[train_mask, "_label_id"].astype(int).to_numpy()
    y_val = val_labeled.loc[val_mask, "_label_id"].astype(int).to_numpy()
    train_obs, val_obs, val_scaler, val_pca = _fit_transform_obs(x_train, x_val, n_components=int(args.pca_components), seed=int(args.seed))
    val_hmm = GaussianHMMDiag(n_states=int(args.states), n_iter=int(args.n_iter), seed=int(args.seed)).fit(train_obs)
    train_state = val_hmm.one_step_posterior(train_obs)
    val_state = val_hmm.one_step_posterior(val_obs)
    state_class_val = _state_class_matrix(train_state, y_train)
    val_proba = _class_proba_from_states(val_state, state_class_val)

    full_label_frame, full_label_meta = _labels(train_raw, int(args.horizon))
    full_labeled = train_raw.loc[full_label_frame.index].copy().join(full_label_frame[["_label_name", "_label_id"]])
    x_full_raw = _matrix(full_labeled, cols)
    full_medians = x_full_raw.median(numeric_only=True).fillna(0.0)
    x_full = x_full_raw.fillna(full_medians).fillna(0.0)
    y_full = full_labeled["_label_id"].astype(int).to_numpy()
    pred_x = _matrix(pred_raw, cols, full_medians)
    full_obs, pred_obs, scaler, pca = _fit_transform_obs(x_full, pred_x, n_components=int(args.pca_components), seed=int(args.seed) + 101)
    final_hmm = GaussianHMMDiag(n_states=int(args.states), n_iter=int(args.n_iter), seed=int(args.seed) + 101).fit(full_obs)
    full_state = final_hmm.one_step_posterior(full_obs)
    pred_state = final_hmm.one_step_posterior(pred_obs)
    state_class = _state_class_matrix(full_state, y_full)
    pred_proba = _class_proba_from_states(pred_state, state_class)
    full_proba = _class_proba_from_states(full_state, state_class)

    pred_output = _output_frame(pred_raw["timestamp"], pred_proba)
    train_output = _output_frame(full_labeled["timestamp"], full_proba)
    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime_pred_hmm_moe.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime_pred_hmm_moe.csv"
    model_path = args.out_dir / "regime_pred_hmm_moe_2024.joblib"
    pred_output.to_csv(pred_sidecar, index=False)
    train_output.to_csv(train_sidecar, index=False)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES,
            "feature_cols": cols,
            "feature_medians": full_medians.to_dict(),
            "scaler": scaler,
            "pca": pca,
            "hmm": final_hmm,
            "state_class_matrix": state_class,
            "horizon": int(args.horizon),
            "states": int(args.states),
            "pca_components": int(args.pca_components),
        },
        model_path,
    )
    report = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "train_source": str(args.train_2024),
        "predict_source": str(args.predict_2025),
        "clean_2024": str(args.clean_2024),
        "clean_2025": str(args.clean_2025),
        "horizon_bars": int(args.horizon),
        "classes": CLASSES,
        "states": int(args.states),
        "pca_components": int(args.pca_components),
        "validation_label_meta": {
            **val_label_meta,
            "threshold_policy": "thresholds_fit_on_pre_validation_2024_rows_only",
            "threshold_fit_rows": int(raw_train_mask.sum()),
        },
        "final_label_meta": {
            **full_label_meta,
            "threshold_policy": "thresholds_fit_on_all_2024_rows_for_final_2024_only_model",
        },
        "feature_count": int(len(cols)),
        "feature_cols": cols,
        "selected_clean_features_used": [c for c in cols if c.startswith(CLEAN_PREFIX)],
        "router_features_used": [c for c in cols if c.startswith("router_")],
        "validation": _eval_report(y_val, val_proba),
        "state_class_matrix_validation": state_class_val.tolist(),
        "state_class_matrix_final": state_class.tolist(),
        "hmm_log_likelihood_validation": val_hmm.log_likelihood_,
        "hmm_log_likelihood_final": final_hmm.log_likelihood_,
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_proba.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_proba.sum(axis=1).max()),
        "predict_counts": {CLASSES[i]: int((np.argmax(pred_proba, axis=1) == i).sum()) for i in range(len(CLASSES))},
        "predict_confidence_mean": float(pred_output[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(pred_output[f"{PRED_PREFIX}entropy"].mean()),
        "predict_path_diagnostics": _predicted_path_diagnostics(pred_raw, pred_output, pred_proba, int(args.horizon)),
        "notes": [
            "Gaussian HMM with diagonal covariance; hmmlearn is not required.",
            "HMM hidden-state posteriors are mapped to 5 trading regime probabilities using 2024 future-path labels.",
            "Sidecar intentionally writes soft regime_pred_* columns only; no hard argmax label columns.",
            "Artifact/model id contains HMM, but output feature names avoid hmm_ to satisfy model feature audits.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] train_sidecar={train_sidecar}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
