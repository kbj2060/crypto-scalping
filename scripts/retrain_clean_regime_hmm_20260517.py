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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX, clean_regime_factors, clean_regime_fit_columns  # noqa: E402
from features.elite import RegimeEngine  # noqa: E402


MODEL_ID = "clean_regime_2024_unsup_hmm_v6_20260517"
CLASSES = ["bull", "bear", "chop", "whipsaw", "normal"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime_hmm_v6_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime_hmm_v6_20260517_report.json"
DEFAULT_TRANSFORMS = (
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
)
NON_FEATURES = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
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
FACTOR_NAMES = (
    "factor_trend",
    "factor_flow",
    "factor_vol",
    "factor_crowding",
    "factor_liquidity",
    "trend_bias",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


@dataclass
class GaussianStateModel:
    n_states: int
    n_iter: int
    seed: int
    min_var: float = 1e-4
    sticky: float = 0.94

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
        score = x[:, 0] if x.shape[1] else np.arange(n, dtype=float)
        order = np.argsort(score)
        centers = []
        for frac in np.linspace(0.04, 0.96, self.n_states):
            centers.append(x[order[int(np.clip(frac * (n - 1), 0, n - 1))]])
        self.mu_ = np.asarray(centers, dtype=np.float64)
        global_var = np.var(x, axis=0).clip(min=self.min_var)
        self.var_ = np.tile(global_var[None, :], (self.n_states, 1))
        self.pi_ = np.ones(self.n_states, dtype=np.float64) / self.n_states
        self.A_ = np.full(
            (self.n_states, self.n_states),
            (1.0 - self.sticky) / max(self.n_states - 1, 1),
            dtype=np.float64,
        )
        np.fill_diagonal(self.A_, self.sticky)
        self.A_ += rng.random(self.A_.shape) * 1e-4
        self.A_ /= self.A_.sum(axis=1, keepdims=True)

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        assert self.mu_ is not None and self.var_ is not None
        diff = x[:, None, :] - self.mu_[None, :, :]
        var = np.maximum(self.var_, self.min_var)
        return -0.5 * np.sum((diff * diff) / var[None, :, :] + np.log(2.0 * np.pi * var[None, :, :]), axis=2)

    def _forward_backward(self, log_emit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
        gamma = np.exp(log_a + log_b - ll)
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-300, None)
        return gamma, log_a, log_b, ll

    def fit(self, x: np.ndarray) -> "GaussianStateModel":
        x = np.asarray(x, dtype=np.float64)
        self._init_params(x)
        assert self.pi_ is not None and self.A_ is not None
        prev_ll = -np.inf
        for _ in range(int(self.n_iter)):
            log_emit = self._log_emission(x)
            gamma, log_a, log_b, ll = self._forward_backward(log_emit)
            log_trans = np.log(self.A_ + 1e-300)
            log_xi_sum = np.full((self.n_states, self.n_states), -np.inf, dtype=np.float64)
            for t in range(len(x) - 1):
                lx = log_a[t][:, None] + log_trans + log_emit[t + 1][None, :] + log_b[t + 1][None, :] - ll
                log_xi_sum = np.logaddexp(log_xi_sum, lx)
            xi_sum = np.exp(log_xi_sum)
            self.pi_ = gamma[0] / np.clip(gamma[0].sum(), 1e-300, None)
            self.A_ = xi_sum / np.clip(xi_sum.sum(axis=1, keepdims=True), 1e-300, None)
            self.A_ = 0.015 / max(self.n_states - 1, 1) + 0.985 * self.A_
            self.A_ /= self.A_.sum(axis=1, keepdims=True)
            wsum = gamma.sum(axis=0) + 1e-300
            self.mu_ = (gamma.T @ x) / wsum[:, None]
            assert self.var_ is not None
            for s in range(self.n_states):
                diff = x - self.mu_[s]
                self.var_[s] = ((gamma[:, s][:, None] * diff * diff).sum(axis=0) / wsum[s]).clip(min=self.min_var)
            self.log_likelihood_.append(ll)
            if abs(ll - prev_ll) < 1e-4 * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll
        return self

    def filter_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.pi_ is not None and self.A_ is not None
        log_emit = self._log_emission(np.asarray(x, dtype=np.float64))
        log_trans = np.log(self.A_ + 1e-300)
        log_alpha = np.zeros_like(log_emit)
        log_alpha[0] = np.log(self.pi_ + 1e-300) + log_emit[0]
        log_alpha[0] -= self._logsumexp(log_alpha[0], axis=0)
        for t in range(1, len(log_emit)):
            log_alpha[t] = log_emit[t] + self._logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
            log_alpha[t] -= self._logsumexp(log_alpha[t], axis=0)
        out = np.exp(log_alpha)
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
        return out


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _safe_numeric(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _safe_numeric(frame, c) for c in cols}, index=frame.index)


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    priority = clean_regime_fit_columns(frame)
    extra_hints = [
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "sum_open_interest_value",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "last_funding_rate",
        "whale_retail_ratio",
        "smart_money_flow",
        "squeeze_power",
        "oi_change_rate",
        "net_taker_ratio",
        "taker_acceleration",
        "trade_intensity",
        "big_trade_ratio",
        "log_return",
        "volatility_z",
        "rsi",
        "macd_hist",
        "bb_width_z",
        "hma_slope",
        "wick_ratio",
        "garman_klass_vol",
        "realized_vol_ratio",
        "rogers_satchell_vol",
        "parkinson_vol",
        "amihud_illiquidity_z",
        "btc_corr_60",
        "eth_btc_ratio_change",
        "fvg_dist",
        "chop_index",
        "cvp_poc_dist",
        "cvp_cluster_position",
        "cvp_volume_imbalance",
        "breakout_strength",
        "long_squeeze_risk",
        "funding_price_divergence",
        "ofi_acceleration",
        "kalman_velocity",
        "realized_skewness",
        "ofti",
        "kel",
        "mta_funding",
        "svps",
        "pred_mdjd",
        "conf_mdjd",
    ]
    factor_cols = [f"{CLEAN_PREFIX}{name}" for name in FACTOR_NAMES]
    selected: list[str] = []
    for col in priority + extra_hints + factor_cols:
        lower = col.lower()
        if col in selected or col in NON_FEATURES or col in FORBIDDEN_EXACT:
            continue
        if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
            continue
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any():
            selected.append(col)
    return selected


def _with_factors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    factors = clean_regime_factors(out)
    for name in FACTOR_NAMES:
        col = f"{CLEAN_PREFIX}{name}"
        out[col] = pd.to_numeric(factors[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _current_labels(frame: pd.DataFrame) -> np.ndarray:
    labeled = RegimeEngine().compute(frame.copy())
    values = labeled[[f"regime_{name}" for name in CLASSES]].to_numpy(dtype=float)
    return np.argmax(values, axis=1).astype(int)


def _state_class_matrix(state_prob: np.ndarray, y: np.ndarray, smoothing: float = 0.05) -> np.ndarray:
    mat = np.full((state_prob.shape[1], len(CLASSES)), float(smoothing), dtype=np.float64)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    for cls in range(len(CLASSES)):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(counts[cls], 1.0)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    out = state_prob @ state_class
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out


def _fit_observations(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str], n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, Any, pd.Series]:
    x_train_raw = _matrix(train, cols)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = _matrix(pred, cols).fillna(medians).fillna(0.0)
    preprocess = make_pipeline(
        RobustScaler(quantile_range=(5.0, 95.0)),
        PCA(n_components=min(int(n_components), len(cols)), whiten=True, random_state=int(seed)),
    )
    train_obs = preprocess.fit_transform(x_train)
    pred_obs = preprocess.transform(x_pred)
    return train_obs, pred_obs, preprocess, medians


def _output_frame(ts: pd.Series, frame: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for name in FACTOR_NAMES:
        col = f"{CLEAN_PREFIX}{name}"
        out[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    for i, name in enumerate(CLASSES):
        out[f"{CLEAN_PREFIX}{name}_prob"] = proba[:, i]
    sorted_prob = np.sort(proba, axis=1)
    out[f"{CLEAN_PREFIX}trend_prob"] = out[f"{CLEAN_PREFIX}bull_prob"] + out[f"{CLEAN_PREFIX}bear_prob"]
    out[f"{CLEAN_PREFIX}micro_prob"] = out[f"{CLEAN_PREFIX}chop_prob"] + out[f"{CLEAN_PREFIX}whipsaw_prob"] + out[f"{CLEAN_PREFIX}normal_prob"]
    out[f"{CLEAN_PREFIX}directional_bias"] = out[f"{CLEAN_PREFIX}bull_prob"] - out[f"{CLEAN_PREFIX}bear_prob"]
    out[f"{CLEAN_PREFIX}range_prob"] = out[f"{CLEAN_PREFIX}chop_prob"] + out[f"{CLEAN_PREFIX}normal_prob"]
    out[f"{CLEAN_PREFIX}instability_prob"] = out[f"{CLEAN_PREFIX}whipsaw_prob"]
    out[f"{CLEAN_PREFIX}confidence"] = sorted_prob[:, -1]
    out[f"{CLEAN_PREFIX}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / math.log(len(CLASSES))
    out[f"{CLEAN_PREFIX}margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    return out


def _eval_report(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CLASSES))))),
        "true_counts": {CLASSES[i]: int((y_true == i).sum()) for i in range(len(CLASSES))},
        "pred_counts": {CLASSES[i]: int((pred == i).sum()) for i in range(len(CLASSES))},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain clean_regime current-state sidecars with a causal-filtered Gaussian state model.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--transform", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--n-iter", type=int, default=18)
    parser.add_argument("--seed", type=int, default=170517)
    args = parser.parse_args()

    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train = _with_factors(_read(args.train_2024))
    cols = _candidate_columns(train)
    if len(cols) < 8:
        raise ValueError(f"not enough HMM clean regime columns: {len(cols)}")

    ts = pd.to_datetime(train["timestamp"])
    train_mask = ts < pd.Timestamp(args.val_start)
    val_mask = ~train_mask
    train_part = train.loc[train_mask].copy()
    val_part = train.loc[val_mask].copy()
    train_obs, val_obs, val_preprocess, val_medians = _fit_observations(train_part, val_part, cols, int(args.pca_components), int(args.seed))
    val_model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed)).fit(train_obs)
    train_state = val_model.filter_proba(train_obs)
    val_state = val_model.filter_proba(val_obs)
    y_train = _current_labels(train_part)
    y_val = _current_labels(val_part)
    state_class_val = _state_class_matrix(train_state, y_train)
    val_proba = _class_proba(val_state, state_class_val)

    full_obs, _, preprocess, medians = _fit_observations(train, train.iloc[:1].copy(), cols, int(args.pca_components), int(args.seed) + 101)
    model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed) + 101).fit(full_obs)
    full_state = model.filter_proba(full_obs)
    y_full = _current_labels(train)
    state_class = _state_class_matrix(full_state, y_full)
    full_proba = _class_proba(full_state, state_class)

    model_path = args.out_dir / "clean_regime_state_v6_2024.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "clean_prefix": CLEAN_PREFIX,
            "classes": CLASSES,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "preprocess": preprocess,
            "model": model,
            "state_class_matrix": state_class,
            "state_count": int(args.states),
            "pca_components": int(args.pca_components),
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "clean_prefix": CLEAN_PREFIX,
        "fit_source": str(args.train_2024),
        "fit_rows": int(len(train)),
        "fit_range": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
        "feature_cols": cols,
        "feature_count": int(len(cols)),
        "states": int(args.states),
        "pca_components": int(args.pca_components),
        "validation": _eval_report(y_val, val_proba),
        "state_class_matrix_validation": state_class_val.tolist(),
        "state_class_matrix_final": state_class.tolist(),
        "log_likelihood_validation": val_model.log_likelihood_,
        "log_likelihood_final": model.log_likelihood_,
        "outputs": {},
        "notes": [
            "This replaces the previous BGMM clean_regime sidecar with an HMM-style Gaussian latent-state model.",
            "Output columns intentionally keep the clean_regime_2024_unsup_v4_ prefix for downstream compatibility.",
            "No risk_off, transition, cluster id, hidden-state probability, or hard label columns are written.",
            "Output probabilities are causal filtered probabilities; inference does not use future rows in the transform sequence.",
            "Hidden states are mapped to the 5 current RegimeEngine taxonomy classes using 2024 labels.",
        ],
    }

    for src in transforms:
        frame = _with_factors(_read(src))
        x = _matrix(frame, cols).fillna(medians).fillna(0.0)
        obs = preprocess.transform(x)
        state = model.filter_proba(obs)
        proba = _class_proba(state, state_class)
        clean = _output_frame(frame["timestamp"], frame, proba)
        sidecar = args.out_dir / f"{src.stem}_clean_regime_hmm_v6.csv"
        clean.to_csv(sidecar, index=False)
        pred = np.argmax(proba, axis=1)
        report["outputs"][src.name] = {
            "source": str(src),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "sidecar": str(sidecar),
            "clean_feature_count": int(len(clean.columns) - 1),
            "probability_sum_min": float(proba.sum(axis=1).min()),
            "probability_sum_max": float(proba.sum(axis=1).max()),
            "pred_counts": {CLASSES[i]: int((pred == i).sum()) for i in range(len(CLASSES))},
            "confidence_mean": float(clean[f"{CLEAN_PREFIX}confidence"].mean()),
            "entropy_mean": float(clean[f"{CLEAN_PREFIX}entropy"].mean()),
        }
        print(f"[{MODEL_ID}] wrote {sidecar} rows={len(frame)} clean_cols={len(clean.columns) - 1}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
