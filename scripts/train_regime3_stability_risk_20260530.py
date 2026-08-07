#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_regime3_pred_tft_vsn_wide24_current_20260529 as pred_mod  # noqa: E402
from scripts.retrain_clean_regime_hmm_20260517 import _json_default  # noqa: E402
from scripts.train_regime3_pred_tft_vsn_wide24_current_20260529 import (  # noqa: E402
    CLASSES3,
    DEFAULT_TRANSFORMS,
    DOCS_REGIME_PRED_FEATURES,
    ROLLING_BASE_COLS,
    _add_rolling_stable_features,
    _feature_cols,
    _read,
)


MODEL_ID = "regime3_stability_risk_h6_20260530"
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
CURRENT_SIDECAR_STEM = "regime3_current_sensitive_hmm_wide24"
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_stability_risk_h6_20260530_report.json"


def _current_path(current_dir: Path, source: Path) -> Path:
    return current_dir / f"{source.stem}_{CURRENT_SIDECAR_STEM}.csv"


def _merge_current(frame: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    current = _read(current_path)
    required = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    missing = [col for col in required if col not in current.columns]
    if missing:
        raise ValueError(f"{current_path} missing required current columns: {missing}")
    keep = ["timestamp"] + [col for col in current.columns if col.startswith(CURRENT_PREFIX)]
    out = frame.merge(current[keep], on="timestamp", how="left", validate="one_to_one")
    null_cols = [col for col in keep if col != "timestamp" and out[col].isna().any()]
    if null_cols:
        raise ValueError(f"current merge produced nulls: {null_cols[:10]}")
    return out


def _current_probs(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    probs = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def _smooth_runs(labels: np.ndarray, min_duration: int) -> np.ndarray:
    out = labels.copy()
    if min_duration <= 1 or len(out) == 0:
        return out
    i = 0
    while i < len(out):
        j = i + 1
        while j < len(out) and out[j] == out[i]:
            j += 1
        if (j - i) < min_duration:
            if i > 0:
                out[i:j] = out[i - 1]
            elif j < len(out):
                out[i:j] = out[j]
        i = j
    return out


def _feature_columns(frames: list[pd.DataFrame], max_features: int) -> list[str]:
    old_prefix = pred_mod.CURRENT_PREFIX
    pred_mod.CURRENT_PREFIX = CURRENT_PREFIX
    try:
        cols = _feature_cols(frames, max_features=max_features, feature_pack="docs_regime_pred_rolled", include_current_features=False)
    finally:
        pred_mod.CURRENT_PREFIX = old_prefix
    bad = [col for col in cols if col.startswith(CURRENT_PREFIX) or "regime3_current" in col]
    if bad:
        raise ValueError(f"current probability features leaked into stability/risk inputs: {bad[:10]}")
    return cols


def _add_stability_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    for w in [3, 6, 12, 24, 48]:
        ret = close.pct_change(w).replace([np.inf, -np.inf], np.nan)
        out[f"sr_ret_abs_{w}"] = ret.abs()
        out[f"sr_ret_signed_{w}"] = ret
        out[f"sr_ret_vol_{w}"] = close.pct_change().rolling(w, min_periods=max(3, w // 3)).std()
    if "bb_width" in out.columns:
        bbw = pd.to_numeric(out["bb_width"], errors="coerce")
        out["sr_bb_width_delta_6"] = bbw - bbw.shift(6)
        out["sr_bb_width_delta_24"] = bbw - bbw.shift(24)
    if "adx_14" in out.columns:
        adx = pd.to_numeric(out["adx_14"], errors="coerce")
        out["sr_adx_delta_6"] = adx - adx.shift(6)
        out["sr_adx_delta_24"] = adx - adx.shift(24)
    for col in ["rsi", "macd_hist", "mean_reversion_z", "dual_momentum", "cvd_12", "cvd_288", "funding_pressure", "volume"]:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[f"sr_{col}_delta_6"] = s - s.shift(6)
            out[f"sr_{col}_delta_24"] = s - s.shift(24)
    return out


def _labels(frame: pd.DataFrame, horizon: int, min_duration: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur = np.argmax(_current_probs(frame), axis=1).astype(np.int64)
    stable = _smooth_runs(cur, min_duration)
    n = max(0, len(frame) - int(horizon))
    now = stable[:n]
    future = stable[int(horizon) : int(horizon) + n]
    transition = (now != future).astype(np.int64)
    # Risk target also penalizes raw sensitive churn inside the horizon, not just final class change.
    raw = cur
    churn = np.zeros(n, dtype=np.float64)
    for i in range(n):
        path = raw[i : i + int(horizon) + 1]
        churn[i] = np.mean(path[1:] != path[:-1]) if len(path) > 1 else 0.0
    risk = np.clip(0.7 * transition + 0.3 * churn, 0.0, 1.0)
    return now, future, transition, risk


def _prepare(frame: pd.DataFrame, cols: list[str], fit_mask: np.ndarray | None = None, scaler: StandardScaler | None = None, medians: pd.Series | None = None):
    raw = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if medians is None:
        fit_raw = raw if fit_mask is None else raw.loc[fit_mask]
        medians = fit_raw.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(filled if fit_mask is None else filled.loc[fit_mask])
    x = scaler.transform(filled).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), scaler, medians


def _weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    w = counts.sum() / np.clip(2.0 * counts, 1.0, None)
    return np.clip(w[y.astype(int)], 0.25, 8.0)


def _fit_classifier(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=520,
        max_leaf_nodes=21,
        l2_regularization=0.10,
        min_samples_leaf=45,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.16,
        n_iter_no_change=30,
    )
    model.fit(x, y, sample_weight=_weights(y))
    return model


def _fit_regressor(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=480,
        max_leaf_nodes=21,
        l2_regularization=0.10,
        min_samples_leaf=45,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.16,
        n_iter_no_change=30,
        loss="squared_error",
    )
    model.fit(x, y)
    return model


def _proba2(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.full((len(x), 2), 1e-9, dtype=np.float64)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, i]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out[:, 1]


def _eval_transition(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(np.int64)
    out = {
        "rows": int(len(y)),
        "transition_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fire_rate": float(np.mean(pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["roc_auc"] = None
    return out


def _eval_risk(y_transition: np.ndarray, risk: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    corr = float(np.corrcoef(risk, pred)[0, 1]) if np.std(risk) > 1e-12 and np.std(pred) > 1e-12 else 0.0
    order = np.argsort(pred)
    top = pred >= np.quantile(pred, 0.80)
    low = pred <= np.quantile(pred, 0.20)
    return {
        "risk_corr": corr,
        "top20_transition_rate": float(y_transition[top].mean()) if top.any() else None,
        "low20_transition_rate": float(y_transition[low].mean()) if low.any() else None,
        "top20_avg_risk_target": float(risk[top].mean()) if top.any() else None,
        "low20_avg_risk_target": float(risk[low].mean()) if low.any() else None,
    }


def _output(ts: pd.Series, transition_p: np.ndarray, risk_score: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": ts.reset_index(drop=True),
        "regime3_stability_h6_score": 1.0 - transition_p,
        "regime3_transition_h6_risk_prob": transition_p,
        "regime3_transition_h6_risk_pred": (transition_p >= threshold).astype(np.int64),
        "regime3_churn_h6_risk_score": np.clip(risk_score, 0.0, 1.0),
    })


def main() -> None:
    p = argparse.ArgumentParser(description="Train Regime3 stability/risk feature heads.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--min-duration", type=int, default=6)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--max-features", type=int, default=128)
    p.add_argument("--seed", type=int, default=50530)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    frames = [
        _merge_current(_add_stability_features(_add_rolling_stable_features(_read(path))), _current_path(args.current_dir, path))
        for path in sources
    ]
    train = _merge_current(_add_stability_features(_add_rolling_stable_features(_read(args.train_2024))), _current_path(args.current_dir, args.train_2024))
    cols = _feature_columns([train] + frames, args.max_features)
    extra_cols = [col for col in train.columns if col.startswith("sr_") and col not in cols]
    cols = (cols + extra_cols)[: args.max_features]
    if any(col.startswith(CURRENT_PREFIX) for col in cols):
        raise ValueError("CURRENT probability features leaked into stability/risk feature set")

    now, future, transition, risk = _labels(train, args.horizon, args.min_duration)
    labeled = train.iloc[: len(transition)].copy()
    fit_mask = pd.to_datetime(labeled["timestamp"]) < pd.Timestamp(args.val_start)
    val_mask = ~fit_mask
    x, _, _ = _prepare(labeled, cols, fit_mask=fit_mask)
    transition_model = _fit_classifier(x[fit_mask], transition[fit_mask], args.seed)
    risk_model = _fit_regressor(x[fit_mask], risk[fit_mask], args.seed + 11)
    val_p = _proba2(transition_model, x[val_mask])
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        ev = _eval_transition(transition[val_mask], val_p, float(threshold))
        score = 0.45 * ev["balanced_accuracy"] + 0.35 * ev["recall"] + 0.20 * ev["precision"]
        if best is None or score > best[0]:
            best = (score, float(threshold), ev)
    assert best is not None
    threshold = best[1]

    x_full, scaler, medians = _prepare(labeled, cols)
    transition_final = _fit_classifier(x_full, transition, args.seed + 101)
    risk_final = _fit_regressor(x_full, risk, args.seed + 111)
    model_path = args.out_dir / "regime3_stability_risk_h6.joblib"
    joblib.dump({
        "model_id": MODEL_ID,
        "classes": CLASSES3,
        "horizon": int(args.horizon),
        "min_duration": int(args.min_duration),
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "feature_cols": cols,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "transition_model": transition_final,
        "risk_model": risk_final,
        "threshold": float(threshold),
    }, model_path)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "horizon_bars": int(args.horizon),
        "min_duration": int(args.min_duration),
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "feature_count": len(cols),
        "feature_cols": cols,
        "threshold_selected_on_validation": float(threshold),
        "validation": {
            "transition": best[2],
            "risk": _eval_risk(transition[val_mask], risk[val_mask], risk_model.predict(x[val_mask])),
        },
        "outputs": {},
        "leakage_audit": {
            "uses_2026_for_selection": False,
            "current_probability_features_used_as_model_inputs": False,
            "current_feature_count": int(sum(col.startswith(CURRENT_PREFIX) for col in cols)),
            "current_sidecar_used_for_label_generation_only": True,
        },
    }

    for path, frame in zip(sources, frames):
        now_i, future_i, transition_i, risk_i = _labels(frame, args.horizon, args.min_duration)
        eval_frame = frame.iloc[: len(transition_i)].copy()
        x_eval, _, _ = _prepare(eval_frame, cols, scaler=scaler, medians=medians)
        p_eval = _proba2(transition_final, x_eval)
        risk_eval = np.clip(risk_final.predict(x_eval), 0.0, 1.0)
        sidecar = args.out_dir / f"{path.stem}_regime3_stability_risk_h6.csv"
        _output(eval_frame["timestamp"], p_eval, risk_eval, threshold).to_csv(sidecar, index=False)
        report["outputs"][path.name] = {
            "source": str(path),
            "sidecar": str(sidecar),
            "rows": int(len(eval_frame)),
            "range": [str(eval_frame["timestamp"].iloc[0]), str(eval_frame["timestamp"].iloc[-1])],
            "transition": _eval_transition(transition_i, p_eval, threshold),
            "risk": _eval_risk(transition_i, risk_i, risk_eval),
        }
        print(f"[{MODEL_ID}] wrote {sidecar}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] threshold={threshold:.3f} features={len(cols)}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
