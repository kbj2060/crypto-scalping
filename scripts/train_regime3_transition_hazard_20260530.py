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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


MODEL_ID = "regime3_transition_hazard_sensitive_h6_20260530"
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
CURRENT_SIDECAR_STEM = "regime3_current_sensitive_hmm_wide24"
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_transition_hazard_sensitive_h6_20260530"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_transition_hazard_sensitive_h6_20260530_report.json"


def _current_path(current_dir: Path, source: Path) -> Path:
    return current_dir / f"{source.stem}_{CURRENT_SIDECAR_STEM}.csv"


def _merge_current(frame: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    current = _read(current_path)
    required = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    missing = [col for col in required if col not in current.columns]
    if missing:
        raise ValueError(f"{current_path} missing required sensitive current columns: {missing}")
    keep = ["timestamp"] + [col for col in current.columns if col.startswith(CURRENT_PREFIX)]
    out = frame.merge(current[keep], on="timestamp", how="left", validate="one_to_one")
    null_cols = [col for col in keep if col != "timestamp" and out[col].isna().any()]
    if null_cols:
        raise ValueError(f"sensitive current merge produced nulls: {null_cols[:10]}")
    out[f"{CURRENT_PREFIX}directional_bias"] = out[f"{CURRENT_PREFIX}bull_prob"] - out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}trend_prob"] = out[f"{CURRENT_PREFIX}bull_prob"] + out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}range_prob"] = out[f"{CURRENT_PREFIX}chop_prob"]
    return out


def _current_probs(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"missing current probability columns: {missing}")
    probs = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def _labels(frame: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur = np.argmax(_current_probs(frame), axis=1).astype(np.int64)
    n = max(0, len(frame) - int(horizon))
    current = cur[:n]
    future = cur[int(horizon) : int(horizon) + n]
    transition = (future != current).astype(np.int64)
    return current, future, transition


def _docs_rolled_cols(frames: list[pd.DataFrame], include_current: bool, max_features: int) -> list[str]:
    global CURRENT_PREFIX  # used by imported _feature_cols
    import scripts.train_regime3_pred_tft_vsn_wide24_current_20260529 as pred_mod

    old_prefix = pred_mod.CURRENT_PREFIX
    pred_mod.CURRENT_PREFIX = CURRENT_PREFIX
    try:
        cols = _feature_cols(frames, max_features=max_features, feature_pack="docs_regime_pred_rolled", include_current_features=include_current)
    finally:
        pred_mod.CURRENT_PREFIX = old_prefix
    if include_current:
        return cols
    bad = [col for col in cols if col.startswith(CURRENT_PREFIX)]
    if bad:
        raise ValueError(f"current features leaked into no-current transition feature set: {bad[:10]}")
    return cols


def _prepare(frame: pd.DataFrame, cols: list[str], fit_mask: np.ndarray | None = None, scaler: StandardScaler | None = None, medians: pd.Series | None = None) -> tuple[np.ndarray, StandardScaler, pd.Series]:
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
    counts = np.bincount(y.astype(int), minlength=max(2, int(y.max()) + 1)).astype(np.float64)
    w = counts.sum() / np.clip(len(counts) * counts, 1.0, None)
    return np.clip(w[y.astype(int)], 0.25, 8.0)


def _fit_hgb(x: np.ndarray, y: np.ndarray, seed: int, sample_weight: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=420,
        max_leaf_nodes=17,
        l2_regularization=0.08,
        min_samples_leaf=45,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.16,
        n_iter_no_change=25,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _safe_proba(model: HistGradientBoostingClassifier, x: np.ndarray, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.full((len(x), n_classes), 1e-9, dtype=np.float64)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, i]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def _combine(current: np.ndarray, hazard_p: np.ndarray, dest_p: np.ndarray, threshold: float) -> np.ndarray:
    pred = current.copy()
    fire = hazard_p >= float(threshold)
    masked = dest_p.copy()
    masked[np.arange(len(masked)), current] = 0.0
    masked /= np.clip(masked.sum(axis=1, keepdims=True), 1e-12, None)
    pred[fire] = np.argmax(masked[fire], axis=1)
    return pred


def _eval_future(current: np.ndarray, future: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    transition = future != current
    out = {
        "rows": int(len(future)),
        "accuracy": float(accuracy_score(future, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(future, pred)),
        "transition_rows": int(transition.sum()),
        "persistence_rows": int((~transition).sum()),
        "confusion_matrix": confusion_matrix(future, pred, labels=[0, 1, 2]).tolist(),
    }
    if transition.any():
        out["transition_accuracy"] = float(accuracy_score(future[transition], pred[transition]))
        out["transition_balanced_accuracy"] = float(balanced_accuracy_score(future[transition], pred[transition]))
        out["transition_confusion_matrix"] = confusion_matrix(future[transition], pred[transition], labels=[0, 1, 2]).tolist()
    if (~transition).any():
        out["persistence_accuracy"] = float(accuracy_score(future[~transition], pred[~transition]))
        out["persistence_balanced_accuracy"] = float(balanced_accuracy_score(future[~transition], pred[~transition]))
    return out


def _eval_hazard(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(np.int64)
    out = {
        "rows": int(len(y)),
        "transition_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.column_stack([1.0 - p, p]), labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["roc_auc"] = None
    return out


def _output(ts: pd.Series, current: np.ndarray, hazard_p: np.ndarray, dest_p: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = _combine(current, hazard_p, dest_p, threshold)
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    out["regime3_transition_h6_prob"] = hazard_p
    out["regime3_transition_h6_pred"] = (hazard_p >= threshold).astype(np.int64)
    for i, name in enumerate(CLASSES3):
        out[f"regime3_transition_h6_dest_{name}_prob"] = dest_p[:, i]
    for i, name in enumerate(CLASSES3):
        out[f"regime3_transition_h6_future_{name}_prob"] = (pred == i).astype(float)
    out["regime3_transition_h6_future_pred_id"] = pred
    out["regime3_transition_h6_future_pred_name"] = [CLASSES3[i] for i in pred]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train Regime3 h6 transition hazard and destination heads.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--include-current-features", action="store_true")
    p.add_argument("--seed", type=int, default=30530)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    raw_frames = [_add_rolling_stable_features(_read(path)) for path in sources]
    frames = [_merge_current(frame, _current_path(args.current_dir, path)) for frame, path in zip(raw_frames, sources)]
    train = _merge_current(_add_rolling_stable_features(_read(args.train_2024)), _current_path(args.current_dir, args.train_2024))
    cols = _docs_rolled_cols([train] + frames, include_current=args.include_current_features, max_features=args.max_features)

    current, future, transition = _labels(train, args.horizon)
    labeled = train.iloc[: len(future)].copy()
    ts = pd.to_datetime(labeled["timestamp"])
    fit_mask = ts < pd.Timestamp(args.val_start)
    val_mask = ~fit_mask
    x, scaler, medians = _prepare(labeled, cols, fit_mask=fit_mask)

    hazard_model = _fit_hgb(x[fit_mask], transition[fit_mask], args.seed, sample_weight=_weights(transition[fit_mask]))
    transition_mask = fit_mask & (transition == 1)
    if int(transition_mask.sum()) < 100:
        raise ValueError(f"not enough transition rows for destination head: {int(transition_mask.sum())}")
    dest_model = _fit_hgb(x[transition_mask], future[transition_mask], args.seed + 11, sample_weight=_weights(future[transition_mask]))

    hazard_val = _safe_proba(hazard_model, x[val_mask], 2)[:, 1]
    dest_val = _safe_proba(dest_model, x[val_mask], len(CLASSES3))
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        pred_val = _combine(current[val_mask], hazard_val, dest_val, float(threshold))
        ev = _eval_future(current[val_mask], future[val_mask], pred_val)
        score = (ev.get("transition_balanced_accuracy", 0.0), ev["balanced_accuracy"])
        if best is None or score > best[0]:
            best = (score, float(threshold), ev)
    assert best is not None
    threshold = best[1]

    # Refit final models on all 2024 labeled rows after threshold selection.
    x_full, scaler_full, medians_full = _prepare(labeled, cols)
    hazard_final = _fit_hgb(x_full, transition, args.seed + 101, sample_weight=_weights(transition))
    trans_rows = transition == 1
    dest_final = _fit_hgb(x_full[trans_rows], future[trans_rows], args.seed + 111, sample_weight=_weights(future[trans_rows]))

    model_path = args.out_dir / "regime3_transition_hazard_h6.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES3,
            "horizon": int(args.horizon),
            "current_prefix": CURRENT_PREFIX,
            "current_sidecar_stem": CURRENT_SIDECAR_STEM,
            "include_current_features": bool(args.include_current_features),
            "feature_cols": cols,
            "feature_medians": medians_full.to_dict(),
            "scaler": scaler_full,
            "hazard_model": hazard_final,
            "destination_model": dest_final,
            "threshold": float(threshold),
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "horizon_bars": int(args.horizon),
        "classes": CLASSES3,
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "include_current_features": bool(args.include_current_features),
        "feature_count": len(cols),
        "feature_cols": cols,
        "threshold_selected_on_validation": float(threshold),
        "validation": {
            "hazard": _eval_hazard(transition[val_mask], hazard_val, threshold),
            "future": best[2],
        },
        "label_counts_2024": {
            "transition": int(transition.sum()),
            "persistence": int((transition == 0).sum()),
            "future": {CLASSES3[i]: int((future == i).sum()) for i in range(len(CLASSES3))},
        },
        "outputs": {},
        "leakage_audit": {
            "uses_2026_for_selection": False,
            "current_used_as_target_source": True,
            "current_used_as_input_features": bool(args.include_current_features),
            "forbidden_current_feature_count_when_disabled": int(sum(col.startswith(CURRENT_PREFIX) for col in cols)),
        },
    }

    for path, frame in zip(sources, frames):
        current_i, future_i, transition_i = _labels(frame, args.horizon)
        eval_frame = frame.iloc[: len(future_i)].copy()
        x_eval, _, _ = _prepare(eval_frame, cols, scaler=scaler_full, medians=medians_full)
        hazard_p = _safe_proba(hazard_final, x_eval, 2)[:, 1]
        dest_p = _safe_proba(dest_final, x_eval, len(CLASSES3))
        pred_i = _combine(current_i, hazard_p, dest_p, threshold)
        sidecar = args.out_dir / f"{path.stem}_regime3_transition_hazard_h6.csv"
        _output(eval_frame["timestamp"], current_i, hazard_p, dest_p, threshold).to_csv(sidecar, index=False)
        report["outputs"][path.name] = {
            "source": str(path),
            "sidecar": str(sidecar),
            "rows": int(len(eval_frame)),
            "range": [str(eval_frame["timestamp"].iloc[0]), str(eval_frame["timestamp"].iloc[-1])],
            "hazard": _eval_hazard(transition_i, hazard_p, threshold),
            "future": _eval_future(current_i, future_i, pred_i),
        }
        print(f"[{MODEL_ID}] wrote {sidecar}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] report={args.report}", flush=True)
    print(f"[{MODEL_ID}] threshold={threshold:.3f} features={len(cols)} include_current={args.include_current_features}", flush=True)


if __name__ == "__main__":
    main()
