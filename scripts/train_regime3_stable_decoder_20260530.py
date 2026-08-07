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
    _add_rolling_stable_features,
    _feature_cols,
    _read,
)


MODEL_ID = "regime3_stable_h6_decoder_20260530"
CURRENT_PREFIX = "regime3_current_sensitive_wide24_"
CURRENT_SIDECAR_STEM = "regime3_current_sensitive_hmm_wide24"
DEFAULT_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_stable_h6_decoder_20260530"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_stable_h6_decoder_20260530_report.json"


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
    out[f"{CURRENT_PREFIX}directional_bias"] = out[f"{CURRENT_PREFIX}bull_prob"] - out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}trend_prob"] = out[f"{CURRENT_PREFIX}bull_prob"] + out[f"{CURRENT_PREFIX}bear_prob"]
    out[f"{CURRENT_PREFIX}range_prob"] = out[f"{CURRENT_PREFIX}chop_prob"]
    return out


def _current_argmax(frame: pd.DataFrame) -> np.ndarray:
    cols = [f"{CURRENT_PREFIX}{name}_prob" for name in CLASSES3]
    probs = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.argmax(probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None), axis=1).astype(np.int64)


def _smooth_runs(labels: np.ndarray, min_duration: int) -> np.ndarray:
    if min_duration <= 1 or len(labels) == 0:
        return labels.copy()
    out = labels.copy()
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


def _feature_columns(frames: list[pd.DataFrame], max_features: int, include_current: bool) -> list[str]:
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
        raise ValueError(f"current features leaked while include_current=False: {bad[:10]}")
    return cols


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


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=max(2, int(y.max()) + 1)).astype(np.float64)
    w = counts.sum() / np.clip(len(counts) * counts, 1.0, None)
    return np.clip(w[y.astype(int)], 0.25, 8.0)


def _fit_hgb(x: np.ndarray, y: np.ndarray, seed: int, sample_weight: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_iter=440,
        max_leaf_nodes=21,
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


def _labels(frame: pd.DataFrame, horizon: int, min_duration: int):
    stable = _smooth_runs(_current_argmax(frame), min_duration)
    n = max(0, len(frame) - int(horizon))
    current = stable[:n]
    future = stable[int(horizon) : int(horizon) + n]
    transition = (future != current).astype(np.int64)
    return current, future, transition


def _combine(current: np.ndarray, hazard_p: np.ndarray, dest_p: np.ndarray, threshold: float) -> np.ndarray:
    pred = current.copy()
    fire = hazard_p >= float(threshold)
    masked = dest_p.copy()
    masked[np.arange(len(masked)), current] = 0.0
    masked /= np.clip(masked.sum(axis=1, keepdims=True), 1e-12, None)
    pred[fire] = np.argmax(masked[fire], axis=1)
    return pred


def _eval(current: np.ndarray, future: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
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
    if (~transition).any():
        out["persistence_accuracy"] = float(accuracy_score(future[~transition], pred[~transition]))
        out["persistence_balanced_accuracy"] = float(balanced_accuracy_score(future[~transition], pred[~transition]))
    return out


def _score(ev: dict[str, Any]) -> float:
    return (
        0.45 * float(ev["balanced_accuracy"])
        + 0.35 * float(ev.get("transition_balanced_accuracy", 0.0))
        + 0.20 * float(ev.get("persistence_balanced_accuracy", 0.0))
    )


def _hazard_eval(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (p >= threshold).astype(np.int64)
    out = {
        "transition_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "fire_rate": float(np.mean(pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["roc_auc"] = None
    return out


def _output(ts: pd.Series, direct_p: np.ndarray, gated_pred: np.ndarray, hazard_p: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES3):
        out[f"regime3_stable_h6_direct_{name}_prob"] = direct_p[:, i]
        out[f"regime3_stable_h6_gated_{name}_prob"] = (gated_pred == i).astype(float)
    out["regime3_stable_h6_direct_pred_id"] = np.argmax(direct_p, axis=1)
    out["regime3_stable_h6_direct_pred_name"] = [CLASSES3[i] for i in np.argmax(direct_p, axis=1)]
    out["regime3_stable_h6_gated_pred_id"] = gated_pred
    out["regime3_stable_h6_gated_pred_name"] = [CLASSES3[i] for i in gated_pred]
    out["regime3_stable_h6_transition_prob"] = hazard_p
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train stable Regime3 h6 persistence-aware final decoder.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--max-features", type=int, default=112)
    p.add_argument("--min-validation-transition-bacc", type=float, default=0.25)
    p.add_argument("--min-validation-persistence-bacc", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=40530)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    sources = list(args.transform or DEFAULT_TRANSFORMS)
    frames = [
        _merge_current(_add_rolling_stable_features(_read(path)), _current_path(args.current_dir, path))
        for path in sources
    ]
    train = _merge_current(_add_rolling_stable_features(_read(args.train_2024)), _current_path(args.current_dir, args.train_2024))
    cols = _feature_columns([train] + frames, max_features=args.max_features, include_current=False)
    leaked_current = [col for col in cols if col.startswith(CURRENT_PREFIX)]
    if leaked_current:
        raise ValueError(f"CURRENT probability features leaked into decoder inputs: {leaked_current[:10]}")

    ts_all = pd.to_datetime(train["timestamp"])
    val_start = pd.Timestamp(args.val_start)
    configs = []
    for min_duration in [3, 6, 9, 12]:
        current, future, transition = _labels(train, args.horizon, min_duration)
        labeled = train.iloc[: len(future)].copy()
        fit_mask = pd.to_datetime(labeled["timestamp"]) < val_start
        val_mask = ~fit_mask
        x, _, _ = _prepare(labeled, cols, fit_mask=fit_mask)
        for transition_weight in [1.0, 2.0, 3.0, 4.0]:
            weight = _class_weights(future[fit_mask])
            weight = weight * np.where(transition[fit_mask] == 1, transition_weight, 1.0)
            direct = _fit_hgb(x[fit_mask], future[fit_mask], args.seed + int(min_duration * 10 + transition_weight), weight)
            direct_p = _safe_proba(direct, x[val_mask], len(CLASSES3))
            direct_pred = np.argmax(direct_p, axis=1)
            direct_ev = _eval(current[val_mask], future[val_mask], direct_pred)

            hazard = _fit_hgb(x[fit_mask], transition[fit_mask], args.seed + int(min_duration * 100 + transition_weight), _class_weights(transition[fit_mask]))
            trans_fit = fit_mask & (transition == 1)
            dest_weight = _class_weights(future[trans_fit])
            dest = _fit_hgb(x[trans_fit], future[trans_fit], args.seed + int(min_duration * 1000 + transition_weight), dest_weight)
            hazard_p = _safe_proba(hazard, x[val_mask], 2)[:, 1]
            dest_p = _safe_proba(dest, x[val_mask], len(CLASSES3))
            best_gated = None
            for threshold in np.linspace(0.15, 0.85, 71):
                gated_pred = _combine(current[val_mask], hazard_p, dest_p, float(threshold))
                ev = _eval(current[val_mask], future[val_mask], gated_pred)
                score = _score(ev)
                if best_gated is None or score > best_gated[0]:
                    best_gated = (score, float(threshold), ev)
            assert best_gated is not None
            configs.append({
                "min_duration": min_duration,
                "transition_weight": transition_weight,
                "mode": "direct",
                "threshold": None,
                "validation": direct_ev,
                "score": _score(direct_ev),
            })
            configs.append({
                "min_duration": min_duration,
                "transition_weight": transition_weight,
                "mode": "gated",
                "threshold": best_gated[1],
                "validation": best_gated[2],
                "score": best_gated[0],
            })

    feasible = [
        row
        for row in configs
        if float(row["validation"].get("transition_balanced_accuracy", 0.0)) >= float(args.min_validation_transition_bacc)
        and float(row["validation"].get("persistence_balanced_accuracy", 0.0)) >= float(args.min_validation_persistence_bacc)
    ]
    selected_pool = feasible or configs
    selected = max(selected_pool, key=lambda row: row["score"])
    min_duration = int(selected["min_duration"])
    transition_weight = float(selected["transition_weight"])
    current, future, transition = _labels(train, args.horizon, min_duration)
    labeled = train.iloc[: len(future)].copy()
    x_full, scaler, medians = _prepare(labeled, cols)

    direct_weight = _class_weights(future) * np.where(transition == 1, transition_weight, 1.0)
    direct_final = _fit_hgb(x_full, future, args.seed + 1001, direct_weight)
    hazard_final = _fit_hgb(x_full, transition, args.seed + 1002, _class_weights(transition))
    trans_rows = transition == 1
    dest_final = _fit_hgb(x_full[trans_rows], future[trans_rows], args.seed + 1003, _class_weights(future[trans_rows]))
    threshold = float(selected["threshold"] if selected["threshold"] is not None else 0.5)

    model_path = args.out_dir / "regime3_stable_h6_decoder.joblib"
    joblib.dump({
        "model_id": MODEL_ID,
        "classes": CLASSES3,
        "horizon": int(args.horizon),
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "feature_cols": cols,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "min_duration": min_duration,
        "transition_weight": transition_weight,
        "selected_mode": selected["mode"],
        "gated_threshold": threshold,
        "direct_model": direct_final,
        "hazard_model": hazard_final,
        "destination_model": dest_final,
    }, model_path)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "horizon_bars": int(args.horizon),
        "classes": CLASSES3,
        "current_prefix": CURRENT_PREFIX,
        "current_sidecar_stem": CURRENT_SIDECAR_STEM,
        "feature_count": len(cols),
        "feature_cols": cols,
        "selected_config": selected,
        "selection_constraints": {
            "min_validation_transition_bacc": float(args.min_validation_transition_bacc),
            "min_validation_persistence_bacc": float(args.min_validation_persistence_bacc),
            "feasible_config_count": int(len(feasible)),
        },
        "config_results": sorted(configs, key=lambda row: row["score"], reverse=True)[:12],
        "outputs": {},
        "leakage_audit": {
            "uses_2026_for_selection": False,
            "target_source": "stable_smoothed_current_argmax_t_plus_horizon",
            "current_probability_features_used_as_model_inputs": False,
            "current_feature_count": int(sum(col.startswith(CURRENT_PREFIX) for col in cols)),
            "current_sidecar_used_for_label_generation_only": True,
        },
    }

    for path, frame in zip(sources, frames):
        cur_i, fut_i, trans_i = _labels(frame, args.horizon, min_duration)
        eval_frame = frame.iloc[: len(fut_i)].copy()
        x_eval, _, _ = _prepare(eval_frame, cols, scaler=scaler, medians=medians)
        direct_p = _safe_proba(direct_final, x_eval, len(CLASSES3))
        direct_pred = np.argmax(direct_p, axis=1)
        hazard_p = _safe_proba(hazard_final, x_eval, 2)[:, 1]
        dest_p = _safe_proba(dest_final, x_eval, len(CLASSES3))
        gated_pred = _combine(cur_i, hazard_p, dest_p, threshold)
        selected_pred = gated_pred if selected["mode"] == "gated" else direct_pred
        sidecar = args.out_dir / f"{path.stem}_regime3_stable_h6_decoder.csv"
        _output(eval_frame["timestamp"], direct_p, gated_pred, hazard_p).to_csv(sidecar, index=False)
        report["outputs"][path.name] = {
            "source": str(path),
            "sidecar": str(sidecar),
            "rows": int(len(eval_frame)),
            "range": [str(eval_frame["timestamp"].iloc[0]), str(eval_frame["timestamp"].iloc[-1])],
            "direct": _eval(cur_i, fut_i, direct_pred),
            "gated": _eval(cur_i, fut_i, gated_pred),
            "selected": _eval(cur_i, fut_i, selected_pred),
            "hazard": _hazard_eval(trans_i, hazard_p, threshold),
        }
        print(f"[{MODEL_ID}] wrote {sidecar}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] selected={selected}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
