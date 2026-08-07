#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.elite import RegimeEngine  # noqa: E402
from ensemble.certified_teacher_regime_moe import clean_regime_factors  # noqa: E402
from scripts.retrain_clean_regime_hmm_20260517 import (  # noqa: E402
    DEFAULT_TRAIN_2024,
    DEFAULT_TRANSFORMS,
    GaussianStateModel,
    _json_default,
    _read,
)
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import (  # noqa: E402
    STATE12_COLS,
    _fit_obs12,
    _with_raw_state12,
)


MODEL_ID = "clean_regime4_2024_unsup_raw_state12_v1_20260517"
PREFIX = "clean_regime4_2024_unsup_v1_"
CLASSES4 = ["bull", "bear", "chop", "whipsaw"]
FACTOR_COLS = [
    "factor_trend",
    "factor_flow",
    "factor_vol",
    "factor_crowding",
    "factor_liquidity",
    "trend_bias",
    "risk_off_prob",
    "transition_risk",
]
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime4_raw_state12_v1_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime4_raw_state12_v1_20260517_report.json"


def _current_labels4(frame: pd.DataFrame) -> np.ndarray:
    labeled = RegimeEngine().compute(frame.copy())
    regime_cols = ["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"]
    y5 = np.argmax(labeled[regime_cols].to_numpy(dtype=float), axis=1)
    out = np.full(len(frame), 2, dtype=int)
    out[y5 == 0] = 0
    out[y5 == 1] = 1
    out[y5 == 2] = 2
    out[y5 == 3] = 3
    normal = y5 == 4
    if normal.any():
        trend = pd.to_numeric(frame["state7_trend_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ret48 = pd.to_numeric(frame["state7_directional_return_48"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        vol = pd.to_numeric(frame["state7_volatility_state"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        flip = pd.to_numeric(frame["state7_sign_flip_rate_24"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        comp = pd.to_numeric(frame["state7_range_compression"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bull = normal & (trend > 0.10) & (ret48 >= 0.0)
        bear = normal & (trend < -0.10) & (ret48 <= 0.0)
        whipsaw = normal & ~(bull | bear) & ((flip >= 0.52) | (vol > 0.25))
        chop = normal & ~(bull | bear | whipsaw)
        out[bull] = 0
        out[bear] = 1
        out[whipsaw] = 3
        out[chop | ((comp > 0.0) & normal & ~(bull | bear | whipsaw))] = 2
    return out


def _eval_report(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CLASSES4))))),
        "true_counts": {CLASSES4[i]: int((y_true == i).sum()) for i in range(len(CLASSES4))},
        "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(len(CLASSES4)))).tolist(),
    }


def _state_class_matrix4(state_prob: np.ndarray, y: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    mat = np.full((state_prob.shape[1], len(CLASSES4)), float(smoothing), dtype=np.float64)
    counts = np.bincount(y, minlength=len(CLASSES4)).astype(np.float64)
    for cls in range(len(CLASSES4)):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(counts[cls], 1.0)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba4(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    out = state_prob @ state_class
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out


def _append_factor_auxiliary(out: pd.DataFrame, frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return out
    factors = clean_regime_factors(frame)
    old_prefix = "clean_regime_2024_unsup_v4_"
    for name in FACTOR_COLS:
        src = f"{old_prefix}{name}"
        dst = f"{PREFIX}{name}"
        if src in factors.columns:
            out[dst] = pd.to_numeric(factors[src], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        else:
            out[dst] = 0.0
    return out


def _output_frame(ts: pd.Series, proba: np.ndarray, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES4):
        out[f"{PREFIX}{name}_prob"] = proba[:, i]
    sorted_prob = np.sort(proba, axis=1)
    out[f"{PREFIX}trend_prob"] = out[f"{PREFIX}bull_prob"] + out[f"{PREFIX}bear_prob"]
    out[f"{PREFIX}micro_prob"] = out[f"{PREFIX}chop_prob"] + out[f"{PREFIX}whipsaw_prob"]
    out[f"{PREFIX}directional_bias"] = out[f"{PREFIX}bull_prob"] - out[f"{PREFIX}bear_prob"]
    out[f"{PREFIX}range_prob"] = out[f"{PREFIX}chop_prob"]
    out[f"{PREFIX}instability_prob"] = out[f"{PREFIX}whipsaw_prob"]
    out[f"{PREFIX}confidence"] = sorted_prob[:, -1]
    out[f"{PREFIX}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / math.log(len(CLASSES4))
    out[f"{PREFIX}margin"] = sorted_prob[:, -1] - sorted_prob[:, -2]
    return _append_factor_auxiliary(out, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train raw-only 4-class clean regime HMM.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--transform", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--states", type=int, default=14)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--seed", type=int, default=410517)
    parser.add_argument("--sticky", type=float, default=0.94)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-file", default="clean_regime4_raw_state12_v1_2024.joblib")
    parser.add_argument("--sidecar-suffix", default="clean_regime4_raw_state12_v1")
    args = parser.parse_args()

    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train = _with_raw_state12(_read(args.train_2024))
    ts = pd.to_datetime(train["timestamp"])
    train_mask = ts < pd.Timestamp(args.val_start)
    val_mask = ~train_mask
    train_part = train.loc[train_mask].copy()
    val_part = train.loc[val_mask].copy()
    train_obs, val_obs, _, _ = _fit_obs12(train_part, val_part)
    val_model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed), sticky=float(args.sticky)).fit(train_obs)
    train_state = val_model.filter_proba(train_obs)
    val_state = val_model.filter_proba(val_obs)
    y_train = _current_labels4(train_part)
    y_val = _current_labels4(val_part)
    state_class_val = _state_class_matrix4(train_state, y_train, smoothing=0.02)
    val_proba = _class_proba4(val_state, state_class_val)

    full_obs, _, scaler, medians = _fit_obs12(train, train.iloc[:1].copy())
    model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed) + 101, sticky=float(args.sticky)).fit(full_obs)
    full_state = model.filter_proba(full_obs)
    y_full = _current_labels4(train)
    state_class = _state_class_matrix4(full_state, y_full, smoothing=0.02)

    model_id = str(args.model_id)
    model_path = args.out_dir / str(args.model_file)
    joblib.dump(
        {
            "model_id": model_id,
            "prefix": PREFIX,
            "classes": CLASSES4,
            "feature_cols": STATE12_COLS,
            "feature_medians": medians.to_dict(),
            "scaler": scaler,
            "model": model,
            "state_class_matrix": state_class,
            "state_count": int(args.states),
            "sticky": float(args.sticky),
        },
        model_path,
    )

    report: dict[str, Any] = {
        "model_id": model_id,
        "model_path": str(model_path),
        "prefix": PREFIX,
        "fit_source": str(args.train_2024),
        "feature_cols": STATE12_COLS,
        "feature_count": len(STATE12_COLS),
        "classes": CLASSES4,
        "auxiliary_features": FACTOR_COLS,
        "states": int(args.states),
        "sticky": float(args.sticky),
        "validation": _eval_report(y_val, val_proba),
        "state_class_matrix_validation": state_class_val.tolist(),
        "state_class_matrix_final": state_class.tolist(),
        "log_likelihood_validation": val_model.log_likelihood_,
        "log_likelihood_final": model.log_likelihood_,
        "outputs": {},
        "notes": [
            "4-class clean regime removes normal as an independent class.",
            "RegimeEngine normal rows are reassigned to bull/bear/chop/whipsaw using raw state12 engineered features.",
            "No clean_regime_* feature is used as HMM input.",
            "factor/risk/transition outputs are causal auxiliary scores from raw current-row features, not extra HMM classes.",
        ],
    }

    for src in transforms:
        frame = _with_raw_state12(_read(src))
        x_raw = frame[STATE12_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        obs = scaler.transform(x_raw)
        state = model.filter_proba(obs)
        proba = _class_proba4(state, state_class)
        clean = _output_frame(frame["timestamp"], proba, frame)
        sidecar = args.out_dir / f"{src.stem}_{args.sidecar_suffix}.csv"
        clean.to_csv(sidecar, index=False)
        pred = np.argmax(proba, axis=1)
        report["outputs"][src.name] = {
            "source": str(src),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "sidecar": str(sidecar),
            "feature_count": int(len(clean.columns) - 1),
            "probability_sum_min": float(proba.sum(axis=1).min()),
            "probability_sum_max": float(proba.sum(axis=1).max()),
            "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
            "confidence_mean": float(clean[f"{PREFIX}confidence"].mean()),
            "entropy_mean": float(clean[f"{PREFIX}entropy"].mean()),
        }
        print(f"[{model_id}] wrote {sidecar} rows={len(frame)} cols={len(clean.columns) - 1}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{model_id}] model={model_path}", flush=True)
    print(f"[{model_id}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
