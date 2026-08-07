#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX  # noqa: E402
from scripts.retrain_clean_regime_hmm_20260517 import (  # noqa: E402
    CLASSES,
    DEFAULT_TRAIN_2024,
    DEFAULT_TRANSFORMS,
    GaussianStateModel,
    _class_proba,
    _current_labels,
    _json_default,
    _read,
    _state_class_matrix,
)
from scripts.retrain_clean_regime_hmm_raw_state7_20260517 import (  # noqa: E402
    STATE7_COLS,
    _eval_report,
    _fit_obs,
    _num,
    _with_raw_state7,
)


MODEL_ID = "clean_regime_2024_unsup_raw_state12_v9_20260517"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime_raw_state12_v9_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime_raw_state12_v9_20260517_report.json"
RAW5_COLS = [
    "state12_log_return",
    "state12_garman_klass_vol",
    "state12_net_taker_ratio",
    "state12_oi_change_rate",
    "state12_chop_index",
]
STATE12_COLS = STATE7_COLS + RAW5_COLS


def _with_raw_state12(frame: pd.DataFrame) -> pd.DataFrame:
    out = _with_raw_state7(frame)
    out["state12_log_return"] = np.tanh(_num(out, "log_return").fillna(0.0) / 0.003)
    out["state12_garman_klass_vol"] = np.tanh(_num(out, "garman_klass_vol").fillna(0.0) / 0.00002)
    out["state12_net_taker_ratio"] = np.tanh(_num(out, "net_taker_ratio").fillna(0.0))
    out["state12_oi_change_rate"] = np.tanh(_num(out, "oi_change_rate").fillna(0.0) / 0.01)
    out["state12_chop_index"] = np.tanh((_num(out, "chop_index", 50.0).fillna(50.0) - 50.0) / 20.0)
    for col in STATE12_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_obs12(train: pd.DataFrame, pred: pd.DataFrame):
    x_train_raw = train[STATE12_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = pred[STATE12_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    return scaler.fit_transform(x_train), scaler.transform(x_pred), scaler, medians


def _output_frame(ts: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    import math

    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain raw-only clean regime HMM with state7 plus five direct raw inputs.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--transform", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--states", type=int, default=14)
    parser.add_argument("--n-iter", type=int, default=24)
    parser.add_argument("--seed", type=int, default=90517)
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
    val_model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed)).fit(train_obs)
    train_state = val_model.filter_proba(train_obs)
    val_state = val_model.filter_proba(val_obs)
    y_train = _current_labels(train_part)
    y_val = _current_labels(val_part)
    state_class_val = _state_class_matrix(train_state, y_train, smoothing=0.02)
    val_proba = _class_proba(val_state, state_class_val)

    full_obs, _, scaler, medians = _fit_obs12(train, train.iloc[:1].copy())
    model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed) + 101).fit(full_obs)
    full_state = model.filter_proba(full_obs)
    y_full = _current_labels(train)
    state_class = _state_class_matrix(full_state, y_full, smoothing=0.02)

    model_path = args.out_dir / "clean_regime_raw_state12_v9_2024.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "clean_prefix": CLEAN_PREFIX,
            "classes": CLASSES,
            "feature_cols": STATE12_COLS,
            "feature_medians": medians.to_dict(),
            "scaler": scaler,
            "model": model,
            "state_class_matrix": state_class,
            "state_count": int(args.states),
            "notes": "Raw-only HMM emission uses seven semantic features plus log_return, garman_klass_vol, net_taker_ratio, oi_change_rate, chop_index.",
        },
        model_path,
    )

    report: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "clean_prefix": CLEAN_PREFIX,
        "fit_source": str(args.train_2024),
        "fit_rows": int(len(train)),
        "fit_range": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
        "feature_cols": STATE12_COLS,
        "feature_count": int(len(STATE12_COLS)),
        "states": int(args.states),
        "pca_components": 0,
        "validation": _eval_report(y_val, val_proba),
        "state_class_matrix_validation": state_class_val.tolist(),
        "state_class_matrix_final": state_class.tolist(),
        "log_likelihood_validation": val_model.log_likelihood_,
        "log_likelihood_final": model.log_likelihood_,
        "outputs": {},
        "notes": [
            "Raw-only inputs; no clean_regime_* feature is used as an HMM input.",
            "Adds five direct raw inputs requested for direction, volatility, flow, transition, and sideways detection.",
            "No risk_off, transition, cluster id, hidden-state id, factor columns, or hard label columns are written.",
        ],
    }

    for src in transforms:
        frame = _with_raw_state12(_read(src))
        x_raw = frame[STATE12_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        obs = scaler.transform(x_raw)
        state = model.filter_proba(obs)
        proba = _class_proba(state, state_class)
        clean = _output_frame(frame["timestamp"], proba)
        sidecar = args.out_dir / f"{src.stem}_clean_regime_raw_state12_v9.csv"
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
