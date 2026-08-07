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
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import CLEAN_PREFIX, clean_regime_factors  # noqa: E402
from features.elite import RegimeEngine  # noqa: E402
from scripts.retrain_clean_regime_hmm_20260517 import (  # noqa: E402
    CLASSES,
    DEFAULT_TRAIN_2024,
    DEFAULT_TRANSFORMS,
    FACTOR_NAMES,
    GaussianStateModel,
    _class_proba,
    _current_labels,
    _json_default,
    _output_frame,
    _read,
    _state_class_matrix,
)


MODEL_ID = "clean_regime_2024_unsup_state7_v7_20260517"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime_state7_v7_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime_state7_v7_20260517_report.json"
STATE7_COLS = [
    "state7_trend_score",
    "state7_trend_efficiency_48",
    "state7_directional_return_48",
    "state7_volatility_state",
    "state7_sign_flip_rate_24",
    "state7_range_compression",
    "state7_flow_alignment",
]


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = s.rolling(window, min_periods=min_periods).mean().ffill()
    std = s.rolling(window, min_periods=min_periods).std().ffill().replace(0, np.nan)
    return ((s - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _with_clean_factors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    factors = clean_regime_factors(out)
    for name in FACTOR_NAMES:
        col = f"{CLEAN_PREFIX}{name}"
        out[col] = pd.to_numeric(factors[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _with_state7(frame: pd.DataFrame) -> pd.DataFrame:
    out = _with_clean_factors(frame)
    close = _num(out, "close").ffill()
    diff_abs = close.diff().abs()
    net_change_48 = close - close.shift(48)
    er_48 = (net_change_48.abs() / (diff_abs.rolling(48, min_periods=8).sum() + 1e-12)).fillna(0.0)
    ret_48 = (close / close.shift(48) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw_vol = ret.rolling(24, min_periods=4).std().fillna(0.0)
    vol_state = _zscore(raw_vol, 288, 24)

    ret_sign = np.sign(ret.where(ret.abs() >= 1e-8, np.nan)).ffill().fillna(0.0)
    sign_flip_24 = (
        (ret_sign != ret_sign.shift(1))
        .astype(float)
        .rolling(24, min_periods=4)
        .mean()
        .fillna(0.0)
    )

    high = _num(out, "high").ffill()
    low = _num(out, "low").ffill()
    range_48 = ((high.rolling(48, min_periods=8).max() - low.rolling(48, min_periods=8).min()) / close.abs().clip(lower=1e-12)).fillna(0.0)
    range_compression = -_zscore(range_48, 288, 24)
    if "bb_width_z" in out.columns:
        range_compression = 0.55 * range_compression - 0.45 * _num(out, "bb_width_z").fillna(0.0)
    if "chop_index" in out.columns:
        range_compression = range_compression + 0.15 * np.tanh((_num(out, "chop_index").fillna(50.0) - 50.0) / 20.0)

    trend_score = (
        0.32 * np.tanh(_num(out, "mtf_trend_1h").fillna(0.0) / 0.0010)
        + 0.24 * np.tanh(_num(out, "mtf_trend_4h").fillna(0.0) / 0.0007)
        + 0.20 * np.tanh(_num(out, f"{CLEAN_PREFIX}trend_bias").fillna(0.0))
        + 0.14 * np.tanh(_num(out, "hma_slope").fillna(0.0))
        + 0.10 * np.tanh(_num(out, "breakout_strength").fillna(0.0))
    )
    flow_raw = (
        0.40 * np.tanh(_num(out, "net_taker_ratio").fillna(0.0))
        + 0.28 * np.tanh(_num(out, "smart_money_flow").fillna(0.0))
        + 0.18 * np.tanh(_num(out, "taker_acceleration").fillna(0.0))
        + 0.14 * np.tanh(_num(out, "ofi_acceleration").fillna(0.0))
    )
    flow_alignment = np.sign(trend_score) * flow_raw

    engineered = {
        "state7_trend_score": np.clip(trend_score, -3.0, 3.0),
        "state7_trend_efficiency_48": np.clip(er_48, 0.0, 1.0),
        "state7_directional_return_48": np.tanh(ret_48 / 0.01),
        "state7_volatility_state": np.tanh(vol_state / 3.0),
        "state7_sign_flip_rate_24": np.clip(sign_flip_24, 0.0, 1.0),
        "state7_range_compression": np.tanh(range_compression / 3.0),
        "state7_flow_alignment": np.clip(flow_alignment, -3.0, 3.0),
    }
    for col, values in engineered.items():
        out[col] = pd.to_numeric(pd.Series(values, index=out.index), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_obs(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, RobustScaler, pd.Series]:
    x_train_raw = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = pred[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    return scaler.fit_transform(x_train), scaler.transform(x_pred), scaler, medians


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
    parser = argparse.ArgumentParser(description="Retrain clean regime HMM with seven semantic state features.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--transform", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--states", type=int, default=12)
    parser.add_argument("--n-iter", type=int, default=22)
    parser.add_argument("--seed", type=int, default=70517)
    args = parser.parse_args()

    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train = _with_state7(_read(args.train_2024))
    ts = pd.to_datetime(train["timestamp"])
    train_mask = ts < pd.Timestamp(args.val_start)
    val_mask = ~train_mask
    train_part = train.loc[train_mask].copy()
    val_part = train.loc[val_mask].copy()
    train_obs, val_obs, val_scaler, val_medians = _fit_obs(train_part, val_part, STATE7_COLS)
    val_model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed)).fit(train_obs)
    train_state = val_model.filter_proba(train_obs)
    val_state = val_model.filter_proba(val_obs)
    y_train = _current_labels(train_part)
    y_val = _current_labels(val_part)
    state_class_val = _state_class_matrix(train_state, y_train, smoothing=0.02)
    val_proba = _class_proba(val_state, state_class_val)

    full_obs, _, scaler, medians = _fit_obs(train, train.iloc[:1].copy(), STATE7_COLS)
    model = GaussianStateModel(int(args.states), int(args.n_iter), int(args.seed) + 101).fit(full_obs)
    full_state = model.filter_proba(full_obs)
    y_full = _current_labels(train)
    state_class = _state_class_matrix(full_state, y_full, smoothing=0.02)

    model_path = args.out_dir / "clean_regime_state7_v7_2024.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "clean_prefix": CLEAN_PREFIX,
            "classes": CLASSES,
            "feature_cols": STATE7_COLS,
            "feature_medians": medians.to_dict(),
            "scaler": scaler,
            "model": model,
            "state_class_matrix": state_class,
            "state_count": int(args.states),
            "notes": "HMM emission uses seven semantic engineered features without PCA.",
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
        "feature_cols": STATE7_COLS,
        "feature_count": int(len(STATE7_COLS)),
        "states": int(args.states),
        "pca_components": 0,
        "validation": _eval_report(y_val, val_proba),
        "state_class_matrix_validation": state_class_val.tolist(),
        "state_class_matrix_final": state_class.tolist(),
        "log_likelihood_validation": val_model.log_likelihood_,
        "log_likelihood_final": model.log_likelihood_,
        "outputs": {},
        "notes": [
            "Seven semantic state features replace the broad raw/PCA HMM inputs.",
            "No PCA is applied; HMM emission sees exactly these seven engineered features.",
            "No risk_off, transition, cluster id, hidden-state id, or hard label columns are written.",
        ],
    }

    for src in transforms:
        frame = _with_state7(_read(src))
        x_raw = frame[STATE7_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
        obs = scaler.transform(x_raw)
        state = model.filter_proba(obs)
        proba = _class_proba(state, state_class)
        clean = _output_frame(frame["timestamp"], frame, proba)
        sidecar = args.out_dir / f"{src.stem}_clean_regime_state7_v7.csv"
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
