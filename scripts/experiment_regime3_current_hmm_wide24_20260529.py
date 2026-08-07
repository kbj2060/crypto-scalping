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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel, _json_default  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import STATE12_COLS, _with_raw_state12  # noqa: E402
from scripts.train_regime3_hmm_mamba_20260529 import CLASSES3, _current_labels3, _read  # noqa: E402


MODEL_ID = "regime3_current_hmm_wide24_experiment_20260529"
LABEL_CONFIGS = {
    "current": {
        "trend_adx_min": 22.0,
        "weak_adx_max": 18.0,
        "slope_min": 0.00025,
        "tight_bb_max": 0.018,
        "prefix_stem": "regime3_current",
    },
    "balancedish_adx16_slope15_bb012": {
        "trend_adx_min": 16.0,
        "weak_adx_max": 12.0,
        "slope_min": 0.00015,
        "tight_bb_max": 0.012,
        "prefix_stem": "regime3_current_sensitive",
    },
}
DEFAULT_TRAIN_2024 = ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv"
DEFAULT_TRANSFORMS = (
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2024.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2025.csv",
    ROOT / "tmp/causal_regen_20260516/funding_clean_splits_20260528/training_features_2026_rebuilt.csv",
)
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_wide24_experiment_20260529"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime3_current_hmm_wide24_experiment_20260529_report.json"
WIDE24_EXTRA_COLS = [
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "breakout_strength",
    "mean_reversion_z",
    "ofi_acceleration",
    "taker_acceleration",
]
DOCS_CURRENT_EXTRA_COLS = [
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "btc_volume_impulse_z",
    "vwap_dist_24",
    "vwap_dist_96",
    "cvd_12",
    "cvd_288",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "btc_lead_eth_follow_gap_3",
    "price_cvd_divergence",
    "crowding_pressure",
    "long_squeeze_risk",
    "funding_oi_divergence",
    "cvp_volume_imbalance",
    "range_contraction_breakout_dir",
    "distance_to_day_high_low_pct",
]
DOCS_CURRENT_ALL_EXTRA_COLS = DOCS_CURRENT_EXTRA_COLS + [
    "last_funding_rate",
    "funding_pressure",
    "funding_roc_288",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
    "volume_btc",
    "quote_volume_btc",
]
FEATURE_SETS = {
    "state12": list(STATE12_COLS),
    "wide24": list(STATE12_COLS) + WIDE24_EXTRA_COLS,
    "docs42": list(STATE12_COLS) + WIDE24_EXTRA_COLS + DOCS_CURRENT_EXTRA_COLS,
    "docs51all": list(STATE12_COLS) + WIDE24_EXTRA_COLS + DOCS_CURRENT_ALL_EXTRA_COLS,
}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()


def _current_labels3_thresholded(frame: pd.DataFrame, cfg: dict[str, float]) -> np.ndarray:
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_slope = (ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    bb_width = _num(frame, "bb_width", np.nan)
    if bb_width.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb_width = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    labels = np.full(len(frame), 2, dtype=np.int64)
    trending = adx.fillna(0.0).to_numpy() >= float(cfg["trend_adx_min"])
    slope = ema_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    slope_min = float(cfg["slope_min"])
    labels[trending & (slope > slope_min)] = 0
    labels[trending & (slope < -slope_min)] = 1
    labels[
        (adx.fillna(0.0).to_numpy() < float(cfg["weak_adx_max"]))
        | (bb_width.fillna(0.0).to_numpy() < float(cfg["tight_bb_max"]))
    ] = 2
    return labels


def _labels(frame: pd.DataFrame, label_mode: str) -> np.ndarray:
    if label_mode == "current":
        return _current_labels3(frame)
    if label_mode not in LABEL_CONFIGS:
        raise ValueError(f"unknown label mode: {label_mode}")
    return _current_labels3_thresholded(frame, LABEL_CONFIGS[label_mode])


def _with_features(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = _with_raw_state12(frame.copy())
    for col in cols:
        if col in out.columns:
            out[col] = _num(out, col).fillna(0.0)
        else:
            raise ValueError(f"missing current HMM feature column: {col}")
    return out


def _fit_obs(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str]):
    x_train_raw = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = pred[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    return scaler.fit_transform(x_train), scaler.transform(x_pred), scaler, medians


def _state_class_matrix(state_prob: np.ndarray, y: np.ndarray, smoothing: float = 0.02) -> np.ndarray:
    mat = np.full((state_prob.shape[1], len(CLASSES3)), smoothing, dtype=np.float64)
    for cls in range(len(CLASSES3)):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(int((y == cls).sum()), 1)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba(state_prob: np.ndarray, state_class: np.ndarray) -> np.ndarray:
    proba = state_prob @ state_class
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-300, None)
    return proba


def _run_lengths(pred: np.ndarray) -> list[int]:
    if len(pred) == 0:
        return []
    lengths: list[int] = []
    start = 0
    for i in range(1, len(pred)):
        if pred[i] != pred[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(pred) - start)
    return lengths


def _eval(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    proba = np.asarray(proba, dtype=np.float64)
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES3))))
    recalls = {}
    for i, name in enumerate(CLASSES3):
        denom = cm[i].sum()
        recalls[name] = None if denom == 0 else float(cm[i, i] / denom)
    runs = _run_lengths(pred)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=list(range(len(CLASSES3))))),
        "recall": recalls,
        "true_counts": {CLASSES3[i]: int((y == i).sum()) for i in range(len(CLASSES3))},
        "pred_counts": {CLASSES3[i]: int((pred == i).sum()) for i in range(len(CLASSES3))},
        "confusion_matrix": cm.tolist(),
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
        "median_state_duration_bars": float(np.median(runs)) if runs else 0.0,
    }


def _train_one(train: pd.DataFrame, feature_set: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    cols = FEATURE_SETS[feature_set]
    work = _with_features(train, cols)
    ts = pd.to_datetime(work["timestamp"])
    train_mask = ts < pd.Timestamp(args.val_start)
    train_part = work.loc[train_mask].copy()
    val_part = work.loc[~train_mask].copy()

    train_obs, val_obs, _, _ = _fit_obs(train_part, val_part, cols)
    val_model = GaussianStateModel(args.states, args.n_iter, args.seed, sticky=args.sticky).fit(train_obs)
    y_train = _labels(train_part, args.label_mode)
    y_val = _labels(val_part, args.label_mode)
    state_class_val = _state_class_matrix(val_model.filter_proba(train_obs), y_train)
    val_proba = _class_proba(val_model.filter_proba(val_obs), state_class_val)

    full_obs, _, scaler, medians = _fit_obs(work, work.iloc[:1].copy(), cols)
    model = GaussianStateModel(args.states, args.n_iter, args.seed + 101, sticky=args.sticky).fit(full_obs)
    y_full = _labels(work, args.label_mode)
    state_class = _state_class_matrix(model.filter_proba(full_obs), y_full)
    payload = {
        "model_id": f"{args.model_id}_{feature_set}",
        "classes": CLASSES3,
        "label_mode": args.label_mode,
        "label_config": LABEL_CONFIGS[args.label_mode],
        "prefix_stem": args.prefix_stem,
        "feature_set": feature_set,
        "feature_cols": cols,
        "feature_medians": medians.to_dict(),
        "scaler": scaler,
        "model": model,
        "state_class_matrix": state_class,
        "state_count": int(args.states),
        "sticky": float(args.sticky),
    }
    report = {
        "feature_set": feature_set,
        "label_mode": args.label_mode,
        "label_config": LABEL_CONFIGS[args.label_mode],
        "feature_cols": cols,
        "feature_count": len(cols),
        "validation": _eval(y_val, val_proba),
        "log_likelihood_validation": val_model.log_likelihood_,
        "log_likelihood_final": model.log_likelihood_,
    }
    return payload, report


def _transform(payload: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = payload["feature_cols"]
    work = _with_features(frame, cols)
    med = pd.Series(payload["feature_medians"])
    x_raw = work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    obs = payload["scaler"].transform(x_raw)
    proba = _class_proba(payload["model"].filter_proba(obs), payload["state_class_matrix"])
    y = _labels(work, payload["label_mode"])
    out = pd.DataFrame({"timestamp": work["timestamp"].reset_index(drop=True)})
    prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
    for i, name in enumerate(CLASSES3):
        out[f"{prefix}{name}_prob"] = proba[:, i]
    sp = np.sort(proba, axis=1)
    out[f"{prefix}confidence"] = sp[:, -1]
    out[f"{prefix}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / np.log(len(CLASSES3))
    out[f"{prefix}margin"] = sp[:, -1] - sp[:, -2]
    return out, _eval(y, proba)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Regime3 current HMM state12 vs wide24 feature sets.")
    p.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    p.add_argument("--transform", type=Path, action="append", default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--states", type=int, default=12)
    p.add_argument("--n-iter", type=int, default=22)
    p.add_argument("--sticky", type=float, default=0.93)
    p.add_argument("--seed", type=int, default=7529)
    p.add_argument("--feature-sets", nargs="+", default=["state12", "wide24", "docs42", "docs51all"], choices=sorted(FEATURE_SETS))
    p.add_argument("--label-mode", default="current", choices=sorted(LABEL_CONFIGS))
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--prefix-stem", default=None)
    args = p.parse_args()
    if args.prefix_stem is None:
        args.prefix_stem = str(LABEL_CONFIGS[args.label_mode]["prefix_stem"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    train = _read(args.train_2024)

    report: dict[str, Any] = {
        "model_id": args.model_id,
        "label_mode": args.label_mode,
        "label_config": LABEL_CONFIGS[args.label_mode],
        "prefix_stem": args.prefix_stem,
        "fit_source": str(args.train_2024),
        "validation_policy": "2024Q4 validation; 2025/2026 forward tests; 2026 not used for selection",
        "feature_sets": {},
        "outputs": {},
    }
    for feature_set in args.feature_sets:
        payload, one_report = _train_one(train, feature_set, args)
        model_path = args.out_dir / f"{args.prefix_stem}_hmm_{feature_set}_2024.joblib"
        joblib.dump(payload, model_path)
        one_report["model_path"] = str(model_path)
        report["feature_sets"][feature_set] = one_report
        report["outputs"][feature_set] = {}
        for src in transforms:
            frame = _read(src)
            sidecar, ev = _transform(payload, frame)
            out_path = args.out_dir / f"{src.stem}_{args.prefix_stem}_hmm_{feature_set}.csv"
            sidecar.to_csv(out_path, index=False)
            report["outputs"][feature_set][src.name] = {
                "source": str(src),
                "sidecar": str(out_path),
                "rows": int(len(frame)),
                "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
                "metrics": ev,
            }
            print(f"[{args.model_id}] {feature_set} wrote {out_path}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{args.model_id}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
