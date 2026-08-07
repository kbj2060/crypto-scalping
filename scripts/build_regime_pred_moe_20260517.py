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


MODEL_ID = "regime_pred_moe_20260517"
CLASSES = ["bull", "bear", "chop", "whipsaw", "normal"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
PRED_PREFIX = "regime_pred_"
CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_PREDICT_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
DEFAULT_CLEAN_2024 = ROOT / "data/ensemble/supervised/clean_regime_bgmm_v5_20260517/training_features_2024_clean_regime_bgmm_v5.csv"
DEFAULT_CLEAN_2025 = ROOT / "data/ensemble/supervised/clean_regime_bgmm_v5_20260517/training_features_2025_clean_regime_bgmm_v5.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime_pred_moe_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime_pred_moe_20260517_report.json"

SELECTED_CLEAN_FEATURES = {
    f"{CLEAN_PREFIX}factor_trend",
    f"{CLEAN_PREFIX}factor_flow",
    f"{CLEAN_PREFIX}factor_vol",
    f"{CLEAN_PREFIX}factor_crowding",
    f"{CLEAN_PREFIX}factor_liquidity",
    f"{CLEAN_PREFIX}trend_bias",
    f"{CLEAN_PREFIX}cluster_confidence",
    f"{CLEAN_PREFIX}cluster_entropy",
    f"{CLEAN_PREFIX}cluster_prob_0",
    f"{CLEAN_PREFIX}cluster_prob_1",
    f"{CLEAN_PREFIX}cluster_prob_2",
    f"{CLEAN_PREFIX}cluster_prob_3",
    f"{CLEAN_PREFIX}cluster_prob_4",
}
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


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _merge_clean(base: pd.DataFrame, clean_path: Path | None) -> pd.DataFrame:
    if clean_path is None or not clean_path.exists():
        return base.copy()
    clean = _read(clean_path)
    keep = ["timestamp"] + [c for c in clean.columns if c in SELECTED_CLEAN_FEATURES]
    out = base.merge(clean[keep], on="timestamp", how="left")
    return out.sort_values("timestamp").reset_index(drop=True)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


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
    data = {c: _num(frame, c) if c in frame.columns else pd.Series(np.nan, index=frame.index) for c in cols}
    out = pd.DataFrame(data, index=frame.index).replace([np.inf, -np.inf], np.nan)
    if medians is not None:
        out = out.fillna(medians).fillna(0.0)
    return out


def _future_path_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    open_ = _num(frame, "open").ffill().to_numpy(dtype=np.float64)
    high = _num(frame, "high").ffill().to_numpy(dtype=np.float64)
    low = _num(frame, "low").ffill().to_numpy(dtype=np.float64)
    close = _num(frame, "close").ffill().to_numpy(dtype=np.float64)
    n = len(frame)
    fut_ret = np.full(n, np.nan)
    mfe_long = np.full(n, np.nan)
    mae_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)
    for i in range(0, max(n - horizon - 1, 0)):
        entry_i = i + 1
        entry = open_[entry_i] if np.isfinite(open_[entry_i]) and open_[entry_i] > 0 else close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(entry_i + int(horizon), n)
        hi = float(np.nanmax(high[entry_i:end]))
        lo = float(np.nanmin(low[entry_i:end]))
        last = float(close[end - 1])
        fut_ret[i] = last / entry - 1.0
        mfe_long[i] = hi / entry - 1.0
        mae_long[i] = max(0.0, 1.0 - lo / entry)
        mfe_short[i] = entry / max(lo, 1e-12) - 1.0
        mae_short[i] = max(0.0, hi / entry - 1.0)
    long_quality = mfe_long - mae_long
    short_quality = mfe_short - mae_short
    range_width = mfe_long + mfe_short
    trend_efficiency = np.abs(fut_ret) / np.clip(range_width, 1e-12, None)
    return pd.DataFrame(
        {
            "_future_ret": fut_ret,
            "_mfe_long": mfe_long,
            "_mae_long": mae_long,
            "_mfe_short": mfe_short,
            "_mae_short": mae_short,
            "_long_quality": long_quality,
            "_short_quality": short_quality,
            "_range_width": range_width,
            "_trend_efficiency": trend_efficiency,
        },
        index=frame.index,
    )


def _label_thresholds(path_frame: pd.DataFrame) -> dict[str, float]:
    p = path_frame.replace([np.inf, -np.inf], np.nan).dropna()
    abs_ret = p["_future_ret"].abs()
    return {
        "abs_ret_45": float(abs_ret.quantile(0.45)),
        "abs_ret_55": float(abs_ret.quantile(0.55)),
        "range_35": float(p["_range_width"].quantile(0.35)),
        "range_55": float(p["_range_width"].quantile(0.55)),
        "range_65": float(p["_range_width"].quantile(0.65)),
        "eff_25": float(p["_trend_efficiency"].quantile(0.25)),
        "eff_45": float(p["_trend_efficiency"].quantile(0.45)),
        "eff_50": float(p["_trend_efficiency"].quantile(0.50)),
        "adverse_45": float(pd.concat([p["_mae_long"], p["_mae_short"]]).quantile(0.45)),
    }


def _labels(frame: pd.DataFrame, horizon: int, thresholds: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _future_path_frame(frame, horizon)
    valid = path.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    p = path.loc[valid].copy()
    abs_ret = p["_future_ret"].abs()
    q = dict(thresholds or _label_thresholds(p))
    labels = pd.Series("normal", index=p.index, dtype=object)
    whipsaw = (
        (p["_range_width"] >= q["range_65"])
        & (abs_ret <= q["abs_ret_55"])
        & (p["_trend_efficiency"] <= q["eff_45"])
        & (p["_mae_long"] >= q["adverse_45"])
        & (p["_mae_short"] >= q["adverse_45"])
    )
    chop = (
        ~whipsaw
        & (
            ((p["_range_width"] <= q["range_35"]) & (abs_ret <= q["abs_ret_55"]))
            | ((p["_trend_efficiency"] <= q["eff_25"]) & (p["_range_width"] <= q["range_55"]))
        )
    )
    bull = (
        ~(whipsaw | chop)
        & (p["_future_ret"] > q["abs_ret_45"])
        & (p["_trend_efficiency"] >= q["eff_50"])
        & (p["_long_quality"] > p["_short_quality"])
        & (p["_mfe_long"] > p["_mfe_short"] * 1.02)
    )
    bear = (
        ~(whipsaw | chop | bull)
        & (p["_future_ret"] < -q["abs_ret_45"])
        & (p["_trend_efficiency"] >= q["eff_50"])
        & (p["_short_quality"] > p["_long_quality"])
        & (p["_mfe_short"] > p["_mfe_long"] * 1.02)
    )
    labels.loc[whipsaw] = "whipsaw"
    labels.loc[chop] = "chop"
    labels.loc[bull] = "bull"
    labels.loc[bear] = "bear"
    p["_label_name"] = labels
    p["_label_id"] = labels.map(CLASS_TO_ID).astype(int)
    meta = {
        "horizon": int(horizon),
        "thresholds": q,
        "label_counts": {k: int(v) for k, v in labels.value_counts().reindex(CLASSES, fill_value=0).items()},
        "label_share": {k: float(v) for k, v in labels.value_counts(normalize=True).reindex(CLASSES, fill_value=0.0).items()},
    }
    return p, meta


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    total = float(counts.sum())
    weights = total / np.clip(len(CLASSES) * counts, 1.0, None)
    return weights[y]


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, seed: int) -> Any:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASSES),
        n_estimators=180,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=45,
        reg_alpha=0.20,
        reg_lambda=1.60,
        subsample=0.82,
        subsample_freq=1,
        colsample_bytree=0.82,
        path_smooth=4.0,
        extra_trees=True,
        random_state=int(seed),
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model


def _predict_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = [int(c) for c in np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])), dtype=int)]
    out = np.zeros((len(x), len(CLASSES)), dtype=np.float64)
    for j, cls in enumerate(classes):
        if 0 <= cls < len(CLASSES):
            out[:, cls] = raw[:, j]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def _output_frame(ts: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": ts.reset_index(drop=True)})
    for i, name in enumerate(CLASSES):
        out[f"{PRED_PREFIX}{name}_prob"] = proba[:, i]
    sorted_prob = np.sort(proba, axis=1)
    confidence = sorted_prob[:, -1]
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / math.log(len(CLASSES))
    label_id = np.argmax(proba, axis=1)
    out[f"{PRED_PREFIX}trend_prob"] = out[f"{PRED_PREFIX}bull_prob"] + out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}micro_prob"] = out[f"{PRED_PREFIX}chop_prob"] + out[f"{PRED_PREFIX}whipsaw_prob"] + out[f"{PRED_PREFIX}normal_prob"]
    out[f"{PRED_PREFIX}directional_bias"] = out[f"{PRED_PREFIX}bull_prob"] - out[f"{PRED_PREFIX}bear_prob"]
    out[f"{PRED_PREFIX}range_prob"] = out[f"{PRED_PREFIX}chop_prob"] + out[f"{PRED_PREFIX}normal_prob"]
    out[f"{PRED_PREFIX}instability_prob"] = out[f"{PRED_PREFIX}whipsaw_prob"]
    out[f"{PRED_PREFIX}confidence"] = confidence
    out[f"{PRED_PREFIX}entropy"] = entropy
    out[f"{PRED_PREFIX}margin"] = margin
    return out


def _eval_report(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(len(CLASSES))))
    return {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "log_loss": float(log_loss(y_true, proba, labels=list(range(len(CLASSES))))),
        "true_counts": {CLASSES[i]: int((y_true == i).sum()) for i in range(len(CLASSES))},
        "pred_counts": {CLASSES[i]: int((pred == i).sum()) for i in range(len(CLASSES))},
        "confusion_matrix": cm.tolist(),
    }


def _predicted_path_diagnostics(frame: pd.DataFrame, output: pd.DataFrame, proba: np.ndarray, horizon: int) -> dict[str, Any]:
    path = _future_path_frame(frame, horizon)
    joined = output.join(path)
    joined["_pred_label"] = [CLASSES[i] for i in np.argmax(proba, axis=1)]
    rows: dict[str, Any] = {}
    for cls in CLASSES:
        sub = joined[joined["_pred_label"] == cls]
        rows[cls] = {
            "rows": int(len(sub)),
            "future_ret_mean": float(pd.to_numeric(sub.get("_future_ret", 0.0), errors="coerce").mean() if len(sub) else 0.0),
            "range_width_mean": float(pd.to_numeric(sub.get("_range_width", 0.0), errors="coerce").mean() if len(sub) else 0.0),
            "long_quality_mean": float(pd.to_numeric(sub.get("_long_quality", 0.0), errors="coerce").mean() if len(sub) else 0.0),
            "short_quality_mean": float(pd.to_numeric(sub.get("_short_quality", 0.0), errors="coerce").mean() if len(sub) else 0.0),
            "trend_efficiency_mean": float(pd.to_numeric(sub.get("_trend_efficiency", 0.0), errors="coerce").mean() if len(sub) else 0.0),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 5-class MoE trading regime predictor features.")
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--predict-2025", type=Path, default=DEFAULT_PREDICT_2025)
    parser.add_argument("--clean-2024", type=Path, default=DEFAULT_CLEAN_2024)
    parser.add_argument("--clean-2025", type=Path, default=DEFAULT_CLEAN_2025)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--seed", type=int, default=517)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train_raw = _merge_clean(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean(_read(args.predict_2025), args.clean_2025)
    val_start = pd.Timestamp(args.val_start)
    raw_ts = pd.to_datetime(train_raw["timestamp"])
    raw_train_mask = raw_ts < val_start
    raw_val_mask = raw_ts >= val_start
    if int(raw_train_mask.sum()) < 1000 or int(raw_val_mask.sum()) < 1000:
        split = int(len(train_raw) * 0.80)
        raw_train_mask = pd.Series(np.arange(len(train_raw)) < split, index=train_raw.index)
        raw_val_mask = ~raw_train_mask

    threshold_path = _future_path_frame(train_raw.loc[raw_train_mask].copy(), int(args.horizon))
    train_only_thresholds = _label_thresholds(threshold_path)
    label_frame, label_meta = _labels(train_raw, int(args.horizon), thresholds=train_only_thresholds)
    train_labeled = train_raw.loc[label_frame.index].copy().join(label_frame[["_label_name", "_label_id"]])
    cols = _feature_cols(train_labeled, pred_raw)
    if len(cols) < 10:
        raise ValueError(f"not enough feature columns: {len(cols)}")

    ts = pd.to_datetime(train_labeled["timestamp"])
    train_mask = ts < val_start
    val_mask = ts >= val_start
    if int(train_mask.sum()) < 1000 or int(val_mask.sum()) < 1000:
        split = int(len(train_labeled) * 0.80)
        train_mask = pd.Series(np.arange(len(train_labeled)) < split, index=train_labeled.index)
        val_mask = ~train_mask

    x_train_raw = _matrix(train_labeled.loc[train_mask], cols)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    y_train = train_labeled.loc[train_mask, "_label_id"].astype(int).to_numpy()
    x_val = _matrix(train_labeled.loc[val_mask], cols, medians)
    y_val = train_labeled.loc[val_mask, "_label_id"].astype(int).to_numpy()
    val_model = _fit_lgbm(x_train, y_train, _class_weights(y_train), int(args.seed))
    val_proba = _predict_proba(val_model, x_val)

    full_label_frame, full_label_meta = _labels(train_raw, int(args.horizon))
    full_train_labeled = train_raw.loc[full_label_frame.index].copy().join(full_label_frame[["_label_name", "_label_id"]])
    x_full_raw = _matrix(full_train_labeled, cols)
    full_medians = x_full_raw.median(numeric_only=True).fillna(0.0)
    x_full = x_full_raw.fillna(full_medians).fillna(0.0)
    y_full = full_train_labeled["_label_id"].astype(int).to_numpy()
    final_model = _fit_lgbm(x_full, y_full, _class_weights(y_full), int(args.seed) + 101)
    pred_x = _matrix(pred_raw, cols, full_medians)
    pred_proba = _predict_proba(final_model, pred_x)
    pred_output = _output_frame(pred_raw["timestamp"], pred_proba)

    train_output = _output_frame(full_train_labeled["timestamp"], _predict_proba(final_model, x_full))
    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime_pred_moe.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime_pred_moe.csv"
    model_path = args.out_dir / "regime_pred_moe_2024.joblib"
    pred_output.to_csv(pred_sidecar, index=False)
    train_output.to_csv(train_sidecar, index=False)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "classes": CLASSES,
            "feature_cols": cols,
            "feature_medians": full_medians.to_dict(),
            "model": final_model,
            "horizon": int(args.horizon),
            "selected_clean_features": sorted(SELECTED_CLEAN_FEATURES),
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
        "validation_label_meta": {
            **label_meta,
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
        "forbidden_outputs_excluded": sorted(FORBIDDEN_EXACT),
        "validation": _eval_report(y_val, val_proba),
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_proba.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_proba.sum(axis=1).max()),
        "predict_counts": {CLASSES[i]: int((np.argmax(pred_proba, axis=1) == i).sum()) for i in range(len(CLASSES))},
        "predict_confidence_mean": float(pred_output[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(pred_output[f"{PRED_PREFIX}entropy"].mean()),
        "predict_path_diagnostics": _predicted_path_diagnostics(pred_raw, pred_output, pred_proba, int(args.horizon)),
        "notes": [
            "5-class supervised trading-regime predictor for MoE expert routing.",
            "risk_off and transition are not classes or outputs.",
            "clean_regime BGMM is used only as selected prior/embedding features.",
            "Final MoE routing should consume regime_pred_* probabilities, not clean_regime_* cluster ids.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] train_sidecar={train_sidecar}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
