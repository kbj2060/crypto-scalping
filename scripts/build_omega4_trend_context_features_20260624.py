#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_trend_context_features_20260624"


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _read_trend_labels(label_dir: Path, year: int, frame: pd.DataFrame, name: str) -> pd.DataFrame:
    path = Path(label_dir) / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = ["timestamp", "zigzag_action", "trend_scan_t_value", "trend_scan_horizon", "trend_scan_forward_log_return"]
    labels = pd.read_csv(path, usecols=cols, parse_dates=["timestamp"], low_memory=False)
    labels = labels.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    aligned_frame, aligned_labels = omega._align(frame[["timestamp"]], labels, name)
    if len(aligned_frame) != len(frame):
        raise RuntimeError(f"{name}: trend label alignment changed row count: {len(frame)} -> {len(aligned_frame)}")
    out = aligned_labels.reset_index(drop=True).copy()
    out["zigzag_action"] = pd.to_numeric(out["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(out["zigzag_action"]).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"{path} invalid trend action classes: {invalid}")
    for col in ("trend_scan_t_value", "trend_scan_horizon", "trend_scan_forward_log_return"):
        out[col] = pd.to_numeric(out[col], errors="raise").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _class_weight(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.int64)
    counts = pd.Series(arr).value_counts().to_dict()
    n = max(len(arr), 1)
    k = max(len(counts), 1)
    return np.asarray([n / (k * max(int(counts[int(v)]), 1)) for v in arr], dtype=np.float64)


def _make_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        l2_regularization=0.10,
        max_leaf_nodes=15,
        min_samples_leaf=28,
        random_state=int(seed),
    )


def _make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.045,
        l2_regularization=0.10,
        max_leaf_nodes=15,
        min_samples_leaf=28,
        random_state=int(seed),
    )


def _predict_proba_3(model: HistGradientBoostingClassifier, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for src_i, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, int(src_i)]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return out / row_sum


def _fit_models(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    seed: int,
) -> tuple[HistGradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    y_action = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    signed_t = pd.to_numeric(labels["trend_scan_t_value"], errors="raise").to_numpy(dtype=np.float64)
    horizon = pd.to_numeric(labels["trend_scan_horizon"], errors="raise").to_numpy(dtype=np.float64)
    clf = _make_classifier(int(seed))
    clf.fit(x, y_action, sample_weight=_class_weight(y_action))
    t_model = _make_regressor(int(seed) + 101)
    t_model.fit(x, signed_t)
    h_model = _make_regressor(int(seed) + 202)
    h_model.fit(x, horizon)
    return clf, t_model, h_model


def _predict_context(
    x: pd.DataFrame,
    *,
    clf: HistGradientBoostingClassifier | None,
    t_model: HistGradientBoostingRegressor | None,
    h_model: HistGradientBoostingRegressor | None,
    oof_valid: np.ndarray,
) -> pd.DataFrame:
    if clf is None or t_model is None or h_model is None:
        prob = np.zeros((len(x), 3), dtype=np.float64)
        prob[:, 0] = 1.0
        signed_t = np.zeros(len(x), dtype=np.float64)
        horizon = np.zeros(len(x), dtype=np.float64)
    else:
        prob = _predict_proba_3(clf, x)
        signed_t = np.asarray(t_model.predict(x), dtype=np.float64)
        horizon = np.asarray(h_model.predict(x), dtype=np.float64)
    top2 = np.sort(prob, axis=1)[:, -2:]
    pred_action = prob.argmax(axis=1).astype(np.int64)
    pred_side = np.where(pred_action == 1, 1.0, np.where(pred_action == 2, -1.0, 0.0))
    out = pd.DataFrame(
        {
            "trend_ctx_prob_cash": prob[:, 0],
            "trend_ctx_prob_long": prob[:, 1],
            "trend_ctx_prob_short": prob[:, 2],
            "trend_ctx_long_minus_short": prob[:, 1] - prob[:, 2],
            "trend_ctx_active_prob": prob[:, 1] + prob[:, 2],
            "trend_ctx_pred_side": pred_side,
            "trend_ctx_side_confidence": top2[:, 1] - top2[:, 0],
            "trend_ctx_signed_t_pred": signed_t,
            "trend_ctx_abs_t_pred": np.abs(signed_t),
            "trend_ctx_horizon_pred": np.clip(horizon, 0.0, None),
            "trend_ctx_oof_valid": np.asarray(oof_valid, dtype=np.float64),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _expanding_oof_context(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    seed: int,
    folds: int,
    min_train_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(x)
    if n < 1000:
        raise RuntimeError(f"too few rows for expanding OOF trend context: {n}")
    start = int(round(n * float(min_train_fraction)))
    start = min(max(start, 1), n - 1)
    edges = np.linspace(start, n, int(folds) + 1, dtype=np.int64)
    out = _predict_context(x, clf=None, t_model=None, h_model=None, oof_valid=np.zeros(n, dtype=np.float64))
    fold_rows: list[dict[str, Any]] = []
    for fold_i in range(int(folds)):
        train_end = int(edges[fold_i])
        test_start = int(edges[fold_i])
        test_end = int(edges[fold_i + 1])
        if test_end <= test_start:
            continue
        clf, t_model, h_model = _fit_models(x.iloc[:train_end], labels.iloc[:train_end], seed=int(seed) + fold_i * 17)
        pred = _predict_context(
            x.iloc[test_start:test_end],
            clf=clf,
            t_model=t_model,
            h_model=h_model,
            oof_valid=np.ones(test_end - test_start, dtype=np.float64),
        )
        out.iloc[test_start:test_end, :] = pred.to_numpy(dtype=np.float32)
        fold_rows.append(
            {
                "fold": int(fold_i),
                "train_rows": int(train_end),
                "test_rows": int(test_end - test_start),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
    return out, {"rows": int(n), "neutral_prefix_rows": int(start), "folds": fold_rows}


def _prediction_diag(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    prob = features[["trend_ctx_prob_cash", "trend_ctx_prob_long", "trend_ctx_prob_short"]].to_numpy(dtype=np.float64)
    pred = prob.argmax(axis=1).astype(np.int64)
    valid = pd.to_numeric(features["trend_ctx_oof_valid"], errors="raise").to_numpy(dtype=np.float64) > 0.5
    scope = valid if bool(valid.any()) else np.ones(len(features), dtype=bool)
    counts = pd.Series(y).value_counts().sort_index()
    pred_counts = pd.Series(pred).value_counts().sort_index()
    return {
        "rows": int(len(features)),
        "valid_rows": int(scope.sum()),
        "label_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "pred_counts": {str(int(k)): int(v) for k, v in pred_counts.items()},
        "label_active_ratio": float((y != 0).mean()) if len(y) else 0.0,
        "pred_active_ratio": float((pred != 0).mean()) if len(pred) else 0.0,
        "accuracy_valid_scope": float(accuracy_score(y[scope], pred[scope])) if bool(scope.any()) else 0.0,
        "balanced_accuracy_valid_scope": float(balanced_accuracy_score(y[scope], pred[scope])) if bool(scope.any()) else 0.0,
        "active_prob_mean": float(features["trend_ctx_active_prob"].mean()),
        "abs_t_pred_mean": float(features["trend_ctx_abs_t_pred"].mean()),
        "horizon_pred_mean": float(features["trend_ctx_horizon_pred"].mean()),
    }


def _write_split(out_dir: Path, split: str, frame: pd.DataFrame, features: pd.DataFrame) -> Path:
    out = pd.concat([frame[["timestamp"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    path = Path(out_dir) / f"{split}_context_features.csv"
    out.to_csv(path, index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build OOF/holdout trend-context prediction features for Omega4 risk sidecar.")
    ap.add_argument("--baseline-bundle", type=Path, required=True)
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--quality-mode", choices=["same_as_direction"], default="same_as_direction")
    ap.add_argument("--trend-label-dir", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--eval-csv", type=Path, default=None)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--oof-min-train-fraction", type=float, default=0.25)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=260624)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    if args.train_csv is not None:
        omega.TRAIN_CSV = Path(args.train_csv)
    if args.eval_csv is not None:
        omega.EVAL_CSV = Path(args.eval_csv)
    device = parent._device(str(args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )

    print("stage=predict_parent", flush=True)
    x_train, train_src, train_dec_base = risk_sidecar._predict_decisions(
        frames["train_raw"],
        oof=True,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )
    x_val, val_src, val_dec_base = risk_sidecar._predict_decisions(
        frames["val_raw"],
        oof=True,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )
    x_oos, oos_src, oos_dec_base = risk_sidecar._predict_decisions(
        frames["oos_raw"],
        oof=False,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )

    print("stage=apply_atr_contract", flush=True)
    train_dec, train_atr_diag = atr_eval._apply_atr_safety_sltp(
        train_dec_base,
        frames["train_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )
    val_dec, val_atr_diag = atr_eval._apply_atr_safety_sltp(
        val_dec_base,
        frames["val_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )
    oos_dec, oos_atr_diag = atr_eval._apply_atr_safety_sltp(
        oos_dec_base,
        frames["oos_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )

    print("stage=build_context_model_features", flush=True)
    train_x = risk_sidecar._risk_feature_frame(
        frames["train_raw"],
        train_src,
        train_dec,
        base_cols,
        atr_pct=atr_eval._atr_pct(frames["train_raw"], int(args.atr_window)),
        feature_mode="parent_outputs",
    )
    val_x = risk_sidecar._risk_feature_frame(
        frames["val_raw"],
        val_src,
        val_dec,
        base_cols,
        atr_pct=atr_eval._atr_pct(frames["val_raw"], int(args.atr_window)),
        feature_mode="parent_outputs",
    )
    oos_x = risk_sidecar._risk_feature_frame(
        frames["oos_raw"],
        oos_src,
        oos_dec,
        base_cols,
        atr_pct=atr_eval._atr_pct(frames["oos_raw"], int(args.atr_window)),
        feature_mode="parent_outputs",
    )

    print("stage=align_trend_labels", flush=True)
    train_labels = _read_trend_labels(Path(args.trend_label_dir), 2025, frames["train_raw"], "train trend context labels")
    val_labels = _read_trend_labels(Path(args.trend_label_dir), 2025, frames["val_raw"], "validation trend context labels")
    oos_labels = _read_trend_labels(Path(args.trend_label_dir), 2026, frames["oos_raw"], "oos trend context labels")

    print("stage=train_oof_trend_context", flush=True)
    train_ctx, oof_diag = _expanding_oof_context(
        train_x,
        train_labels,
        seed=int(args.seed),
        folds=int(args.oof_folds),
        min_train_fraction=float(args.oof_min_train_fraction),
    )

    print("stage=train_final_trend_context", flush=True)
    clf, t_model, h_model = _fit_models(train_x, train_labels, seed=int(args.seed) + 1000)
    val_ctx = _predict_context(
        val_x,
        clf=clf,
        t_model=t_model,
        h_model=h_model,
        oof_valid=np.ones(len(val_x), dtype=np.float64),
    )
    oos_ctx = _predict_context(
        oos_x,
        clf=clf,
        t_model=t_model,
        h_model=h_model,
        oof_valid=np.ones(len(oos_x), dtype=np.float64),
    )

    print("stage=write_artifacts", flush=True)
    paths = {
        "train": str(_write_split(out_dir, "train", frames["train_raw"], train_ctx)),
        "validation": str(_write_split(out_dir, "validation", frames["val_raw"], val_ctx)),
        "oos": str(_write_split(out_dir, "oos", frames["oos_raw"], oos_ctx)),
    }
    audit = {
        "artifact_id": "omega4_trend_context_features_20260624",
        "contract": {
            "label_use": "trend-scanning raw labels are used only to train a context predictor; risk sidecar receives OOF/holdout predictions only",
            "train_prediction": "expanding-window OOF; initial prefix neutral with trend_ctx_oof_valid=0",
            "validation_oos_prediction": "trend context model fit on train split only",
            "feature_prefix": "trend_ctx_",
        },
        "baseline_bundle": str(args.baseline_bundle),
        "direction_label_dir": str(args.direction_label_dir),
        "trend_label_dir": str(args.trend_label_dir),
        "paths": paths,
        "oof": oof_diag,
        "atr_diag": {"train": train_atr_diag, "validation": val_atr_diag, "oos": oos_atr_diag},
        "diagnostics": {
            "train": _prediction_diag(train_ctx, train_labels),
            "validation": _prediction_diag(val_ctx, val_labels),
            "oos": _prediction_diag(oos_ctx, oos_labels),
        },
    }
    (out_dir / "trend_context_feature_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "audit": str(out_dir / "trend_context_feature_audit.json"), "diagnostics": audit["diagnostics"]}, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
