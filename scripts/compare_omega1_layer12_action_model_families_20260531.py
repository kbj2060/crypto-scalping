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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_omega1_layer12_action_confidence_20260531 as base


MODEL_ID = "omega1_layer12_action_model_family_compare_20260531"
DEFAULT_OUT_DIR = base.ROOT / "tmp/causal_regen_20260516/omega1_layer12_action_model_family_compare_20260531"


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _sample_weight(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y)


def _model_specs(seed: int) -> dict[str, Any]:
    return {
        "hgb": {
            "family": "sklearn_hist_gradient_boosting",
            "model": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            loss="log_loss",
                            learning_rate=0.035,
                            max_iter=260,
                            max_leaf_nodes=31,
                            l2_regularization=0.10,
                            min_samples_leaf=45,
                            early_stopping=True,
                            validation_fraction=0.12,
                            n_iter_no_change=30,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "fit_params": "model__sample_weight",
        },
        "catboost": {
            "family": "catboost_multiclass",
            "model": CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1",
                iterations=900,
                depth=6,
                learning_rate=0.035,
                l2_leaf_reg=5.0,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
            "fit_params": "sample_weight",
        },
        "lgbm": {
            "family": "lightgbm_multiclass",
            "model": LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=800,
                learning_rate=0.025,
                num_leaves=31,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=-1,
                verbosity=-1,
            ),
            "fit_params": "sample_weight",
        },
        "xgb": {
            "family": "xgboost_multiclass_hist",
            "model": XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=600,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=-1,
            ),
            "fit_params": "sample_weight",
        },
        "extratrees": {
            "family": "extra_trees_balanced",
            "model": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=220,
                            max_depth=10,
                            min_samples_leaf=60,
                            max_features=0.55,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "fit_params": None,
        },
        "rf": {
            "family": "random_forest_balanced",
            "model": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=180,
                            max_depth=9,
                            min_samples_leaf=70,
                            max_features=0.50,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "fit_params": None,
        },
        "logistic": {
            "family": "standardized_multinomial_logistic",
            "model": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.35,
                            class_weight="balanced",
                            max_iter=800,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "fit_params": None,
        },
    }


def _fit(spec: dict[str, Any], x: pd.DataFrame, y: np.ndarray) -> Any:
    model = spec["model"]
    key = spec.get("fit_params")
    if key:
        model.fit(x, y, **{key: _sample_weight(y)})
    else:
        model.fit(x, y)
    return model


def _classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_, dtype=int)
    if isinstance(model, Pipeline):
        return np.asarray(model.steps[-1][1].classes_, dtype=int)
    raise TypeError(f"cannot resolve classes for {type(model)}")


def _proba3(model: Any, x: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = _classes(model)
    full = np.zeros((len(x), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]
    row_sum = full.sum(axis=1)
    bad = row_sum <= 0.0
    if bad.any():
        raise RuntimeError(f"model produced empty probability rows: {int(bad.sum())}")
    return full / row_sum[:, None]


def _metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    return base._classification_metrics(y, proba, threshold)


def _select_threshold(y: np.ndarray, proba: np.ndarray, val_frame: pd.DataFrame, args: argparse.Namespace, grid: list[float]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        m = _metrics(y, proba, threshold)
        dec = base._decisions(val_frame.reset_index(drop=True), proba, threshold)
        cost = base._cost_metrics(val_frame.reset_index(drop=True), dec, args)
        c3 = cost["cost3"]
        calmar = float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))
        # Select execution threshold by validation Cost3 only. Label metrics are
        # diagnostic here; rewarding trade count makes losing dense policies win.
        score = calmar
        rows.append({"threshold": float(threshold), "score": float(score), "metrics": m, "validation_backtest": cost})
    best = max(rows, key=lambda r: float(r["score"]))
    return float(best["threshold"]), rows


def _prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    train = base._read_csv(args.split_dir / "training_features_2025.csv")
    oos = base._read_csv(args.split_dir / "training_features_2026_rebuilt.csv")
    train = base._add_layer2(train, 2025, args)
    oos = base._add_layer2(oos, 2026, args)
    train = base._add_labels(train, 2025, args.label_dir)
    oos = base._add_labels(oos, 2026, args.label_dir)
    feature_sets = base._feature_sets(train, oos)
    keep_names = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    if keep_names:
        missing = sorted(set(keep_names) - set(feature_sets))
        if missing:
            raise ValueError(f"unknown feature sets: {missing}; available={sorted(feature_sets)}")
        feature_sets = {name: feature_sets[name] for name in keep_names}
    return train, oos, feature_sets


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Omega1 Layer1+Layer2 action/confidence model families.")
    parser.add_argument("--split-dir", type=Path, default=base.DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=base.DEFAULT_LABEL_DIR)
    parser.add_argument("--ai-dir", type=Path, default=base.DEFAULT_AI_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=base.DEFAULT_CHRONOS_DIR)
    parser.add_argument("--regime3-stability-dir", type=Path, default=base.DEFAULT_REGIME3_STABILITY_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=base.DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=base.DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--dir3-patch-dir", type=Path, default=base.DEFAULT_DIR3_PATCH_DIR)
    parser.add_argument("--dir3-vsnlstm-dir", type=Path, default=base.DEFAULT_DIR3_VSNLSTM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-start", default="2025-10-01")
    parser.add_argument("--confidence-grid", default="0.34,0.38,0.42,0.46,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--feature-sets", default="l1all_safe_layer2,architect_strict_l1all_layer2")
    parser.add_argument("--models", default="hgb,catboost,lgbm,xgb,extratrees,rf,logistic")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--tp-pct", type=float, default=0.018)
    parser.add_argument("--sl-pct", type=float, default=0.010)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slip", type=float, default=0.00015)
    parser.add_argument("--exposure", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, oos, feature_sets = _prepare(args)
    y_all = train["zigzag_action"].astype(int).to_numpy()
    y_oos = oos["zigzag_action"].astype(int).to_numpy()
    val_mask = pd.to_datetime(train["timestamp"]) >= pd.Timestamp(args.val_start)
    fit_idx = np.flatnonzero(~val_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    if len(fit_idx) < 1000 or len(val_idx) < 1000:
        raise RuntimeError(f"bad 2025 fit/validation split: fit={len(fit_idx)} val={len(val_idx)}")

    thresholds = [float(x.strip()) for x in str(args.confidence_grid).split(",") if x.strip()]
    model_names = [x.strip() for x in str(args.models).split(",") if x.strip()]
    specs = _model_specs(int(args.seed))
    unknown = sorted(set(model_names) - set(specs))
    if unknown:
        raise ValueError(f"unknown model names: {unknown}; available={sorted(specs)}")

    runs: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for feature_set, cols in feature_sets.items():
        for model_name in model_names:
            spec = _model_specs(int(args.seed))[model_name]
            model = _fit(spec, train.iloc[fit_idx][cols], y_all[fit_idx])
            val_proba = _proba3(model, train.iloc[val_idx][cols])
            threshold, threshold_grid = _select_threshold(
                y_all[val_idx],
                val_proba,
                train.iloc[val_idx],
                args,
                thresholds,
            )
            val_metrics = _metrics(y_all[val_idx], val_proba, threshold)
            val_dec = base._decisions(train.iloc[val_idx].reset_index(drop=True), val_proba, threshold)
            val_cost = base._cost_metrics(train.iloc[val_idx].reset_index(drop=True), val_dec, args)
            c3 = val_cost["cost3"]
            val_calmar = float(c3["pnl"] / max(abs(float(c3["mdd"])), 1e-9))
            run = {
                "model_name": model_name,
                "model_family": spec["family"],
                "feature_set": feature_set,
                "feature_count": int(len(cols)),
                "threshold": float(threshold),
                "validation": val_metrics,
                "validation_backtest": val_cost,
                "validation_cost3_calmar": val_calmar,
                "threshold_grid": threshold_grid,
                "selection_score": float(val_calmar),
            }
            runs.append(run)
            print(json.dumps({"run": run}, ensure_ascii=False, default=_json_default), flush=True)
            if best is None or float(run["selection_score"]) > float(best["selection_score"]):
                best = {**run, "feature_cols": cols}
    assert best is not None

    final_spec = _model_specs(int(args.seed))[best["model_name"]]
    final_model = _fit(final_spec, train[best["feature_cols"]], y_all)
    val_proba_final = _proba3(final_model, train.iloc[val_idx][best["feature_cols"]])
    oos_proba = _proba3(final_model, oos[best["feature_cols"]])
    val_dec = base._decisions(train.iloc[val_idx].reset_index(drop=True), val_proba_final, float(best["threshold"]))
    oos_dec = base._decisions(oos, oos_proba, float(best["threshold"]))
    val_class = _metrics(y_all[val_idx], val_proba_final, float(best["threshold"]))
    oos_class = _metrics(y_oos, oos_proba, float(best["threshold"]))
    val_cost = base._cost_metrics(train.iloc[val_idx].reset_index(drop=True), val_dec, args)
    oos_cost = base._cost_metrics(oos, oos_dec, args)

    val_dec.to_csv(args.out_dir / "validation_decisions.csv", index=False)
    oos_dec.to_csv(args.out_dir / "oos_2026_decisions.csv", index=False)
    joblib.dump(
        {
            "model": final_model,
            "model_name": best["model_name"],
            "model_family": best["model_family"],
            "feature_cols": best["feature_cols"],
            "confidence_threshold": float(best["threshold"]),
            "feature_set": best["feature_set"],
            "model_id": MODEL_ID,
        },
        args.out_dir / "model.joblib",
    )
    summary = {
        "model_id": MODEL_ID,
        "layer_contract": "Layer1 + Layer2 inputs only; teacher_* and other Layer3 outputs forbidden",
        "label_source": "zigzag_action",
        "selection": "model/threshold selected on 2025 validation Cost3 Calmar; 2026 is final OOS only",
        "train_window": "2025",
        "validation_start": str(args.val_start),
        "oos_window": "2026",
        "best": {
            "model_name": best["model_name"],
            "model_family": best["model_family"],
            "feature_set": best["feature_set"],
            "feature_count": int(len(best["feature_cols"])),
            "confidence_threshold": float(best["threshold"]),
            "validation_selection": best["validation"],
            "validation_backtest_selection": best["validation_backtest"],
            "validation_final": val_class,
            "validation_backtest_final": val_cost,
            "oos_2026": oos_class,
            "oos_2026_backtest": oos_cost,
        },
        "all_runs": runs,
        "feature_sets": feature_sets,
        "row_drop_events": base.DROP_EVENTS,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "model": str(args.out_dir / "model.joblib"),
            "validation_decisions": str(args.out_dir / "validation_decisions.csv"),
            "oos_2026_decisions": str(args.out_dir / "oos_2026_decisions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    (args.out_dir / "selected_features.json").write_text(json.dumps(best["feature_cols"], indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
