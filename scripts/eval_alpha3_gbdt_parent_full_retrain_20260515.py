#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
    train_policy,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha3_gbdt_parent_full_retrain_20260515"
BASE_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_gbdt_parent_full_retrain_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_gbdt_parent_full_retrain_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_gbdt_parent_full_retrain_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_gbdt_parent_full_retrain_20260515_grid.csv"
FAST_MODE = False
DROP_RETRAIN_FEATURES = {
    "patchtst_pred",
    "patchtst_confidence",
    "pred_patchtst",
    "conf_patchtst",
    "ai_anchor_revert_prob",
    "ai_anchor_overheat",
    "ai_anchor_trend_escape_prob",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
}
CRITICAL_NONCONSTANT_FEATURES = ("garch_vol_z",)


def _drop_retrain_features(cols: list[str]) -> list[str]:
    return [c for c in cols if c not in DROP_RETRAIN_FEATURES]


def _series_noninformative(frame: pd.DataFrame, col: str) -> bool:
    if col not in frame.columns:
        return True
    s = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not s.notna().any():
        return True
    return int(s.nunique(dropna=True)) <= 1 or float((s.fillna(0.0) == 0.0).mean()) >= 0.999


def _feature_preflight(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    parent_cols: list[str],
    contract_cols: list[str],
    base_audit: dict[str, Any],
) -> dict[str, Any]:
    blocking = list(base_audit.get("blocking", []))
    warnings = list(base_audit.get("warnings", []))
    selected = list(dict.fromkeys(parent_cols + contract_cols))
    blocked_selected = [c for c in selected if c in DROP_RETRAIN_FEATURES]
    if blocked_selected:
        blocking.append("blocked_retrain_features_selected:" + ",".join(blocked_selected[:40]))
    for frame_name, frame in (("train", train), ("eval", eval_df)):
        missing = [c for c in selected if c not in frame.columns and c != "side_hint" and not c.startswith(("mom_", "abs_mom_"))]
        if missing:
            blocking.append(f"{frame_name}_selected_features_missing:" + ",".join(missing[:40]))
        for col in CRITICAL_NONCONSTANT_FEATURES:
            if col in selected and _series_noninformative(frame, col):
                blocking.append(f"{frame_name}_critical_feature_noninformative:{col}")
    raw_train = set(train.columns)
    raw_eval = set(eval_df.columns)
    if raw_train != raw_eval:
        warnings.append(f"raw_column_mismatch train_only={len(raw_train - raw_eval)} eval_only={len(raw_eval - raw_train)}")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "blocked_feature_denylist": sorted(DROP_RETRAIN_FEATURES),
        "critical_nonconstant_features": list(CRITICAL_NONCONSTANT_FEATURES),
    }


def _n(full: int, fast: int) -> int:
    return int(fast if FAST_MODE else full)


class FillNAWrapper:
    def __init__(self, model: Any, fill_value: float | None = None) -> None:
        self.model = model
        self.fill_value = fill_value

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.model.classes_)

    def _x(self, x: pd.DataFrame) -> pd.DataFrame:
        xx = x.replace([np.inf, -np.inf], np.nan)
        if self.fill_value is not None:
            xx = xx.fillna(float(self.fill_value))
        return xx

    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "FillNAWrapper":
        try:
            self.model.fit(self._x(x), y, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(self._x(x), y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._x(x))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(x))


class EncodedClassifierWrapper(FillNAWrapper):
    def fit(self, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "EncodedClassifierWrapper":
        self._classes = np.asarray(sorted(np.unique(y)), dtype=int)
        remap = {int(c): i for i, c in enumerate(self._classes)}
        yy = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
        try:
            self.model.fit(self._x(x), yy, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(self._x(x), yy)
        return self

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self._classes, dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._x(x))


class SoftVotingClassifierWrapper:
    def __init__(self, models: list[Any]) -> None:
        self.models = models
        all_classes = sorted({int(c) for m in models for c in np.asarray(m.classes_, dtype=int)})
        self.classes_ = np.asarray(all_classes, dtype=int)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        for m in self.models:
            p = np.asarray(m.predict_proba(x), dtype=np.float64)
            cls = np.asarray(m.classes_, dtype=int)
            for j, c in enumerate(cls):
                out[:, int(np.flatnonzero(self.classes_ == c)[0])] += p[:, j]
        out /= max(len(self.models), 1)
        return out


class MeanRegressorWrapper:
    def __init__(self, models: list[Any]) -> None:
        self.models = models

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        preds = [np.asarray(m.predict(x), dtype=np.float64) for m in self.models]
        return np.mean(np.vstack(preds), axis=0)


def _fit_classifier(kind: str, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, seed: int) -> Any | None:
    n_classes = int(np.unique(y).size)
    if n_classes < 2:
        return None
    if kind == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=_n(300, 80),
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=8.0,
            random_strength=0.8,
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            od_type="Iter",
            od_wait=60,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
            task_type="GPU" if False else "CPU",
        )
        return EncodedClassifierWrapper(model, None).fit(x, y, w)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=_n(160, 80),
            learning_rate=0.025,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=35,
            reg_alpha=0.15,
            reg_lambda=1.2,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.82,
            path_smooth=4.0,
            extra_trees=True,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return EncodedClassifierWrapper(model, None).fit(x, y, w)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            n_estimators=_n(250, 80),
            learning_rate=0.030,
            max_depth=5,
            min_child_weight=8.0,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.15,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
        return EncodedClassifierWrapper(model, None).fit(x, y, w)
    raise ValueError(kind)


def _fit_regressor(kind: str, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, seed: int) -> Any:
    if kind == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=_n(300, 80),
            learning_rate=0.035,
            depth=7,
            l2_leaf_reg=8.0,
            random_strength=0.8,
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            od_type="Iter",
            od_wait=60,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
        )
        return FillNAWrapper(model, None).fit(x, y, w)
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            objective="regression",
            n_estimators=_n(160, 80),
            learning_rate=0.025,
            num_leaves=47,
            max_depth=-1,
            min_child_samples=35,
            reg_alpha=0.15,
            reg_lambda=1.2,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.82,
            path_smooth=4.0,
            extra_trees=True,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        return FillNAWrapper(model, None).fit(x, y, w)
    if kind == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=_n(250, 80),
            learning_rate=0.030,
            max_depth=5,
            min_child_weight=8.0,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.15,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        return FillNAWrapper(model, None).fit(x, y, w)
    raise ValueError(kind)


def _fit_parent(kind: str, x: pd.DataFrame, y: Mapping[str, np.ndarray], cfg: FullyLearnedGovernorConfig, feature_cols: list[str], seed: int) -> dict[str, Any]:
    if kind == "hgb":
        return train_policy(x, y, cfg=cfg, random_state=seed, feature_cols=feature_cols)
    action_weights = np.where(np.asarray(y["action"]) == ACTION_CASH, 0.35, 1.0)
    quality_weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64)), 0.03, 1.0)
    weights = np.maximum(action_weights, quality_weights)
    trade_mask = np.asarray(y["action"]) != ACTION_CASH
    x_trade = x.loc[trade_mask].copy()
    trade_weights = weights[trade_mask]
    bundle: dict[str, Any] = {
        "model_type": f"fully_learned_governor_policy_{kind}_20260515",
        "feature_cols": list(feature_cols),
        "config": asdict(cfg),
        "action_model": _fit_classifier(kind, x, np.asarray(y["action"]), weights, seed),
        "quality_model": _fit_regressor(kind, x, np.asarray(y["quality"], dtype=np.float64), weights, seed + 99),
        "default_bucket_indexes": {
            key: int(pd.Series(np.asarray(y[key])[trade_mask]).mode().iloc[0]) if np.any(trade_mask) else 0
            for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
        },
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict()
            for key, vals in y.items()
            if key != "quality"
        },
    }
    for offset, key in enumerate(("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"), start=1):
        model = _fit_classifier(kind, x_trade, np.asarray(y[key])[trade_mask], trade_weights, seed + offset)
        if model is not None:
            bundle[f"{key}_model"] = model
    bundle["label_distribution"]["quality_mean"] = float(np.mean(y["quality"]))
    bundle["label_distribution"]["quality_p95"] = float(np.quantile(y["quality"], 0.95))
    return bundle


def _ensemble_parent(cat: dict[str, Any], lgbm: dict[str, Any]) -> dict[str, Any]:
    keys = ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
    out = {
        "model_type": "fully_learned_governor_policy_catboost_lightgbm_soft_ensemble_20260515",
        "feature_cols": list(cat["feature_cols"]),
        "config": dict(cat["config"]),
        "action_model": SoftVotingClassifierWrapper([cat["action_model"], lgbm["action_model"]]),
        "quality_model": MeanRegressorWrapper([cat["quality_model"], lgbm["quality_model"]]),
        "default_bucket_indexes": dict(cat["default_bucket_indexes"]),
        "label_distribution": dict(cat["label_distribution"]),
    }
    for key in keys[1:]:
        mk = f"{key}_model"
        if mk in cat and mk in lgbm:
            out[mk] = SoftVotingClassifierWrapper([cat[mk], lgbm[mk]])
    return out


def _with_runtime_overlay(bundle: dict[str, Any], runtime_parent: dict[str, Any]) -> dict[str, Any]:
    """Train labels with the base h288 recipe, then run with Alpha3 margin110 buckets."""
    out = copy.deepcopy(bundle)
    out["training_config"] = dict(out.get("config", {}))
    out["config"] = dict(runtime_parent["config"])
    out["runtime_overlay_source"] = str(v31.DEFAULT_PARENT)
    out["runtime_overlay_note"] = "Models are trained on the original base h288 label recipe, then evaluated with Alpha3 margin110 notional/margin config."
    return out


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _metrics(df: pd.DataFrame, parent_for_features: dict[str, Any], runner: dict[str, Any], cfg: CostRunnerConfig, q: np.ndarray, decisions: pd.DataFrame, overlay: Any, limit_cfg: Any, *, fee: float, slip: float) -> dict[str, Any]:
    return alpha3_close._metrics_signal_limit_close(df, parent_for_features, runner, cfg, q, decisions, overlay, limit_cfg, fee=fee, slip=slip)


def _train_downstream(
    *,
    name: str,
    parent_for_features: dict[str, Any],
    parent_bundle: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    contract_cols: list[str],
    v27_model: Any,
    v27_payload: dict[str, Any],
    overlay: Any,
    limit_cfg: Any,
    existing_runner: dict[str, Any],
    existing_add_cfg: CostRunnerConfig,
    fee: float,
    slip: float,
    teacher_epochs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    print(f"[{MODEL_ID}] {name}: predicting parent decisions", flush=True)
    train_dec = predict_policy_frame(parent_bundle, train_df, close=_close(train_df))
    val_dec = predict_policy_frame(parent_bundle, val_df, close=_close(val_df))
    eval_dec = predict_policy_frame(parent_bundle, eval_df, close=_close(eval_df))

    buckets = tuple(float(x) for x in FullyLearnedGovernorConfig(**dict(parent_for_features["config"])).notional_buckets)
    print(f"[{MODEL_ID}] {name}: retraining teacher gate", flush=True)
    train_features = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=contract_cols)
    train_seq = teacher._seq_tensor(train_features, np.arange(len(train_df), dtype=np.int64), contract_cols)
    y_action = train_dec["action"].astype(int).to_numpy(dtype=np.int64)
    y_quality = pd.to_numeric(train_dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_notional = teacher._bucket_labels(train_dec, buckets)
    teacher_model, teacher_meta = teacher._train_teacher_model(train_seq, y_action, y_quality, y_notional, n_buckets=len(buckets), epochs=teacher_epochs)

    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=contract_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=contract_cols)
    train_pred = teacher._predict_deep(teacher_model, train_features, contract_cols, teacher_meta["norm"])
    val_pred = teacher._predict_deep(teacher_model, val_features, contract_cols, teacher_meta["norm"])
    eval_pred = teacher._predict_deep(teacher_model, eval_features, contract_cols, teacher_meta["norm"])
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] {name}: selecting teacher runtime", flush=True)
    rows: list[dict[str, Any]] = []
    selected_rt: alpha2.Alpha2Runtime | None = None
    best_score = -1e18
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    for rt in alpha2._runtimes():
        dec = alpha2._decisions(val_dec, val_pred, buckets, rt)
        metrics = _metrics(val_df, parent_for_features, existing_runner, noop_cfg, val_q, dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"candidate": name, "stage": "teacher_runtime", **asdict(rt), "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"]})
        if score > best_score:
            best_score = score
            selected_rt = rt
            print(f"[{MODEL_ID}] {name}: new teacher {rt.name} score={score:.2f} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}", flush=True)
    assert selected_rt is not None

    train_final_dec = alpha2._decisions(train_dec, train_pred, buckets, selected_rt)
    val_final_dec = alpha2._decisions(val_dec, val_pred, buckets, selected_rt)
    eval_final_dec = alpha2._decisions(eval_dec, eval_pred, buckets, selected_rt)

    print(f"[{MODEL_ID}] {name}: retraining V21.2 runner", flush=True)
    runner = _fit_cost_runner_with_decisions(train_df, parent_for_features, train_final_dec, fee=fee, slip=slip)

    print(f"[{MODEL_ID}] {name}: selecting runner config", flush=True)
    selected_cfg: CostRunnerConfig | None = None
    selected_val_metrics: dict[str, Any] | None = None
    best_runner_score = -1e18
    for add_cfg in _runner_grid():
        metrics = _metrics(val_df, parent_for_features, runner, add_cfg, val_q, val_final_dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"candidate": name, "stage": "runner_config", **asdict(selected_rt), "runner_config": add_cfg.name, "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost1_trades": metrics["cost1"]["trades"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"]})
        if score > best_runner_score:
            best_runner_score = score
            selected_cfg = add_cfg
            selected_val_metrics = metrics
            print(f"[{MODEL_ID}] {name}: new runner {add_cfg.name} score={score:.2f} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}", flush=True)
    assert selected_cfg is not None
    assert selected_val_metrics is not None

    metrics = _metrics(eval_df, parent_for_features, runner, selected_cfg, eval_q, eval_final_dec, overlay, limit_cfg, fee=fee, slip=slip)
    result = {
        "name": name,
        "metrics": metrics,
        "score": _score(selected_val_metrics),
        "validation_metrics": selected_val_metrics,
        "oos_score": _score(metrics),
        "selected_teacher_runtime": asdict(selected_rt),
        "selected_runner_config": asdict(selected_cfg),
        "runner_meta": {k: v for k, v in runner.items() if k not in {"regressor", "q10_regressor", "q90_regressor", "classifier", "jackpot_classifier", "bad_classifier", "cost3_classifier", "feature_cols"}},
    }
    artifact_dir = OUT_DIR / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(parent_bundle, artifact_dir / "parent.pkl")
    joblib.dump({"model_id": MODEL_ID, "candidate": name, "cost_runner": runner, "selected_config": asdict(selected_cfg), "teacher_runtime": asdict(selected_rt)}, artifact_dir / "runner.pkl")
    import torch

    torch.save({"model_id": MODEL_ID, "candidate": name, "state_dict": teacher_model.state_dict(), "feature_cols": contract_cols, "train_meta": teacher_meta, "buckets": buckets}, artifact_dir / "teacher_gate.pt")
    result["artifacts"] = {"parent": str(artifact_dir / "parent.pkl"), "runner": str(artifact_dir / "runner.pkl"), "teacher_gate": str(artifact_dir / "teacher_gate.pt")}
    print(f"[{MODEL_ID}] {name}: OOS c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}", flush=True)
    return result, rows, {"teacher_runtime": asdict(selected_rt), "runner_config": asdict(selected_cfg)}


def main() -> int:
    global FAST_MODE, OUT_DIR, REPORT_OUT, AUDIT_OUT, GRID_OUT
    p = argparse.ArgumentParser(description="Train CatBoost/LightGBM/XGBoost/ensemble parents and retrain Alpha3 downstream for each.")
    p.add_argument("--train-csv", type=Path, default=v31.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v31.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--report-out", type=Path, default=REPORT_OUT)
    p.add_argument("--audit-out", type=Path, default=AUDIT_OUT)
    p.add_argument("--grid-out", type=Path, default=GRID_OUT)
    p.add_argument("--teacher-epochs", type=int, default=35)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--fast", action="store_true", help="Use small tree budgets for path validation.")
    p.add_argument("--only", nargs="*", default=None, choices=["hgb", "catboost", "lightgbm", "xgboost", "cat_lgbm_ensemble"])
    args = p.parse_args()
    FAST_MODE = bool(args.fast)
    OUT_DIR = args.out_dir
    REPORT_OUT = args.report_out
    AUDIT_OUT = args.audit_out
    GRID_OUT = args.grid_out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    runtime_cfg = FullyLearnedGovernorConfig(**dict(parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    feature_cols = _drop_retrain_features(list(parent_ref["feature_cols"]))
    parent_feature_ref = copy.deepcopy(parent_ref)
    parent_feature_ref["feature_cols"] = list(feature_cols)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    contract_cols = _drop_retrain_features(_feature_cols(train_all, eval_df))
    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    preflight = _feature_preflight(train_all, eval_df, parent_cols=feature_cols, contract_cols=contract_cols, base_audit=audit_base)
    if preflight["blocking"]:
        audit = {
            "status": "fail",
            "verdict": "blocked",
            "blocking": preflight["blocking"],
            "warnings": preflight["warnings"],
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS only after parent/downstream selection",
            "base_feature_audit": audit_base,
            "preflight_feature_audit": preflight,
        }
        report = {"model_id": MODEL_ID, "experiments": [], "audit": audit, "artifacts": {"out_dir": str(OUT_DIR), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)}}
        REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        print(json.dumps({"status": "blocked", "audit": str(AUDIT_OUT), "blocking": preflight["blocking"][:5]}, ensure_ascii=False), flush=True)
        return 2
    print(f"[{MODEL_ID}] building labels stride={args.stride} with base h288 config, runtime uses margin110 overlay", flush=True)
    x_train, y_train, train_meta = build_training_set(train_df, cfg=label_cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)

    existing_runner_payload = joblib.load(v31.DEFAULT_JACKPOT)
    existing_runner = existing_runner_payload["cost_runner"]
    existing_add_cfg = CostRunnerConfig(**dict(existing_runner_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    existing_teacher_model, existing_teacher_cols, existing_teacher_norm, existing_teacher_buckets = ft_v2.ft_v1._load_teacher()
    existing_alpha3_runtime = ft_v2.ft_v1._selected_alpha3_runtime()
    val_hgb_dec = predict_policy_frame(parent_ref, val_df, close=_close(val_df))
    eval_hgb_dec = predict_policy_frame(parent_ref, eval_df, close=_close(eval_df))
    existing_teacher_val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=existing_teacher_cols)
    existing_teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=existing_teacher_cols)
    existing_teacher_val_pred = teacher._predict_deep(existing_teacher_model, existing_teacher_val_features, existing_teacher_cols, existing_teacher_norm)
    existing_teacher_pred = teacher._predict_deep(existing_teacher_model, existing_teacher_features, existing_teacher_cols, existing_teacher_norm)
    alpha3_current_val_dec = alpha2._decisions(val_hgb_dec, existing_teacher_val_pred, existing_teacher_buckets, existing_alpha3_runtime)
    alpha3_current_dec = alpha2._decisions(eval_hgb_dec, existing_teacher_pred, existing_teacher_buckets, existing_alpha3_runtime)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    baseline_val_metrics = _metrics(val_df, parent_ref, existing_runner, existing_add_cfg, val_q, alpha3_current_val_dec, overlay, limit_cfg, fee=fee, slip=slip)
    baseline_metrics = _metrics(eval_df, parent_ref, existing_runner, existing_add_cfg, eval_q, alpha3_current_dec, overlay, limit_cfg, fee=fee, slip=slip)
    baseline = {"name": "alpha3_current_hgb_parent_teacher_downstream", "metrics": baseline_metrics, "score": _score(baseline_val_metrics), "validation_metrics": baseline_val_metrics, "oos_score": _score(baseline_metrics)}
    print(f"[{MODEL_ID}] baseline val_c1={baseline_val_metrics['cost1']['pnl']:.2f} oos_c1={baseline_metrics['cost1']['pnl']:.2f} oos_mdd={baseline_metrics['cost1']['mdd']:.2f}", flush=True)

    candidates = list(args.only or ["catboost", "lightgbm", "xgboost", "cat_lgbm_ensemble"])
    parent_bundles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = [baseline]
    for kind in ("hgb", "catboost", "lightgbm", "xgboost"):
        if kind in candidates or (kind in ("catboost", "lightgbm") and "cat_lgbm_ensemble" in candidates):
            print(f"[{MODEL_ID}] fitting parent {kind}", flush=True)
            fitted = _fit_parent(kind, x_train, y_train, label_cfg, feature_cols, int(args.seed) + len(parent_bundles) * 100)
            parent_bundles[kind] = _with_runtime_overlay(fitted, parent_ref)
            joblib.dump(parent_bundles[kind], OUT_DIR / f"{kind}_parent.pkl")
    if "cat_lgbm_ensemble" in candidates:
        parent_bundles["cat_lgbm_ensemble"] = _ensemble_parent(parent_bundles["catboost"], parent_bundles["lightgbm"])
        joblib.dump(parent_bundles["cat_lgbm_ensemble"], OUT_DIR / "cat_lgbm_ensemble_parent.pkl")

    for name in candidates:
        result, result_rows, _selected = _train_downstream(
            name=name,
            parent_for_features=parent_feature_ref,
            parent_bundle=parent_bundles[name],
            train_df=train_df,
            val_df=val_df,
            eval_df=eval_df,
            contract_cols=contract_cols,
            v27_model=v27_model,
            v27_payload=v27_payload,
            overlay=overlay,
            limit_cfg=limit_cfg,
            existing_runner=existing_runner,
            existing_add_cfg=existing_add_cfg,
            fee=fee,
            slip=slip,
            teacher_epochs=int(args.teacher_epochs),
        )
        rows.extend(result_rows)
        experiments.append(result)
        pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
        REPORT_OUT.write_text(json.dumps({"model_id": MODEL_ID, "experiments": experiments}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    best = max(experiments[1:], key=lambda e: float(e["score"])) if len(experiments) > 1 else None
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best is None or float(best["score"]) <= float(baseline["score"]):
        warnings.append("no_gbdt_retrained_candidate_beat_alpha3_current")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if best and not blocking and float(best["score"]) > float(baseline["score"]) else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after parent/downstream selection",
        "train_meta": train_meta,
        "label_training_config": asdict(label_cfg),
        "runtime_config": asdict(runtime_cfg),
        "base_feature_audit": audit_base,
        "preflight_feature_audit": preflight,
        "best_candidate": None if best is None else {k: v for k, v in best.items() if k != "metrics"},
        "alpha3_execution_contract": asdict(limit_cfg),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "CatBoost, LightGBM, XGBoost, and CatBoost+LightGBM parent candidates. Parents are trained with the original base h288 label recipe, then evaluated with the Alpha3 margin110 runtime overlay. For every parent candidate, teacher gate and V21.2 runner are retrained, teacher runtime and runner config are selected on 2025Q4, and OOS is fixed 2026 under Alpha3 corrected next_open_limit_touch0_fee20 execution.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": None if best is None else best["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
