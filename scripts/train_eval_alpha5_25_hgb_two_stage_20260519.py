#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import (  # noqa: E402
    BASELINE_MODEL,
    _balanced_weights,
    _direction_feature_cols,
    _entry_feature_cols,
    _eval,
    _feature_cols,
    _grid,
    _x,
)
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_25_gpu_two_stage_compare_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_25_two_stage_labels_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_25_gpu_two_stage_compare_20260519"


@dataclass(frozen=True)
class CatSpec:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float
    bagging_temperature: float


@dataclass(frozen=True)
class LGBSpec:
    name: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    reg_alpha: float
    reg_lambda: float
    subsample: float
    colsample_bytree: float


def _cat_specs() -> list[dict[str, CatSpec]]:
    specs = {
        "regularized": CatSpec("regularized", 480, 5, 0.035, 6.0, 2.0, 0.5),
        "deeper": CatSpec("deeper", 680, 8, 0.026, 3.0, 0.5, 0.0),
    }
    return [
        {"name": "cat25_v1", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "cat25_v2", "entry": specs["deeper"], "direction": specs["regularized"]},
        {"name": "cat25_v3", "entry": specs["regularized"], "direction": specs["regularized"]},
    ]


def _lgb_specs() -> list[dict[str, LGBSpec]]:
    specs = {
        "regularized": LGBSpec("regularized", 420, 0.035, 23, 120, 0.08, 0.25, 0.85, 0.85),
        "deeper": LGBSpec("deeper", 620, 0.028, 63, 50, 0.00, 0.08, 0.95, 0.95),
    }
    return [
        {"name": "lgb25_v1", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "lgb25_v2", "entry": specs["deeper"], "direction": specs["regularized"]},
        {"name": "lgb25_v3", "entry": specs["regularized"], "direction": specs["regularized"]},
    ]


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return raw[:, classes.index(1)]
    if raw.ndim == 2 and raw.shape[1] >= 2:
        return raw[:, 1]
    return raw.reshape(-1)


def _whipsaw_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame["regime4_state"].astype(str).to_numpy() == "whipsaw"


def _compose_2stage(
    p_entry: np.ndarray,
    p_long: np.ndarray,
    *,
    regime: np.ndarray,
    entry_threshold: float,
    side_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p_entry = np.clip(p_entry, 0.0, 1.0)
    p_long = np.clip(p_long, 0.0, 1.0)
    p_short = 1.0 - p_long
    margin = np.abs(p_long - p_short)
    best_side = np.maximum(p_long, p_short)
    actions = np.where(p_long >= p_short, 1, 2).astype(np.int64)
    actions = np.where(p_entry < float(entry_threshold), 0, actions)
    actions = np.where(best_side < float(side_threshold), 0, actions)
    actions = np.where(margin < float(margin_threshold), 0, actions)
    actions = np.where(regime == "whipsaw", 0, actions)
    return actions, {"p_entry": p_entry, "p_long": p_long, "p_short": p_short, "margin": margin, "best_side": best_side}


def _fit_cat(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    spec: CatSpec,
    seed: int,
    *,
    devices: str,
) -> Any:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=int(spec.iterations),
        depth=int(spec.depth),
        learning_rate=float(spec.learning_rate),
        l2_leaf_reg=float(spec.l2_leaf_reg),
        random_strength=float(spec.random_strength),
        bagging_temperature=float(spec.bagging_temperature),
        task_type="GPU",
        devices=str(devices),
        random_seed=int(seed),
        verbose=100,
        use_best_model=True,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=(x_val, y_val),
        use_best_model=True,
        early_stopping_rounds=80,
        verbose=100,
    )
    return model


def _fit_lgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    w_val: np.ndarray,
    spec: LGBSpec,
    seed: int,
    *,
    gpu_device_id: int,
) -> Any:
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=int(spec.n_estimators),
        learning_rate=float(spec.learning_rate),
        num_leaves=int(spec.num_leaves),
        min_child_samples=int(spec.min_child_samples),
        reg_alpha=float(spec.reg_alpha),
        reg_lambda=float(spec.reg_lambda),
        subsample=float(spec.subsample),
        colsample_bytree=float(spec.colsample_bytree),
        random_state=int(seed),
        n_jobs=-1,
        max_bin=255,
        device_type="gpu",
        gpu_device_id=int(gpu_device_id),
        verbosity=1,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(x_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[lgb.early_stopping(80, verbose=True), lgb.log_evaluation(100)],
    )
    return model


def _run_backend(
    *,
    backend: str,
    specs: list[dict[str, Any]],
    train_entry: pd.DataFrame,
    train_dir: pd.DataFrame,
    val_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    x_train_entry: pd.DataFrame,
    x_val_entry: pd.DataFrame,
    x_oos_entry: pd.DataFrame,
    x_train_dir: pd.DataFrame,
    x_val_dir: pd.DataFrame,
    x_oos_dir: pd.DataFrame,
    y_entry: np.ndarray,
    y_dir: np.ndarray,
    y_val: np.ndarray,
    y_oos: np.ndarray,
    w_entry: np.ndarray,
    w_dir: np.ndarray,
    entry_cols: list[str],
    direction_cols: list[str],
    entry_thresholds: list[float],
    side_thresholds: list[float],
    margin_thresholds: list[float],
    val_trade_min: int,
    val_trade_max: int,
    val_trade_penalty: float,
    fee: float,
    slip: float,
    unit_exposure: float,
    max_hold_bars: int,
    seed: int,
    out_dir: Path,
    devices: str,
    gpu_device_id: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    val_entry_target = (_num(val_df, "entry_label", 0.0).astype(np.int64))
    val_dir_mask = _num(val_df, "direction_valid", 0.0).astype(np.int64) == 1
    val_dir_target = (_num(val_df.loc[val_dir_mask], "direction_label", 0.0).astype(np.int64) == 1).astype(np.int64)

    for i, spec in enumerate(specs, start=1):
        print(json.dumps({"stage": "fit_2stage", "backend": backend, "done": i, "total": len(specs), "name": spec["name"]}, ensure_ascii=False), flush=True)
        t0 = time.perf_counter()
        if backend == "catboost":
            entry_model = _fit_cat(x_train_entry, y_entry, w_entry, x_val_entry, val_entry_target, np.ones(len(val_entry_target), dtype=np.float64), spec["entry"], int(seed + i * 100 + 1), devices=devices)
            direction_model = _fit_cat(x_train_dir, y_dir, w_dir, x_val_dir.loc[val_dir_mask].reset_index(drop=True), val_dir_target, np.ones(len(val_dir_target), dtype=np.float64), spec["direction"], int(seed + i * 100 + 2), devices=devices)
        else:
            entry_model = _fit_lgbm(x_train_entry, y_entry, w_entry, x_val_entry, val_entry_target, np.ones(len(val_entry_target), dtype=np.float64), spec["entry"], int(seed + i * 100 + 1), gpu_device_id=gpu_device_id)
            direction_model = _fit_lgbm(x_train_dir, y_dir, w_dir, x_val_dir.loc[val_dir_mask].reset_index(drop=True), val_dir_target, np.ones(len(val_dir_target), dtype=np.float64), spec["direction"], int(seed + i * 100 + 2), gpu_device_id=gpu_device_id)
        train_seconds = float(time.perf_counter() - t0)

        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(direction_model, x_val_dir)
        regime_val = val_df["regime4_state"].astype(str).to_numpy()

        best_val: dict[str, Any] | None = None
        for entry_threshold in entry_thresholds:
            for side_threshold in side_thresholds:
                for margin_threshold in margin_thresholds:
                    val_actions, _ = _compose_2stage(
                        p_entry_val,
                        p_long_val,
                        regime=regime_val,
                        entry_threshold=entry_threshold,
                        side_threshold=side_threshold,
                        margin_threshold=margin_threshold,
                    )
                    metrics = _eval(val_df, val_actions, y_val, fee=fee, slip=slip, exposure=unit_exposure, max_hold=max_hold_bars)
                    val_trades = int(metrics["backtest"]["cost1"]["trades"])
                    trade_penalty = 0.0
                    if val_trades > int(val_trade_max):
                        trade_penalty = float(val_trade_penalty) * float(val_trades - int(val_trade_max))
                    elif val_trades < int(val_trade_min):
                        trade_penalty = float(val_trade_penalty) * 1.5 * float(int(val_trade_min) - val_trades)
                    selection_score = float(metrics["score"]) - trade_penalty
                    candidate = {
                        "backend": backend,
                        "family": "two_stage",
                        "architecture": spec["name"],
                        "entry_model_type": spec["entry"].name,
                        "direction_model_type": spec["direction"].name,
                        "entry_threshold": float(entry_threshold),
                        "side_threshold": float(side_threshold),
                        "margin_threshold": float(margin_threshold),
                        "validation": metrics,
                        "selection_score": float(selection_score),
                        "selection_trade_penalty": float(trade_penalty),
                        "train_seconds": train_seconds,
                    }
                    if best_val is None or float(candidate["selection_score"]) > float(best_val["selection_score"]):
                        best_val = candidate
        assert best_val is not None

        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = _binary_proba(direction_model, x_oos_dir)
        regime_oos = oos_df["regime4_state"].astype(str).to_numpy()
        oos_actions, _ = _compose_2stage(
            p_entry_oos,
            p_long_oos,
            regime=regime_oos,
            entry_threshold=float(best_val["entry_threshold"]),
            side_threshold=float(best_val["side_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
        )
        best_val["oos"] = _eval(oos_df, oos_actions, y_oos, fee=fee, slip=slip, exposure=unit_exposure, max_hold=max_hold_bars)
        entry_art = out_dir / f"{spec['name']}_entry_model.joblib"
        dir_art = out_dir / f"{spec['name']}_direction_model.joblib"
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "two_stage", "head": "entry", "backend": backend}, entry_art)
        joblib.dump({"model": direction_model, "feature_cols": direction_cols, "family": "two_stage", "head": "direction", "backend": backend}, dir_art)
        best_val["artifact_paths"] = {"entry_model": str(entry_art), "direction_model": str(dir_art)}
        rows.append(best_val)
        if best is None or float(best_val["validation"]["score"]) > float(best["validation"]["score"]):
            best = best_val

    assert best is not None
    return {"best": best, "all_results": rows}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default).to_numpy(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Train CatBoost GPU and LightGBM GPU two-stage models on alpha5_25 labels and compare against HGB baseline.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-file", default="alpha5_25_two_stage_labels_train.parquet")
    p.add_argument("--val-file", default="alpha5_25_two_stage_labels_val.parquet")
    p.add_argument("--oos-file", default="alpha5_25_two_stage_labels_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--devices", default="0")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--backends", default="catboost,lightgbm")
    p.add_argument("--entry-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--val-trade-min", type=int, default=20)
    p.add_argument("--val-trade-max", type=int, default=35)
    p.add_argument("--val-trade-penalty", type=float, default=0.35)
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=52525)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_2025 = _read(args.raw_2025_csv)
    raw_2026 = _read(args.raw_2026_csv)
    audit = _verify_state24_sticky090_inputs(raw_2025, raw_2026, args.manifest, args.clean4_report)

    train_df = pd.read_parquet(args.data_dir / str(args.train_file))
    val_df = pd.read_parquet(args.data_dir / str(args.val_file))
    oos_df = pd.read_parquet(args.data_dir / str(args.oos_file))

    cols = _feature_cols(raw_2025, raw_2026, set(train_df.columns))
    entry_cols = _entry_feature_cols(cols)
    direction_cols = _direction_feature_cols(cols)
    if not entry_cols or not direction_cols:
        raise ValueError("failed to select entry/direction regime4_core features")

    train_entry = train_df[train_df["entry_train_keep"] == 1].reset_index(drop=True)
    train_dir = train_df[train_df["direction_train_keep"] == 1].reset_index(drop=True)
    if int(np.sum(train_dir["regime4_state"].astype(str) == "whipsaw")) != 0:
        raise ValueError("whipsaw rows remain in direction training subset")

    x_train_entry = _x(train_entry, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_dir = _x(train_dir, direction_cols)
    x_val_dir = _x(val_df, direction_cols)
    x_oos_dir = _x(oos_df, direction_cols)

    y_entry = pd.to_numeric(train_entry["entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_dir = (pd.to_numeric(train_dir["direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    w_entry = np.clip(pd.to_numeric(train_entry["entry_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_dir = np.clip(pd.to_numeric(train_dir["direction_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_entry *= _balanced_weights(y_entry)
    w_dir *= _balanced_weights(y_dir)

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "path": str(BASELINE_MODEL)},
        "rows": {"train_all": int(len(train_df)), "train_entry": int(len(train_entry)), "train_direction": int(len(train_dir)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
        "ratios": {"entry_positive": float(np.mean(y_entry)), "direction_long_positive": float(np.mean(y_dir))},
        "gpu": {"catboost_devices": args.devices, "lightgbm_gpu_device_id": int(args.gpu_device_id)},
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    entry_thresholds = _grid(args.entry_thresholds)
    side_thresholds = _grid(args.side_thresholds)
    margin_thresholds = _grid(args.margin_thresholds)
    backends = {x.strip().lower() for x in str(args.backends).split(",") if x.strip()}

    cat_dir = args.out_dir / "catboost"
    lgb_dir = args.out_dir / "lightgbm"
    cat_dir.mkdir(parents=True, exist_ok=True)
    lgb_dir.mkdir(parents=True, exist_ok=True)

    backend_errors: dict[str, str] = {}
    cat_result: dict[str, Any] | None = None
    lgb_result: dict[str, Any] | None = None
    if "catboost" in backends:
        try:
            cat_result = _run_backend(
                backend="catboost",
                specs=_cat_specs(),
                train_entry=train_entry,
                train_dir=train_dir,
                val_df=val_df,
                oos_df=oos_df,
                x_train_entry=x_train_entry,
                x_val_entry=x_val_entry,
                x_oos_entry=x_oos_entry,
                x_train_dir=x_train_dir,
                x_val_dir=x_val_dir,
                x_oos_dir=x_oos_dir,
                y_entry=y_entry,
                y_dir=y_dir,
                y_val=y_val,
                y_oos=y_oos,
                w_entry=w_entry,
                w_dir=w_dir,
                entry_cols=entry_cols,
                direction_cols=direction_cols,
                entry_thresholds=entry_thresholds,
                side_thresholds=side_thresholds,
                margin_thresholds=margin_thresholds,
                val_trade_min=int(args.val_trade_min),
                val_trade_max=int(args.val_trade_max),
                val_trade_penalty=float(args.val_trade_penalty),
                fee=float(args.fee),
                slip=float(args.slip),
                unit_exposure=float(args.unit_exposure),
                max_hold_bars=int(args.max_hold_bars),
                seed=int(args.seed),
                out_dir=cat_dir,
                devices=str(args.devices),
                gpu_device_id=int(args.gpu_device_id),
            )
        except Exception as exc:
            backend_errors["catboost"] = str(exc)
            print(json.dumps({"stage": "backend_error", "backend": "catboost", "error": str(exc)}, ensure_ascii=False), flush=True)

    if "lightgbm" in backends:
        try:
            lgb_result = _run_backend(
                backend="lightgbm",
                specs=_lgb_specs(),
                train_entry=train_entry,
                train_dir=train_dir,
                val_df=val_df,
                oos_df=oos_df,
                x_train_entry=x_train_entry,
                x_val_entry=x_val_entry,
                x_oos_entry=x_oos_entry,
                x_train_dir=x_train_dir,
                x_val_dir=x_val_dir,
                x_oos_dir=x_oos_dir,
                y_entry=y_entry,
                y_dir=y_dir,
                y_val=y_val,
                y_oos=y_oos,
                w_entry=w_entry,
                w_dir=w_dir,
                entry_cols=entry_cols,
                direction_cols=direction_cols,
                entry_thresholds=entry_thresholds,
                side_thresholds=side_thresholds,
                margin_thresholds=margin_thresholds,
                val_trade_min=int(args.val_trade_min),
                val_trade_max=int(args.val_trade_max),
                val_trade_penalty=float(args.val_trade_penalty),
                fee=float(args.fee),
                slip=float(args.slip),
                unit_exposure=float(args.unit_exposure),
                max_hold_bars=int(args.max_hold_bars),
                seed=int(args.seed + 10000),
                out_dir=lgb_dir,
                devices=str(args.devices),
                gpu_device_id=int(args.gpu_device_id),
            )
        except Exception as exc:
            backend_errors["lightgbm"] = str(exc)
            print(json.dumps({"stage": "backend_error", "backend": "lightgbm", "error": str(exc)}, ensure_ascii=False), flush=True)

    summary = {
        "model_id": MODEL_ID,
        "baseline_fixed": {
            "track": "regime4_core",
            "single_hgb": "deeper",
            "oos_cost1_pnl": 1.22,
            "oos_cost1_mdd": -6.58,
            "oos_cost1_trades": 35,
            "path": str(BASELINE_MODEL),
        },
        "audit": audit,
        "feature_counts": {"entry": len(entry_cols), "direction": len(direction_cols)},
        "backend_errors": backend_errors,
        "catboost": cat_result,
        "lightgbm": lgb_result,
    }

    rows = []
    for backend_name, backend_result in (("catboost", cat_result), ("lightgbm", lgb_result)):
        if backend_result is None:
            continue
        for row in backend_result["all_results"]:
            rows.append({
                "backend": backend_name,
                "architecture": row["architecture"],
                "entry_model_type": row["entry_model_type"],
                "direction_model_type": row["direction_model_type"],
                "entry_threshold": row["entry_threshold"],
                "side_threshold": row["side_threshold"],
                "margin_threshold": row["margin_threshold"],
                "train_seconds": row["train_seconds"],
                "selection_score": row.get("selection_score"),
                "selection_trade_penalty": row.get("selection_trade_penalty"),
                "val_score": row["validation"]["score"],
                "val_cost1_pnl": row["validation"]["backtest"]["cost1"]["pnl"],
                "val_cost1_mdd": row["validation"]["backtest"]["cost1"]["mdd"],
                "val_cost1_trades": row["validation"]["backtest"]["cost1"]["trades"],
                "oos_score": row["oos"]["score"],
                "oos_cost1_pnl": row["oos"]["backtest"]["cost1"]["pnl"],
                "oos_cost1_mdd": row["oos"]["backtest"]["cost1"]["mdd"],
                "oos_cost1_trades": row["oos"]["backtest"]["cost1"]["trades"],
                "oos_trades_per_day": row["oos"]["backtest"]["cost1"]["trades_per_day"],
                "oos_trade_precision": row["oos"]["direction"]["trade_precision"],
                "oos_balanced_precision": row["oos"]["direction"]["balanced_trade_precision"],
                "oos_long_precision": row["oos"]["direction"]["long_precision"],
                "oos_short_precision": row["oos"]["direction"]["short_precision"],
            })

    grid_csv = args.out_dir / "alpha5_25_gpu_two_stage_grid.csv"
    pd.DataFrame(rows).sort_values(["backend", "val_score"], ascending=[True, False]).to_csv(grid_csv, index=False)

    summary_path = args.out_dir / "alpha5_25_gpu_two_stage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_csv),
        "best_catboost": None if cat_result is None else {
            "architecture": cat_result["best"]["architecture"],
            "validation_cost1_pnl": cat_result["best"]["validation"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_pnl": cat_result["best"]["oos"]["backtest"]["cost1"]["pnl"],
            "oos_trades": cat_result["best"]["oos"]["backtest"]["cost1"]["trades"],
            "train_seconds": cat_result["best"]["train_seconds"],
        },
        "best_lightgbm": None if lgb_result is None else {
            "architecture": lgb_result["best"]["architecture"],
            "validation_cost1_pnl": lgb_result["best"]["validation"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_pnl": lgb_result["best"]["oos"]["backtest"]["cost1"]["pnl"],
            "oos_trades": lgb_result["best"]["oos"]["backtest"]["cost1"]["trades"],
            "train_seconds": lgb_result["best"]["train_seconds"],
        },
        "backend_errors": backend_errors,
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
