#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import (  # noqa: E402
    BASELINE_MODEL,
    _balanced_weights,
    _compose_2stage,
    _compose_ovr,
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


MODEL_ID = "alpha5_24_catboost_gpu_direction_refined_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_entry_rebalanced_labels_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_catboost_gpu_direction_refined_20260519"


@dataclass(frozen=True)
class CatSpec:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float
    bagging_temperature: float


def _cat_specs() -> list[CatSpec]:
    return [
        CatSpec("base", 400, 6, 0.040, 3.0, 1.0, 0.0),
        CatSpec("regularized", 320, 5, 0.035, 6.0, 2.0, 0.5),
        CatSpec("deeper", 520, 8, 0.028, 3.0, 0.5, 0.0),
    ]


def _fit_cat(
    x: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    spec: CatSpec,
    seed: int,
    *,
    task_type: str,
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
        task_type=str(task_type),
        devices=str(devices),
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y, sample_weight=w)
    return model


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    if raw.ndim == 2 and raw.shape[1] >= 2:
        return raw[:, 1]
    return raw.reshape(-1)


def _stage2_specs() -> list[dict[str, CatSpec]]:
    specs = {spec.name: spec for spec in _cat_specs()}
    return [
        {"name": "cat2_v1", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "cat2_v2", "entry": specs["deeper"], "direction": specs["regularized"]},
        {"name": "cat2_v3", "entry": specs["regularized"], "direction": specs["regularized"]},
    ]


def _ovr_specs() -> list[dict[str, CatSpec]]:
    specs = {spec.name: spec for spec in _cat_specs()}
    return [
        {"name": "catovr_v1", "entry": specs["regularized"], "long": specs["deeper"], "short": specs["deeper"]},
        {"name": "catovr_v2", "entry": specs["deeper"], "long": specs["regularized"], "short": specs["regularized"]},
        {"name": "catovr_v3", "entry": specs["regularized"], "long": specs["regularized"], "short": specs["deeper"]},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Train CatBoost GPU retries on alpha5_24 direction-refined labels.")
    p.add_argument(
        "--allow-deprecated-action-model",
        action="store_true",
        help="Allow historical reproduction of the deprecated CatBoost Major/Direction direct-action path.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-file", default="alpha5_24_entry_rebalanced_train.parquet")
    p.add_argument("--val-file", default="alpha5_24_entry_rebalanced_val.parquet")
    p.add_argument("--oos-file", default="alpha5_24_entry_rebalanced_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--task-type", default="GPU")
    p.add_argument("--devices", default="0")
    p.add_argument("--entry-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--score-thresholds", default="0.25,0.30,0.35,0.40")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=52501)
    args = p.parse_args()
    if not bool(args.allow_deprecated_action_model):
        p.error(
            "CatBoost Major/Direction is deprecated and not allowed in active live/backtest paths. "
            "Use Router5 a5dir_* only as auxiliary DSAC features, or pass "
            "--allow-deprecated-action-model for historical reproduction."
        )

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

    x_train_entry = _x(train_entry, entry_cols)
    x_val_entry = _x(val_df, entry_cols)
    x_oos_entry = _x(oos_df, entry_cols)
    x_train_dir = _x(train_dir, direction_cols)
    x_val_dir = _x(val_df, direction_cols)
    x_oos_dir = _x(oos_df, direction_cols)

    y_entry = pd.to_numeric(train_entry["entry_label"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_dir = (pd.to_numeric(train_dir["direction_label"], errors="coerce").fillna(0).to_numpy(np.int64) == 1).astype(np.int64)
    y_long = y_dir.copy()
    y_short = 1 - y_dir
    y_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)
    y_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0).to_numpy(np.int64)

    w_entry = np.clip(pd.to_numeric(train_entry["entry_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_dir = np.clip(pd.to_numeric(train_dir["direction_sample_weight"], errors="coerce").fillna(0.0).to_numpy(np.float64), 1e-4, None)
    w_entry *= _balanced_weights(y_entry)
    w_dir *= _balanced_weights(y_dir)

    rows: list[dict[str, Any]] = []
    best_by_family: dict[str, dict[str, Any]] = {}

    print(json.dumps({
        "stage": "start",
        "model_id": MODEL_ID,
        "task_type": args.task_type,
        "devices": args.devices,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "path": str(BASELINE_MODEL)},
        "rows": {"train_all": int(len(train_df)), "train_entry": int(len(train_entry)), "train_direction": int(len(train_dir)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
        "ratios": {"entry_positive": float(np.mean(y_entry)), "direction_long_positive": float(np.mean(y_dir))},
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for i, spec in enumerate(_stage2_specs(), start=1):
        print(json.dumps({"stage": "fit_2stage", "done": i, "total": len(_stage2_specs()), "name": spec["name"], "entry_cat": spec["entry"].name, "direction_cat": spec["direction"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_cat(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + i * 100 + 1), task_type=str(args.task_type), devices=str(args.devices))
        direction_model = _fit_cat(x_train_dir, y_dir, w_dir, spec["direction"], int(args.seed + i * 100 + 2), task_type=str(args.task_type), devices=str(args.devices))
        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(direction_model, x_val_dir)

        best_val: dict[str, Any] | None = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    val_actions, _ = _compose_2stage(p_entry_val, p_long_val, entry_threshold=entry_threshold, side_threshold=side_threshold, margin_threshold=margin_threshold)
                    metrics = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                    candidate = {
                        "family": "two_stage",
                        "architecture": spec["name"],
                        "entry_cat": spec["entry"].name,
                        "direction_cat": spec["direction"].name,
                        "entry_threshold": float(entry_threshold),
                        "side_threshold": float(side_threshold),
                        "margin_threshold": float(margin_threshold),
                        "validation": metrics,
                    }
                    if best_val is None or float(candidate["validation"]["score"]) > float(best_val["validation"]["score"]):
                        best_val = candidate
        assert best_val is not None
        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = _binary_proba(direction_model, x_oos_dir)
        oos_actions, _ = _compose_2stage(p_entry_oos, p_long_oos, entry_threshold=float(best_val["entry_threshold"]), side_threshold=float(best_val["side_threshold"]), margin_threshold=float(best_val["margin_threshold"]))
        best_val["oos"] = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
        best_val["artifact_paths"] = {
            "entry_model": str(args.out_dir / f"{spec['name']}_entry_model.joblib"),
            "direction_model": str(args.out_dir / f"{spec['name']}_direction_model.joblib"),
        }
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "two_stage", "head": "entry", "backend": "catboost"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": direction_model, "feature_cols": direction_cols, "family": "two_stage", "head": "direction", "backend": "catboost"}, best_val["artifact_paths"]["direction_model"])
        rows.append(best_val)
        if ("two_stage" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["two_stage"]["validation"]["score"])):
            best_by_family["two_stage"] = best_val

    for i, spec in enumerate(_ovr_specs(), start=1):
        print(json.dumps({"stage": "fit_ovr", "done": i, "total": len(_ovr_specs()), "name": spec["name"], "entry_cat": spec["entry"].name, "long_cat": spec["long"].name, "short_cat": spec["short"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_cat(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + 1000 + i * 100 + 1), task_type=str(args.task_type), devices=str(args.devices))
        long_model = _fit_cat(x_train_dir, y_long, w_dir, spec["long"], int(args.seed + 1000 + i * 100 + 2), task_type=str(args.task_type), devices=str(args.devices))
        short_model = _fit_cat(x_train_dir, y_short, w_dir, spec["short"], int(args.seed + 1000 + i * 100 + 3), task_type=str(args.task_type), devices=str(args.devices))
        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(long_model, x_val_dir)
        p_short_val = _binary_proba(short_model, x_val_dir)

        best_val = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    for score_threshold in _grid(args.score_thresholds):
                        val_actions, _ = _compose_ovr(
                            p_entry_val,
                            p_long_val,
                            p_short_val,
                            trade_threshold=entry_threshold,
                            side_threshold=side_threshold,
                            margin_threshold=margin_threshold,
                            score_threshold=score_threshold,
                        )
                        metrics = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                        candidate = {
                            "family": "ovr",
                            "architecture": spec["name"],
                            "entry_cat": spec["entry"].name,
                            "long_cat": spec["long"].name,
                            "short_cat": spec["short"].name,
                            "entry_threshold": float(entry_threshold),
                            "side_threshold": float(side_threshold),
                            "margin_threshold": float(margin_threshold),
                            "score_threshold": float(score_threshold),
                            "validation": metrics,
                        }
                        if best_val is None or float(candidate["validation"]["score"]) > float(best_val["validation"]["score"]):
                            best_val = candidate
        assert best_val is not None
        p_entry_oos = _binary_proba(entry_model, x_oos_entry)
        p_long_oos = _binary_proba(long_model, x_oos_dir)
        p_short_oos = _binary_proba(short_model, x_oos_dir)
        oos_actions, _ = _compose_ovr(
            p_entry_oos,
            p_long_oos,
            p_short_oos,
            trade_threshold=float(best_val["entry_threshold"]),
            side_threshold=float(best_val["side_threshold"]),
            margin_threshold=float(best_val["margin_threshold"]),
            score_threshold=float(best_val["score_threshold"]),
        )
        best_val["oos"] = _eval(oos_df, oos_actions, y_oos, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
        best_val["artifact_paths"] = {
            "entry_model": str(args.out_dir / f"{spec['name']}_entry_model.joblib"),
            "long_model": str(args.out_dir / f"{spec['name']}_long_model.joblib"),
            "short_model": str(args.out_dir / f"{spec['name']}_short_model.joblib"),
        }
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "ovr", "head": "entry", "backend": "catboost"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": long_model, "feature_cols": direction_cols, "family": "ovr", "head": "long", "backend": "catboost"}, best_val["artifact_paths"]["long_model"])
        joblib.dump({"model": short_model, "feature_cols": direction_cols, "family": "ovr", "head": "short", "backend": "catboost"}, best_val["artifact_paths"]["short_model"])
        rows.append(best_val)
        if ("ovr" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["ovr"]["validation"]["score"])):
            best_by_family["ovr"] = best_val

    summary = {
        "model_id": MODEL_ID,
        "backend": "catboost",
        "task_type": args.task_type,
        "devices": args.devices,
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "oos_cost1_pnl": 1.22, "oos_cost1_mdd": -6.58, "oos_cost1_trades": 35, "path": str(BASELINE_MODEL)},
        "audit": audit,
        "feature_counts": {"entry": len(entry_cols), "direction": len(direction_cols)},
        "best_two_stage": best_by_family.get("two_stage"),
        "best_ovr": best_by_family.get("ovr"),
        "all_results": rows,
    }

    grid_csv = args.out_dir / "alpha5_24_catboost_gpu_direction_refined_grid.csv"
    pd.DataFrame([
        {
            "family": row["family"],
            "architecture": row["architecture"],
            "entry_cat": row.get("entry_cat"),
            "direction_cat": row.get("direction_cat"),
            "long_cat": row.get("long_cat"),
            "short_cat": row.get("short_cat"),
            "entry_threshold": row["entry_threshold"],
            "side_threshold": row["side_threshold"],
            "margin_threshold": row["margin_threshold"],
            "score_threshold": row.get("score_threshold"),
            "val_score": row["validation"]["score"],
            "val_cost1_pnl": row["validation"]["backtest"]["cost1"]["pnl"],
            "val_cost1_mdd": row["validation"]["backtest"]["cost1"]["mdd"],
            "val_cost1_trades": row["validation"]["backtest"]["cost1"]["trades"],
            "oos_score": row["oos"]["score"],
            "oos_cost1_pnl": row["oos"]["backtest"]["cost1"]["pnl"],
            "oos_cost1_mdd": row["oos"]["backtest"]["cost1"]["mdd"],
            "oos_cost1_trades": row["oos"]["backtest"]["cost1"]["trades"],
            "oos_trade_precision": row["oos"]["direction"]["trade_precision"],
            "oos_balanced_precision": row["oos"]["direction"]["balanced_trade_precision"],
        }
        for row in rows
    ]).sort_values(["family", "val_score"], ascending=[True, False]).to_csv(grid_csv, index=False)

    summary_path = args.out_dir / "alpha5_24_catboost_gpu_direction_refined_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "stage": "complete",
        "summary": str(summary_path),
        "grid": str(grid_csv),
        "best_two_stage": {
            "architecture": summary["best_two_stage"]["architecture"] if summary.get("best_two_stage") else None,
            "validation_cost1_pnl": summary["best_two_stage"]["validation"]["backtest"]["cost1"]["pnl"] if summary.get("best_two_stage") else None,
            "oos_cost1_pnl": summary["best_two_stage"]["oos"]["backtest"]["cost1"]["pnl"] if summary.get("best_two_stage") else None,
            "oos_trades": summary["best_two_stage"]["oos"]["backtest"]["cost1"]["trades"] if summary.get("best_two_stage") else None,
        },
        "best_ovr": {
            "architecture": summary["best_ovr"]["architecture"] if summary.get("best_ovr") else None,
            "validation_cost1_pnl": summary["best_ovr"]["validation"]["backtest"]["cost1"]["pnl"] if summary.get("best_ovr") else None,
            "oos_cost1_pnl": summary["best_ovr"]["oos"]["backtest"]["cost1"]["pnl"] if summary.get("best_ovr") else None,
            "oos_trades": summary["best_ovr"]["oos"]["backtest"]["cost1"]["trades"] if summary.get("best_ovr") else None,
        },
    }, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
