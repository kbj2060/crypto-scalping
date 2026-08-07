#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import prepare_features  # noqa: E402
from scripts.train_eval_alpha5_3_hmm_dqn_router_parent_20260517 import (  # noqa: E402
    DEFAULT_CLEAN4_REPORT,
    DEFAULT_PREPROCESS_MANIFEST,
    _verify_state24_sticky090_inputs,
)
from scripts.train_eval_alpha5_8_hgb_action_feature_contract_compare_20260518 import _alpha4_mapped_features  # noqa: E402
from scripts.train_eval_alpha5_13_hgb_single_20260518 import _backtest_barrier, _direction_metrics  # noqa: E402
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
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "alpha5_24_lgbm_gpu_direction_refined_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_entry_rebalanced_labels_20260519"
DEFAULT_RAW_2025 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_RAW_2026 = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_24_lgbm_gpu_direction_refined_20260519"


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


def _lgb_specs() -> list[LGBSpec]:
    return [
        LGBSpec("base", 320, 0.040, 31, 80, 0.00, 0.10, 0.90, 0.90),
        LGBSpec("regularized", 260, 0.035, 23, 120, 0.08, 0.25, 0.85, 0.85),
        LGBSpec("deeper", 420, 0.030, 63, 50, 0.00, 0.08, 0.95, 0.95),
    ]


def _fit_lgbm(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, spec: LGBSpec, seed: int, *, device_type: str, gpu_device_id: int) -> Any:
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
        device_type=str(device_type),
        gpu_device_id=int(gpu_device_id),
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model


def _binary_proba(model: Any, x: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return raw[:, classes.index(1)]
    return raw[:, -1]


def _stage2_specs() -> list[dict[str, LGBSpec]]:
    specs = {spec.name: spec for spec in _lgb_specs()}
    return [
        {"name": "lgb2_v1", "entry": specs["regularized"], "direction": specs["deeper"]},
        {"name": "lgb2_v2", "entry": specs["deeper"], "direction": specs["regularized"]},
        {"name": "lgb2_v3", "entry": specs["regularized"], "direction": specs["regularized"]},
    ]


def _ovr_specs() -> list[dict[str, LGBSpec]]:
    specs = {spec.name: spec for spec in _lgb_specs()}
    return [
        {"name": "lgbovr_v1", "entry": specs["regularized"], "long": specs["deeper"], "short": specs["deeper"]},
        {"name": "lgbovr_v2", "entry": specs["deeper"], "long": specs["regularized"], "short": specs["regularized"]},
        {"name": "lgbovr_v3", "entry": specs["regularized"], "long": specs["regularized"], "short": specs["deeper"]},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM GPU retries on alpha5_24 direction-refined labels.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--train-file", default="alpha5_24_entry_rebalanced_train.parquet")
    p.add_argument("--val-file", default="alpha5_24_entry_rebalanced_val.parquet")
    p.add_argument("--oos-file", default="alpha5_24_entry_rebalanced_oos.parquet")
    p.add_argument("--raw-2025-csv", type=Path, default=DEFAULT_RAW_2025)
    p.add_argument("--raw-2026-csv", type=Path, default=DEFAULT_RAW_2026)
    p.add_argument("--manifest", type=Path, default=DEFAULT_PREPROCESS_MANIFEST)
    p.add_argument("--clean4-report", type=Path, default=DEFAULT_CLEAN4_REPORT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device-type", default="gpu")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--entry-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--side-thresholds", default="0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.08,0.12")
    p.add_argument("--score-thresholds", default="0.25,0.30,0.35,0.40")
    p.add_argument("--max-hold-bars", type=int, default=96)
    p.add_argument("--unit-exposure", type=float, default=1.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=52401)
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
        "device_type": args.device_type,
        "gpu_device_id": int(args.gpu_device_id),
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "path": str(BASELINE_MODEL)},
        "rows": {"train_all": int(len(train_df)), "train_entry": int(len(train_entry)), "train_direction": int(len(train_dir)), "validation": int(len(val_df)), "oos": int(len(oos_df))},
        "feature_count": {"entry": int(len(entry_cols)), "direction": int(len(direction_cols))},
        "ratios": {"entry_positive": float(np.mean(y_entry)), "direction_long_positive": float(np.mean(y_dir))},
        "audit_expected_model_found": audit.get("expected_model_found_in_manifest"),
    }, ensure_ascii=False, default=_json_default), flush=True)

    for i, spec in enumerate(_stage2_specs(), start=1):
        print(json.dumps({"stage": "fit_2stage", "done": i, "total": len(_stage2_specs()), "name": spec["name"], "entry_lgbm": spec["entry"].name, "direction_lgbm": spec["direction"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_lgbm(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + i * 100 + 1), device_type=str(args.device_type), gpu_device_id=int(args.gpu_device_id))
        direction_model = _fit_lgbm(x_train_dir, y_dir, w_dir, spec["direction"], int(args.seed + i * 100 + 2), device_type=str(args.device_type), gpu_device_id=int(args.gpu_device_id))

        p_entry_val = _binary_proba(entry_model, x_val_entry)
        p_long_val = _binary_proba(direction_model, x_val_dir)
        best_val: dict[str, Any] | None = None
        for entry_threshold in _grid(args.entry_thresholds):
            for side_threshold in _grid(args.side_thresholds):
                for margin_threshold in _grid(args.margin_thresholds):
                    val_actions, val_diag = _compose_2stage(p_entry_val, p_long_val, entry_threshold=entry_threshold, side_threshold=side_threshold, margin_threshold=margin_threshold)
                    metrics = _eval(val_df, val_actions, y_val, fee=float(args.fee), slip=float(args.slip), exposure=float(args.unit_exposure), max_hold=int(args.max_hold_bars))
                    candidate = {
                        "family": "two_stage",
                        "architecture": spec["name"],
                        "entry_lgbm": spec["entry"].name,
                        "direction_lgbm": spec["direction"].name,
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
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "two_stage", "head": "entry", "backend": "lightgbm"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": direction_model, "feature_cols": direction_cols, "family": "two_stage", "head": "direction", "backend": "lightgbm"}, best_val["artifact_paths"]["direction_model"])
        rows.append(best_val)
        if ("two_stage" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["two_stage"]["validation"]["score"])):
            best_by_family["two_stage"] = best_val

    for i, spec in enumerate(_ovr_specs(), start=1):
        print(json.dumps({"stage": "fit_ovr", "done": i, "total": len(_ovr_specs()), "name": spec["name"], "entry_lgbm": spec["entry"].name, "long_lgbm": spec["long"].name, "short_lgbm": spec["short"].name}, ensure_ascii=False), flush=True)
        entry_model = _fit_lgbm(x_train_entry, y_entry, w_entry, spec["entry"], int(args.seed + 1000 + i * 100 + 1), device_type=str(args.device_type), gpu_device_id=int(args.gpu_device_id))
        long_model = _fit_lgbm(x_train_dir, y_long, w_dir, spec["long"], int(args.seed + 1000 + i * 100 + 2), device_type=str(args.device_type), gpu_device_id=int(args.gpu_device_id))
        short_model = _fit_lgbm(x_train_dir, y_short, w_dir, spec["short"], int(args.seed + 1000 + i * 100 + 3), device_type=str(args.device_type), gpu_device_id=int(args.gpu_device_id))

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
                            "entry_lgbm": spec["entry"].name,
                            "long_lgbm": spec["long"].name,
                            "short_lgbm": spec["short"].name,
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
        joblib.dump({"model": entry_model, "feature_cols": entry_cols, "family": "ovr", "head": "entry", "backend": "lightgbm"}, best_val["artifact_paths"]["entry_model"])
        joblib.dump({"model": long_model, "feature_cols": direction_cols, "family": "ovr", "head": "long", "backend": "lightgbm"}, best_val["artifact_paths"]["long_model"])
        joblib.dump({"model": short_model, "feature_cols": direction_cols, "family": "ovr", "head": "short", "backend": "lightgbm"}, best_val["artifact_paths"]["short_model"])
        rows.append(best_val)
        if ("ovr" not in best_by_family) or (float(best_val["validation"]["score"]) > float(best_by_family["ovr"]["validation"]["score"])):
            best_by_family["ovr"] = best_val

    summary = {
        "model_id": MODEL_ID,
        "backend": "lightgbm",
        "device_type": args.device_type,
        "gpu_device_id": int(args.gpu_device_id),
        "baseline_fixed": {"track": "regime4_core", "single_hgb": "deeper", "oos_cost1_pnl": 1.22, "oos_cost1_mdd": -6.58, "oos_cost1_trades": 35, "path": str(BASELINE_MODEL)},
        "audit": audit,
        "feature_counts": {"entry": len(entry_cols), "direction": len(direction_cols)},
        "best_two_stage": best_by_family.get("two_stage"),
        "best_ovr": best_by_family.get("ovr"),
        "all_results": rows,
    }

    grid_csv = args.out_dir / "alpha5_24_lgbm_gpu_direction_refined_grid.csv"
    pd.DataFrame([
        {
            "family": row["family"],
            "architecture": row["architecture"],
            "entry_lgbm": row.get("entry_lgbm"),
            "direction_lgbm": row.get("direction_lgbm"),
            "long_lgbm": row.get("long_lgbm"),
            "short_lgbm": row.get("short_lgbm"),
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

    summary_path = args.out_dir / "alpha5_24_lgbm_gpu_direction_refined_summary.json"
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
