#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _grid as _runner_grid  # noqa: E402

# Parent artifacts produced by eval_alpha4_new_features_full_retrain_20260517
# were originally pickled while that file was executed as __main__.
# Expose the wrapper classes in this module's __main__ namespace so joblib can
# resolve those historical artifacts without retraining them.
FillNAWrapper = a4.FillNAWrapper
EncodedClassifierWrapper = a4.EncodedClassifierWrapper


MODEL_ID = "alpha4_2_teacher_ablation_20260517"
DEFAULT_ROOT = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517"
DEFAULT_TRAIN = DEFAULT_ROOT / "trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = DEFAULT_ROOT / "trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = DEFAULT_ROOT / "artifacts/hgb/parent.pkl"
DEFAULT_TEACHER = DEFAULT_ROOT / "artifacts/hgb/teacher_gate.pt"
DEFAULT_REPORT = DEFAULT_ROOT / "alpha4_2_teacher_ablation_summary.json"
DEFAULT_AUDIT = DEFAULT_ROOT / "alpha4_2_teacher_ablation_audit.json"
DEFAULT_GRID = DEFAULT_ROOT / "alpha4_2_teacher_ablation_grid.csv"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "teacher_ablation_artifacts"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _q0(df: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(df), 2), dtype=np.float32)


def _scaled_parent_decisions(base_dec: pd.DataFrame, rt: alpha2.Alpha2Runtime) -> pd.DataFrame:
    return alpha2._scale_parent_notional(base_dec, rt)


def _load_teacher(path: Path) -> tuple[Any, list[str], dict[str, np.ndarray], tuple[float, ...]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = alpha2._load_teacher_model(payload)
    return model, list(payload["feature_cols"]), dict(payload["train_meta"]["norm"]), tuple(float(x) for x in payload["buckets"])


def _teacher_decisions(
    df: pd.DataFrame,
    base_dec: pd.DataFrame,
    *,
    teacher_model: Any,
    teacher_cols: list[str],
    teacher_norm: dict[str, np.ndarray],
    buckets: tuple[float, ...],
    rt: alpha2.Alpha2Runtime,
) -> pd.DataFrame:
    features = prepare_features(df, side_hint=0, close=_close(df), feature_cols=teacher_cols)
    pred = teacher._predict_deep(teacher_model, features, teacher_cols, teacher_norm)
    return alpha2._decisions(base_dec, pred, buckets, rt)


def _select_teacher_runtime(
    *,
    val_df: pd.DataFrame,
    parent_for_features: dict[str, Any],
    existing_runner: dict[str, Any],
    base_val_dec: pd.DataFrame,
    teacher_model: Any,
    teacher_cols: list[str],
    teacher_norm: dict[str, np.ndarray],
    buckets: tuple[float, ...],
    limit_cfg: Any,
    fee: float,
    slip: float,
) -> tuple[alpha2.Alpha2Runtime, list[dict[str, Any]]]:
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    rows: list[dict[str, Any]] = []
    best_rt: alpha2.Alpha2Runtime | None = None
    best_score = -1e18
    for rt in alpha2._runtimes():
        dec = _teacher_decisions(
            val_df,
            base_val_dec,
            teacher_model=teacher_model,
            teacher_cols=teacher_cols,
            teacher_norm=teacher_norm,
            buckets=buckets,
            rt=rt,
        )
        metrics = a4._metrics(val_df, parent_for_features, existing_runner, noop_cfg, _q0(val_df), dec, _no_deep_overlay(), limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"stage": "teacher_runtime", "runtime": asdict(rt), "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"], "val_trades": metrics["cost1"]["trades"]})
        if score > best_score:
            best_score = score
            best_rt = rt
    assert best_rt is not None
    return best_rt, rows


def _no_deep_overlay() -> v31.OverlayConfig:
    return v31.OverlayConfig("alpha4_2_no_deep_parent_only", 99.0, 99.0, 0.0, 999, 0.04, 0.018, 48, 0.0, 1.0, 0.0, 0.0, 999, 0.0, 0.07, 0.035)


def _select_runner(
    *,
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    parent_for_features: dict[str, Any],
    train_dec: pd.DataFrame,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    limit_cfg: Any,
    fee: float,
    slip: float,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = _fit_cost_runner_with_decisions(train_df, parent_for_features, train_dec, fee=fee, slip=slip)
    rows: list[dict[str, Any]] = []
    selected_cfg: CostRunnerConfig | None = None
    selected_val: dict[str, Any] | None = None
    best_score = -1e18
    overlay = _no_deep_overlay()
    for cfg in _runner_grid():
        metrics = a4._metrics(val_df, parent_for_features, runner, cfg, _q0(val_df), val_dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"candidate": name, "stage": "runner_config", "runner_config": cfg.name, "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"], "val_trades": metrics["cost1"]["trades"]})
        if score > best_score:
            best_score = score
            selected_cfg = cfg
            selected_val = metrics
    assert selected_cfg is not None and selected_val is not None
    metrics = a4._metrics(eval_df, parent_for_features, runner, selected_cfg, _q0(eval_df), eval_dec, overlay, limit_cfg, fee=fee, slip=slip)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = out_dir / f"{name}_runner.pkl"
    joblib.dump({"model_id": MODEL_ID, "candidate": name, "cost_runner": runner, "selected_config": asdict(selected_cfg)}, runner_path)
    result = {
        "name": name,
        "teacher_layer": name.startswith("teacher_"),
        "metrics": metrics,
        "validation_metrics": selected_val,
        "selection_score": best_score,
        "oos_score": _score(metrics),
        "selected_runner_config": asdict(selected_cfg),
        "artifacts": {"runner": str(runner_path)},
    }
    return result, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate Alpha4.2 teacher layer under no-deep execution.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent = joblib.load(args.parent_model)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(parent["feature_cols"])
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    existing_runner_payload = joblib.load(v31.DEFAULT_JACKPOT)
    existing_runner = existing_runner_payload["cost_runner"]

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    raw_result, raw_rows = _select_runner(
        name="parent_direct_raw_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=base_train_dec,
        val_dec=base_val_dec,
        eval_dec=base_eval_dec,
        limit_cfg=limit_cfg,
        fee=fee,
        slip=slip,
        out_dir=args.out_dir,
    )
    experiments.append(raw_result)
    rows.extend(raw_rows)

    best_scale: tuple[alpha2.Alpha2Runtime, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    overlay = _no_deep_overlay()
    for rt in alpha2._runtimes():
        train_dec = _scaled_parent_decisions(base_train_dec, rt)
        val_dec = _scaled_parent_decisions(base_val_dec, rt)
        eval_dec = _scaled_parent_decisions(base_eval_dec, rt)
        metrics = a4._metrics(val_df, parent_for_features, existing_runner, noop_cfg, _q0(val_df), val_dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        rows.append({"candidate": "parent_direct_scaled_no_teacher", "stage": "scale_runtime", **asdict(rt), "score": score, "val_cost1_pnl": metrics["cost1"]["pnl"], "val_cost1_mdd": metrics["cost1"]["mdd"], "val_cost2_pnl": metrics["cost2"]["pnl"], "val_cost3_pnl": metrics["cost3"]["pnl"], "val_trades": metrics["cost1"]["trades"]})
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": metrics}, train_dec, val_dec, eval_dec)
    assert best_scale is not None
    scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
    scaled_result, scaled_rows = _select_runner(
        name="parent_direct_scaled_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=scale_train_dec,
        val_dec=scale_val_dec,
        eval_dec=scale_eval_dec,
        limit_cfg=limit_cfg,
        fee=fee,
        slip=slip,
        out_dir=args.out_dir,
    )
    scaled_result["selected_parent_scale_runtime"] = asdict(scale_rt)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)
    rows.extend(scaled_rows)

    teacher_model, teacher_cols, teacher_norm, buckets = _load_teacher(args.teacher_model)
    teacher_rt, teacher_rows = _select_teacher_runtime(
        val_df=val_df,
        parent_for_features=parent_for_features,
        existing_runner=existing_runner,
        base_val_dec=base_val_dec,
        teacher_model=teacher_model,
        teacher_cols=teacher_cols,
        teacher_norm=teacher_norm,
        buckets=buckets,
        limit_cfg=limit_cfg,
        fee=fee,
        slip=slip,
    )
    rows.extend([{"candidate": "teacher_constrained", **r} for r in teacher_rows])
    teacher_train_dec = _teacher_decisions(train_df, base_train_dec, teacher_model=teacher_model, teacher_cols=teacher_cols, teacher_norm=teacher_norm, buckets=buckets, rt=teacher_rt)
    teacher_val_dec = _teacher_decisions(val_df, base_val_dec, teacher_model=teacher_model, teacher_cols=teacher_cols, teacher_norm=teacher_norm, buckets=buckets, rt=teacher_rt)
    teacher_eval_dec = _teacher_decisions(eval_df, base_eval_dec, teacher_model=teacher_model, teacher_cols=teacher_cols, teacher_norm=teacher_norm, buckets=buckets, rt=teacher_rt)
    teacher_result, teacher_runner_rows = _select_runner(
        name="teacher_constrained",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=teacher_train_dec,
        val_dec=teacher_val_dec,
        eval_dec=teacher_eval_dec,
        limit_cfg=limit_cfg,
        fee=fee,
        slip=slip,
        out_dir=args.out_dir,
    )
    teacher_result["selected_teacher_runtime"] = asdict(teacher_rt)
    experiments.append(teacher_result)
    rows.extend(teacher_runner_rows)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    audit = {
        "status": "pass",
        "verdict": "remove_teacher" if best["name"].startswith("parent_direct") else "keep_teacher",
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after no-deep teacher ablation selection",
        "deep_sleeve": "disabled_for_teacher_ablation",
        "best_by_validation": best["name"],
        "parent_model": str(args.parent_model),
        "teacher_model": str(args.teacher_model),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Teacher layer ablation for Alpha4.2 under no-deep execution. Parent artifact is fixed. Each candidate retrains its own V21.2 runner on its own train decisions, selects runner config on 2025Q4, then evaluates fixed 2026 OOS.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "out_dir": str(args.out_dir)},
    }
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(args.grid_out, index=False)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
