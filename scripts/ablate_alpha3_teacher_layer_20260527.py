#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import (  # noqa: E402
    ExitGuardConfig,
    _default_limit_cfg,
    _metrics_guard,
    _sl_ratio,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_teacher_layer_ablation_20260527"
BASE_REPORT = ROOT / "data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json"
GRID_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_grid.csv"
REPORT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_summary.json"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _merge_state24(base: pd.DataFrame, side_path: Path) -> pd.DataFrame:
    side = alpha3_full._rename_state24_sidecar(_read(side_path))
    merged, _ = alpha3_full._merge_state24(base, side)
    return merged


def _load_stack() -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    rep = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    exp = dict(rep["experiments"][-1])
    parent = joblib.load(exp["artifacts"]["parent"])
    runner_payload = joblib.load(exp["artifacts"]["runner"])
    runner = runner_payload["cost_runner"]
    add_cfg = alpha3_full.v21.CostRunnerConfig(**dict(exp["selected_runner_config"]))
    overlay = alpha3_full.v31.OverlayConfig(**dict(exp["selected_overlay"]))
    runtime = alpha2.Alpha2Runtime(**dict(exp["selected_teacher_runtime"]))
    teacher_payload = torch.load(exp["artifacts"]["teacher"], map_location="cpu", weights_only=False)
    teacher_model = alpha2._load_teacher_model(teacher_payload)
    teacher_cols = list(teacher_payload["feature_cols"])
    teacher_norm = dict(teacher_payload["train_meta"]["norm"])
    teacher_buckets = tuple(float(x) for x in teacher_payload["buckets"])
    deep_payload = torch.load(exp["artifacts"]["deep_scout"], map_location="cpu", weights_only=False)
    deep_model = v27.DeepAlphaTCN(len(deep_payload["seq_cols"]))
    deep_model.load_state_dict(deep_payload["state_dict"])
    deep_model = deep_model.cpu().eval()
    baseline_metrics = rep.get("alpha_sub_version_metrics", rep.get("candidate_metrics"))
    return parent, runner, add_cfg, overlay, runtime, teacher_model, teacher_cols, teacher_norm, teacher_buckets, deep_model, deep_payload, baseline_metrics


def _build_decisions(
    variant: str,
    parent_dec: pd.DataFrame,
    pred: dict[str, np.ndarray],
    buckets: tuple[float, ...],
    rt: alpha2.Alpha2Runtime,
) -> pd.DataFrame:
    if variant == "baseline_teacher":
        return alpha2._decisions(parent_dec, pred, buckets, rt)
    if variant == "no_teacher_parent_direct":
        return parent_dec.copy()
    if variant == "no_teacher_parent_scaled":
        return alpha2._scale_parent_notional(parent_dec.copy(), rt)
    if variant == "teacher_noflip_conf0":
        tr = teacher.Runtime(
            name="teacher_noflip_conf0",
            confidence=0.0,
            skip_on_cash=True,
            allow_flip=False,
            use_learned_size=False,
            notional_scale=1.0,
            max_notional=float(rt.max_notional),
        )
        d = teacher._constrained_decisions(parent_dec, pred, buckets, tr)
        return alpha2._scale_parent_notional(d, rt)
    if variant == "teacher_flip_conf0":
        tr = teacher.Runtime(
            name="teacher_flip_conf0",
            confidence=0.0,
            skip_on_cash=True,
            allow_flip=True,
            use_learned_size=False,
            notional_scale=1.0,
            max_notional=float(rt.max_notional),
        )
        d = teacher._constrained_decisions(parent_dec, pred, buckets, tr)
        return alpha2._scale_parent_notional(d, rt)
    raise ValueError(f"unknown variant: {variant}")


def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    parent, runner, add_cfg, overlay, runtime, teacher_model, teacher_cols, teacher_norm, teacher_buckets, deep_model, deep_payload, baseline_metrics = _load_stack()
    fee = float(parent["config"]["fee"])
    slip = float(parent["config"]["slip"])
    limit_cfg = _default_limit_cfg()
    guard_cfg = ExitGuardConfig(
        name="guard_soft3_hard1p45",
        hard_sl_mult=1.45,
        soft_sl_mult=1.0,
        early_bars=18,
        early_sl_mult=1.35,
        soft_min_hold=3,
        soft_persist_bars=3,
        regime_bad_th=0.50,
        flow_bad_th=0.02,
        giveback_trigger=0.72,
        giveback_min_mfe=0.014,
        giveback_min_hold=3,
        entry_quality_min=-999.0,
        entry_conf_min=0.0,
        same_side_entry_gap=0,
        cooldown_after_hard_stop=0,
        cooldown_after_soft_stop=0,
        cooldown_after_giveback=0,
    )

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_all = _merge_state24(train_all, alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    val_parent = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_parent = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=teacher_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    val_pred = teacher._predict_deep(teacher_model, val_features, teacher_cols, teacher_norm)
    eval_pred = teacher._predict_deep(teacher_model, eval_features, teacher_cols, teacher_norm)
    val_q = v27._predict_all(deep_model, val_df, deep_payload["seq_cols"], deep_payload["norm"])
    eval_q = v27._predict_all(deep_model, eval_df, deep_payload["seq_cols"], deep_payload["norm"])

    variants = [
        "baseline_teacher",
        "no_teacher_parent_direct",
        "no_teacher_parent_scaled",
        "teacher_noflip_conf0",
        "teacher_flip_conf0",
    ]
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for name in variants:
        val_dec = _build_decisions(name, val_parent, val_pred, teacher_buckets, runtime)
        eval_dec = _build_decisions(name, eval_parent, eval_pred, teacher_buckets, runtime)
        val_metrics = _metrics_guard(val_df, parent, runner, add_cfg, val_q, val_dec, overlay, limit_cfg, guard_cfg, fee=fee, slip=slip)
        eval_metrics = _metrics_guard(eval_df, parent, runner, add_cfg, eval_q, eval_dec, overlay, limit_cfg, guard_cfg, fee=fee, slip=slip)
        c3 = eval_metrics["cost3"]
        row = {
            "variant": name,
            "val_score": float(_score(val_metrics)),
            "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
            "oos_score": float(_score(eval_metrics)),
            "oos_cost3_pnl": float(c3["pnl"]),
            "oos_cost3_mdd": float(c3["mdd"]),
            "oos_cost3_wr": float(c3["wr"]),
            "oos_cost3_trades": int(c3["trades"]),
            "oos_sl_ratio": float(_sl_ratio(c3)),
        }
        rows.append(row)
        details.append({"variant": name, "metrics": eval_metrics})

    grid = pd.DataFrame(rows).sort_values("val_score", ascending=False).reset_index(drop=True)
    grid.to_csv(GRID_OUT, index=False)
    selected = str(grid.iloc[0]["variant"])
    selected_row = next(r for r in rows if r["variant"] == selected)
    baseline_row = next(r for r in rows if r["variant"] == "baseline_teacher")
    delta_vs_teacher = {
        "cost3_pnl": float(selected_row["oos_cost3_pnl"] - baseline_row["oos_cost3_pnl"]),
        "cost3_mdd": float(selected_row["oos_cost3_mdd"] - baseline_row["oos_cost3_mdd"]),
        "cost3_wr": float(selected_row["oos_cost3_wr"] - baseline_row["oos_cost3_wr"]),
        "cost3_trades": int(selected_row["oos_cost3_trades"] - baseline_row["oos_cost3_trades"]),
        "sl_ratio": float(selected_row["oos_sl_ratio"] - baseline_row["oos_sl_ratio"]),
    }
    delta_vs_alpha3_baseline = {
        "cost3_pnl": float(selected_row["oos_cost3_pnl"] - float(baseline_metrics["cost3"]["pnl"])),
        "cost3_mdd": float(selected_row["oos_cost3_mdd"] - float(baseline_metrics["cost3"]["mdd"])),
        "cost3_wr": float(selected_row["oos_cost3_wr"] - float(baseline_metrics["cost3"]["wr"])),
        "cost3_trades": int(selected_row["oos_cost3_trades"] - int(baseline_metrics["cost3"]["trades"])),
        "sl_ratio": float(selected_row["oos_sl_ratio"] - _sl_ratio(baseline_metrics["cost3"])),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Teacher layer ablation under fixed guard/execution stack: full remove, scaled remove, confidence gate off, and flip enabled.",
        "protocol": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
        },
        "base_model": str(BASE_REPORT),
        "guard_config": asdict(guard_cfg),
        "rows": rows,
        "selected_variant": selected,
        "selected_row": selected_row,
        "delta_vs_teacher_baseline": delta_vs_teacher,
        "delta_vs_alpha3_baseline": delta_vs_alpha3_baseline,
        "grid": str(GRID_OUT),
        "details": details,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "selected_variant": selected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
