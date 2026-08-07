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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    EXTRA_FEATURES,
    MODEL_ID as BASE_MODEL_ID,
    OLD_CLEAN_PREFIX,
    REGIME4_PREFIXES,
    _compact_costs,
    _metrics,
    _q0,
    _scale_decisions,
    _select_runner,
)
from ensemble.fully_learned_governor_policy import FullyLearnedGovernorConfig, build_training_set, predict_policy_frame, train_policy  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha5_1_regime4_state24_sticky090_interactions_no_teacher_no_deep_20260517"
BASE_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_1_regime4_state24_sticky090_interactions_no_teacher_no_deep_20260517"
CLASSES = ("bull", "bear", "chop", "whipsaw")


def _safe(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def add_regime_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    tp = _safe(out, "tp_sl_action_score")
    created: list[str] = []
    for prefix, tag in (
        ("regime4_pred_", "future"),
        ("clean_regime4_2024_unsup_v1_", "current"),
    ):
        for cls in CLASSES:
            src = f"{prefix}{cls}_prob"
            if src in out.columns:
                name = f"tp_sl_x_{tag}_regime4_{cls}"
                out[name] = tp * _safe(out, src)
                created.append(name)
        for src_name, suffix in (
            ("trend_prob", "trend_prob"),
            ("micro_prob", "micro_prob"),
            ("directional_bias", "directional_bias"),
            ("range_prob", "range_prob"),
            ("instability_prob", "instability_prob"),
            ("confidence", "confidence"),
            ("margin", "margin"),
        ):
            src = f"{prefix}{src_name}"
            if src in out.columns:
                name = f"tp_sl_x_{tag}_{suffix}"
                out[name] = tp * _safe(out, src)
                created.append(name)
    return out, created


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame, interaction_cols: list[str]) -> list[str]:
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    drop = set(__import__("scripts.eval_alpha4_new_features_full_retrain_20260517", fromlist=["DROP_RETRAIN_FEATURES"]).DROP_RETRAIN_FEATURES)
    cols = [
        c for c in list(parent_ref["feature_cols"])
        if c not in drop and not c.startswith(OLD_CLEAN_PREFIX)
    ]
    common = set(train.columns) & set(eval_df.columns)
    for col in EXTRA_FEATURES:
        if col in common and col not in cols:
            cols.append(col)
    for col in sorted(c for c in common if c.startswith(REGIME4_PREFIXES)):
        if col not in cols:
            cols.append(col)
    for col in interaction_cols:
        if col in common and col not in cols:
            cols.append(col)
    return cols


def _score_v2(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    pnl = 0.30 * float(c1["pnl"]) + 0.35 * float(c2["pnl"]) + 0.35 * float(c3["pnl"])
    mdd = 0.25 * abs(float(c1["mdd"])) + 0.35 * abs(float(c2["mdd"])) + 0.40 * abs(float(c3["mdd"]))
    trades = min(float(c1["trades"]), 130.0)
    trade_floor_penalty = max(0.0, 45.0 - float(c1["trades"])) * 1.5
    notional_penalty = max(0.0, float(c1.get("avg_notional", 0.0)) - 2.55) * 16.0
    return float(pnl + 0.05 * trades + 1.4 * pnl / max(mdd, 1.0) - max(0.0, mdd - 22.0) * 2.6 - trade_floor_penalty - notional_penalty)


def _select_runner_v2(
    *,
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    parent_for_features: dict[str, Any],
    train_dec: pd.DataFrame,
    val_dec: pd.DataFrame,
    eval_dec: pd.DataFrame,
    fee: float,
    slip: float,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = _fit_cost_runner_with_decisions(train_df, parent_for_features, train_dec, fee=fee, slip=slip)
    rows: list[dict[str, Any]] = []
    best_cfg = None
    best_val = None
    best_score = -1e18
    for cfg in _runner_grid():
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score_v2(val_metrics)
        rows.append({"candidate": name, "runner_config": cfg.name, "score": score, "val_cost1_pnl": val_metrics["cost1"]["pnl"], "val_cost1_mdd": val_metrics["cost1"]["mdd"], "val_cost2_pnl": val_metrics["cost2"]["pnl"], "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_trades": val_metrics["cost1"]["trades"]})
        if score > best_score:
            best_score = float(score)
            best_cfg = cfg
            best_val = val_metrics
    assert best_cfg is not None and best_val is not None
    eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=best_cfg, dec=eval_dec, fee=fee, slip=slip)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = out_dir / f"{name}_runner.pkl"
    joblib.dump({"model_id": MODEL_ID, "candidate": name, "cost_runner": runner, "selected_config": dict(best_cfg.__dict__), "selection_score_v2": best_score}, runner_path)
    return {
        "name": name,
        "selection_score": best_score,
        "selection_score_policy": "cost_stress_v2",
        "validation_metrics": best_val,
        "metrics": eval_metrics,
        "selected_runner_config": dict(best_cfg.__dict__),
        "artifacts": {"runner": str(runner_path)},
    }, rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Alpha5.1 Regime4 interaction parent under no-teacher/no-deep structure.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=5617)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all, interaction_cols = add_regime_interactions(_read(args.train_csv))
    eval_df, eval_interactions = add_regime_interactions(_read(args.eval_csv))
    interaction_cols = sorted(set(interaction_cols) & set(eval_interactions))
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    feature_cols = _feature_cols(train_all, eval_df, interaction_cols)
    if any(c.startswith(OLD_CLEAN_PREFIX) for c in feature_cols):
        raise ValueError("old clean regime feature leaked")
    x_train, y_train, train_meta = build_training_set(train_df, cfg=label_cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    parent = train_policy(x_train, y_train, cfg=label_cfg, random_state=int(args.seed), feature_cols=feature_cols)
    parent_path = args.out_dir / "parent.pkl"
    joblib.dump(parent, parent_path)
    parent_for_features = dict(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    raw_result, raw_rows = _select_runner_v2(
        name="parent_direct_raw_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=base_train_dec,
        val_dec=base_val_dec,
        eval_dec=base_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=args.out_dir / "runners",
    )
    experiments.append(raw_result)
    grid_rows.extend(raw_rows)

    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    best_scale = None
    for rt in alpha2._runtimes():
        train_dec = _scale_decisions(base_train_dec, rt)
        val_dec = _scale_decisions(base_val_dec, rt)
        eval_dec = _scale_decisions(base_eval_dec, rt)
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score_v2(val_metrics)
        grid_rows.append({"candidate": "parent_direct_scaled_no_teacher", "stage": "scale_runtime", **dict(rt.__dict__), "score": score, "val_cost1_pnl": val_metrics["cost1"]["pnl"], "val_cost1_mdd": val_metrics["cost1"]["mdd"], "val_cost2_pnl": val_metrics["cost2"]["pnl"], "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_trades": val_metrics["cost1"]["trades"]})
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
    assert best_scale is not None
    scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
    scaled_result, scaled_rows = _select_runner_v2(
        name="parent_direct_scaled_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=scale_train_dec,
        val_dec=scale_val_dec,
        eval_dec=scale_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=args.out_dir / "runners",
    )
    scaled_result["selected_parent_scale_runtime"] = dict(scale_rt.__dict__)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)
    grid_rows.extend(scaled_rows)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    report = {
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "design": "Alpha5.1 adds Regime4 x TP/SL interaction features and uses a cost-stress selection score. Teacher and deep scout remain disabled.",
        "feature_contract": {
            "feature_count": int(len(feature_cols)),
            "old_clean_regime_feature_count": int(sum(c.startswith(OLD_CLEAN_PREFIX) for c in feature_cols)),
            "current_regime4_feature_count": int(sum(c.startswith("clean_regime4_2024_unsup_v1_") for c in feature_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
            "interaction_feature_count": int(len([c for c in feature_cols if c.startswith("tp_sl_x_")])),
            "contains_tp_sl_action_score": "tp_sl_action_score" in feature_cols,
            "interaction_cols": [c for c in feature_cols if c.startswith("tp_sl_x_")],
            "feature_cols": feature_cols,
        },
        "split": {
            "train": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
            "selection": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
            "oos": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        },
        "experiments": experiments,
        "best_by_selection": best["name"],
        "selected_metrics": _compact_costs(best["metrics"]),
        "audit": {"status": "pass", "teacher_layer": "disabled", "deep_scout": "disabled", "selection_uses_2026": False, "train_meta": train_meta},
        "artifacts": {
            "parent": str(parent_path),
            "report": str(args.out_dir / "alpha5_1_regime4_interactions_no_teacher_no_deep_summary.json"),
            "grid": str(args.out_dir / "alpha5_1_regime4_interactions_no_teacher_no_deep_grid.csv"),
        },
    }
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(args.out_dir / "alpha5_1_regime4_interactions_no_teacher_no_deep_grid.csv", index=False)
    (args.out_dir / "alpha5_1_regime4_interactions_no_teacher_no_deep_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": report["artifacts"]["report"], "best": best["name"], "metrics": _compact_costs(best["metrics"])}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
