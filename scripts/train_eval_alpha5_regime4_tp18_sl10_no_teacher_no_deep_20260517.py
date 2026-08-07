#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_v2_retrained_downstream_20260515 import _fit_cost_runner_with_decisions  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha5_regime4_state24_sticky090_tp18_sl10_no_teacher_no_deep_20260517"
BASE_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
DEFAULT_TRAIN = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/fixed_regime4_state24_sticky090_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_state24_sticky090_tp18_sl10_fixed.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_regime4_state24_sticky090_tp18_sl10_no_teacher_no_deep_20260517"
OLD_CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
REGIME4_PREFIXES = ("clean_regime4_2024_unsup_v1_", "regime4_pred_")
EXTRA_FEATURES = ("tp_sl_action_score",)
DROP_RETRAIN_FEATURES = set(a4.DROP_RETRAIN_FEATURES)


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    cols = [
        c for c in list(parent_ref["feature_cols"])
        if c not in DROP_RETRAIN_FEATURES and not c.startswith(OLD_CLEAN_PREFIX)
    ]
    common = set(train.columns) & set(eval_df.columns)
    for col in EXTRA_FEATURES:
        if col in common and col not in cols:
            cols.append(col)
    for col in sorted(c for c in common if c.startswith(REGIME4_PREFIXES)):
        if col not in cols:
            cols.append(col)
    return cols


def _no_deep_overlay() -> v31.OverlayConfig:
    return v31.OverlayConfig("alpha5_regime4_no_deep", 99.0, 99.0, 0.0, 999, 0.04, 0.018, 48, 0.0, 1.0, 0.0, 0.0, 999, 0.0, 0.07, 0.035)


def _q0(df: pd.DataFrame) -> np.ndarray:
    return np.zeros((len(df), 2), dtype=np.float32)


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _scale_decisions(base_dec: pd.DataFrame, rt: alpha2.Alpha2Runtime) -> pd.DataFrame:
    return alpha2._scale_parent_notional(base_dec, rt)


def _metrics(
    df: pd.DataFrame,
    *,
    parent_for_features: dict[str, Any],
    runner: dict[str, Any],
    runner_cfg: CostRunnerConfig,
    dec: pd.DataFrame,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return a4._metrics(
        df,
        parent_for_features,
        runner,
        runner_cfg,
        _q0(df),
        dec,
        _no_deep_overlay(),
        ft_v2.ft_v1._limit_cfg(),
        fee=fee,
        slip=slip,
    )


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
    fee: float,
    slip: float,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = _fit_cost_runner_with_decisions(train_df, parent_for_features, train_dec, fee=fee, slip=slip)
    rows: list[dict[str, Any]] = []
    best_cfg: CostRunnerConfig | None = None
    best_val: dict[str, Any] | None = None
    best_score = -1e18
    for cfg in _runner_grid():
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score(val_metrics)
        rows.append({"candidate": name, "runner_config": cfg.name, "score": score, "val_cost1_pnl": val_metrics["cost1"]["pnl"], "val_cost1_mdd": val_metrics["cost1"]["mdd"], "val_cost2_pnl": val_metrics["cost2"]["pnl"], "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_trades": val_metrics["cost1"]["trades"]})
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_val = val_metrics
    assert best_cfg is not None and best_val is not None
    eval_metrics = _metrics(eval_df, parent_for_features=parent_for_features, runner=runner, runner_cfg=best_cfg, dec=eval_dec, fee=fee, slip=slip)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = out_dir / f"{name}_runner.pkl"
    joblib.dump({"model_id": MODEL_ID, "candidate": name, "cost_runner": runner, "selected_config": asdict(best_cfg)}, runner_path)
    return {
        "name": name,
        "selection_score": float(best_score),
        "validation_metrics": best_val,
        "metrics": eval_metrics,
        "selected_runner_config": asdict(best_cfg),
        "artifacts": {"runner": str(runner_path)},
    }, rows


def _compact_costs(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "cost1": {k: metrics["cost1"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
        "cost2": {k: metrics["cost2"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
        "cost3": {k: metrics["cost3"][k] for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Alpha5 Regime4 + TP/SL parent under Alpha4.3 no-teacher/no-deep structure.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--report-stem", default="alpha5_regime4_tp18_sl10_no_teacher_no_deep")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=5517)
    return p.parse_args()


def main() -> int:
    global MODEL_ID
    args = parse_args()
    MODEL_ID = str(args.model_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    feature_cols = _feature_cols(train_all, eval_df)
    forbidden_old = [c for c in feature_cols if c.startswith(OLD_CLEAN_PREFIX)]
    if forbidden_old:
        raise ValueError("old clean regime features leaked into parent input: " + ",".join(forbidden_old[:20]))
    x_train, y_train, train_meta = build_training_set(train_df, cfg=label_cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    parent = train_policy(x_train, y_train, cfg=label_cfg, random_state=int(args.seed), feature_cols=feature_cols)
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)
    parent_path = args.out_dir / "parent.pkl"
    joblib.dump(parent, parent_path)

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    raw_result, raw_rows = _select_runner(
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

    best_scale: tuple[alpha2.Alpha2Runtime, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    for rt in alpha2._runtimes():
        train_dec = _scale_decisions(base_train_dec, rt)
        val_dec = _scale_decisions(base_val_dec, rt)
        eval_dec = _scale_decisions(base_eval_dec, rt)
        val_metrics = _metrics(val_df, parent_for_features=parent_for_features, runner=noop_runner, runner_cfg=noop_cfg, dec=val_dec, fee=fee, slip=slip)
        score = _score(val_metrics)
        grid_rows.append({"candidate": "parent_direct_scaled_no_teacher", "stage": "scale_runtime", **asdict(rt), "score": score, "val_cost1_pnl": val_metrics["cost1"]["pnl"], "val_cost1_mdd": val_metrics["cost1"]["mdd"], "val_cost2_pnl": val_metrics["cost2"]["pnl"], "val_cost3_pnl": val_metrics["cost3"]["pnl"], "val_trades": val_metrics["cost1"]["trades"]})
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
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
        fee=fee,
        slip=slip,
        out_dir=args.out_dir / "runners",
    )
    scaled_result["selected_parent_scale_runtime"] = asdict(scale_rt)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)
    grid_rows.extend(scaled_rows)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    report_path = args.out_dir / f"{args.report_stem}_summary.json"
    grid_path = args.out_dir / f"{args.report_stem}_grid.csv"
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha4.3 no-teacher/no-deep structure retrained with Regime4 replacement features and fixed TP/SL action score. Old clean_regime_2024_unsup_v4_* inputs are removed; clean_regime4_2024_unsup_v1_* and regime4_pred_* are used instead.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "feature_contract": {
            "feature_count": int(len(feature_cols)),
            "old_clean_regime_feature_count": int(sum(c.startswith(OLD_CLEAN_PREFIX) for c in feature_cols)),
            "current_regime4_feature_count": int(sum(c.startswith("clean_regime4_2024_unsup_v1_") for c in feature_cols)),
            "future_regime4_feature_count": int(sum(c.startswith("regime4_pred_") for c in feature_cols)),
            "contains_tp_sl_action_score": "tp_sl_action_score" in feature_cols,
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
        "audit": {
            "status": "pass",
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "teacher_layer": "disabled",
            "deep_scout": "disabled",
            "old_clean_regime_replaced": True,
            "train_meta": train_meta,
        },
        "artifacts": {
            "parent": str(parent_path),
            "report": str(report_path),
            "grid": str(grid_path),
        },
    }
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(grid_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": report["artifacts"]["report"], "best": best["name"], "metrics": _compact_costs(best["metrics"])}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
