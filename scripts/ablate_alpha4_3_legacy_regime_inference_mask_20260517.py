#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts.eval_alpha4_3_no_teacher_no_deep_20260517 import (  # noqa: E402
    DEFAULT_EVAL,
    DEFAULT_PARENT,
    DEFAULT_RUNNER,
    DEFAULT_TRAIN,
    _metrics,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha4_3_legacy_regime_inference_mask_20260517"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_legacy_regime_inference_mask_20260517"
OLD_CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"

FACTOR_CORE = [
    f"{OLD_CLEAN_PREFIX}factor_trend",
    f"{OLD_CLEAN_PREFIX}factor_flow",
    f"{OLD_CLEAN_PREFIX}factor_vol",
    f"{OLD_CLEAN_PREFIX}factor_crowding",
    f"{OLD_CLEAN_PREFIX}factor_liquidity",
    f"{OLD_CLEAN_PREFIX}trend_bias",
]
RISK_TRANSITION = [
    f"{OLD_CLEAN_PREFIX}risk_off_prob",
    f"{OLD_CLEAN_PREFIX}transition_risk",
]
SEMANTIC_PROBS = [
    f"{OLD_CLEAN_PREFIX}bull_prob",
    f"{OLD_CLEAN_PREFIX}bear_prob",
    f"{OLD_CLEAN_PREFIX}chop_prob",
    f"{OLD_CLEAN_PREFIX}whipsaw_prob",
    f"{OLD_CLEAN_PREFIX}normal_prob",
    f"{OLD_CLEAN_PREFIX}confidence",
    f"{OLD_CLEAN_PREFIX}entropy",
]
CLUSTER_STATE = [
    f"{OLD_CLEAN_PREFIX}state_code",
    f"{OLD_CLEAN_PREFIX}cluster",
    f"{OLD_CLEAN_PREFIX}cluster_confidence",
    f"{OLD_CLEAN_PREFIX}cluster_prob_0",
    f"{OLD_CLEAN_PREFIX}cluster_prob_1",
    f"{OLD_CLEAN_PREFIX}cluster_prob_2",
    f"{OLD_CLEAN_PREFIX}cluster_prob_3",
    f"{OLD_CLEAN_PREFIX}cluster_prob_4",
]

GROUPS = {
    "none": [],
    "factor_core": FACTOR_CORE,
    "risk_transition": RISK_TRANSITION,
    "semantic_probs": SEMANTIC_PROBS,
    "cluster_state": CLUSTER_STATE,
    "all_legacy": FACTOR_CORE + RISK_TRANSITION + SEMANTIC_PROBS + CLUSTER_STATE,
}


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        cost: {
            key: metrics[cost][key]
            for key in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional")
        }
        for cost in ("cost1", "cost2", "cost3")
    }


def _mask_frame(frame: pd.DataFrame, cols: list[str], medians: dict[str, float]) -> pd.DataFrame:
    if not cols:
        return frame.copy()
    out = frame.copy()
    for col in cols:
        if col in out.columns:
            out[col] = float(medians.get(col, 0.0))
    return out


def _feature_medians(train_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in cols:
        if col not in train_df.columns:
            medians[col] = 0.0
            continue
        s = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = s.median()
        medians[col] = float(0.0 if pd.isna(med) else med)
    return medians


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mask Alpha4.3 legacy clean-regime groups at inference with parent/runner/runtime fixed.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--runner-model", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--scale", type=float, default=0.85)
    p.add_argument("--max-notional", type=float, default=2.75)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(args.eval_csv)

    parent = joblib.load(args.parent_model)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    parent_for_features = dict(parent_ref)
    parent_for_features["feature_cols"] = list(parent["feature_cols"])
    runner_payload = joblib.load(args.runner_model)
    runner = runner_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(runner_payload["selected_config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    rt = alpha2.Alpha2Runtime(
        name=f"parent_direct_scale{float(args.scale):.2f}",
        confidence=0.0,
        parent_notional_scale=float(args.scale),
        max_notional=float(args.max_notional),
    )

    all_cols = sorted({c for cols in GROUPS.values() for c in cols})
    medians = _feature_medians(train_df, all_cols)
    results: list[dict[str, Any]] = []
    for group, cols in GROUPS.items():
        masked_val = _mask_frame(val_df, cols, medians)
        masked_eval = _mask_frame(eval_df, cols, medians)
        val_metrics = _metrics(
            masked_val,
            parent_for_features=parent_for_features,
            parent=parent,
            runner=runner,
            add_cfg=add_cfg,
            rt=rt,
            fee=fee,
            slip=slip,
        )
        eval_metrics = _metrics(
            masked_eval,
            parent_for_features=parent_for_features,
            parent=parent,
            runner=runner,
            add_cfg=add_cfg,
            rt=rt,
            fee=fee,
            slip=slip,
        )
        result = {
            "mask_group": group,
            "masked_cols": cols,
            "masked_col_count": int(len(cols)),
            "mask_policy": "replace_with_2025_train_median",
            "validation_metrics": val_metrics,
            "selected_metrics": eval_metrics,
        }
        results.append(result)
        print(json.dumps({"mask_group": group, "metrics": _compact(eval_metrics)}, ensure_ascii=False, default=_json_default), flush=True)

    baseline = next(r for r in results if r["mask_group"] == "none")
    rows = []
    for result in results:
        row = {
            "mask_group": result["mask_group"],
            "masked_col_count": result["masked_col_count"],
        }
        for cost in ("cost1", "cost2", "cost3"):
            m = result["selected_metrics"][cost]
            b = baseline["selected_metrics"][cost]
            row[f"{cost}_pnl"] = m["pnl"]
            row[f"{cost}_pnl_delta_vs_none"] = m["pnl"] - b["pnl"]
            row[f"{cost}_mdd"] = m["mdd"]
            row[f"{cost}_mdd_delta_vs_none"] = m["mdd"] - b["mdd"]
            row[f"{cost}_trades"] = m["trades"]
        rows.append(row)
    results_csv = args.out_dir / "alpha4_3_legacy_regime_inference_mask_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    report = {
        "model_id": MODEL_ID,
        "method": "fixed Alpha4.3 parent/runner/runtime; replace selected legacy clean-regime columns with 2025 train medians at inference",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "parent_model": str(args.parent_model),
        "runner_model": str(args.runner_model),
        "runtime": dict(rt.__dict__),
        "selected_runner_config": dict(runner_payload["selected_config"]),
        "parent_feature_count": int(len(parent["feature_cols"])),
        "parent_legacy_feature_count": int(sum(c.startswith(OLD_CLEAN_PREFIX) for c in parent["feature_cols"])),
        "groups": {k: v for k, v in GROUPS.items()},
        "feature_medians": medians,
        "results": results,
        "artifacts": {
            "report": str(args.out_dir / "alpha4_3_legacy_regime_inference_mask_summary.json"),
            "results_csv": str(results_csv),
        },
    }
    report_path = args.out_dir / "alpha4_3_legacy_regime_inference_mask_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "results_csv": str(results_csv)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
