#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import joblib
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
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    BASE_PARENT,
    OLD_CLEAN_PREFIX,
    _compact_costs,
    _metrics,
    _scale_decisions,
    _score,
    _select_runner,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha4_3_legacy_regime_block_ablation_20260517"
DEFAULT_TRAIN = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_legacy_regime_block_ablation_20260517"
ALPHA4_3_PARENT = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl"

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
STICKY_CURRENT_PREFIX = "clean_regime4_2024_unsup_v1_"
STICKY_CURRENT = [
    f"{STICKY_CURRENT_PREFIX}bull_prob",
    f"{STICKY_CURRENT_PREFIX}bear_prob",
    f"{STICKY_CURRENT_PREFIX}chop_prob",
    f"{STICKY_CURRENT_PREFIX}whipsaw_prob",
    f"{STICKY_CURRENT_PREFIX}trend_prob",
    f"{STICKY_CURRENT_PREFIX}micro_prob",
    f"{STICKY_CURRENT_PREFIX}directional_bias",
    f"{STICKY_CURRENT_PREFIX}range_prob",
    f"{STICKY_CURRENT_PREFIX}instability_prob",
    f"{STICKY_CURRENT_PREFIX}confidence",
    f"{STICKY_CURRENT_PREFIX}entropy",
    f"{STICKY_CURRENT_PREFIX}margin",
    f"{STICKY_CURRENT_PREFIX}factor_trend",
    f"{STICKY_CURRENT_PREFIX}factor_flow",
    f"{STICKY_CURRENT_PREFIX}factor_vol",
    f"{STICKY_CURRENT_PREFIX}factor_crowding",
    f"{STICKY_CURRENT_PREFIX}factor_liquidity",
    f"{STICKY_CURRENT_PREFIX}trend_bias",
    f"{STICKY_CURRENT_PREFIX}risk_off_prob",
    f"{STICKY_CURRENT_PREFIX}transition_risk",
]
ALPHA61_META_MARKER = "__alpha61_meta__"
ALPHA61_DERIVED_META_MARKER = "__alpha61_derived_meta__"
ALPHA61_DERIVED_META_COLS = [
    "a61_consensus_long",
    "a61_consensus_short",
    "a61_quality_top",
    "a61_quality_mean",
    "a61_quality_dispersion",
    "a61_long_edge_sum",
    "a61_short_edge_sum",
    "a61_disagreement_entropy",
    "a61_horizon_mean",
    "a61_horizon_std",
    "a61_short_long_edge",
    "a61_short_short_edge",
    "a61_mid_long_edge",
    "a61_mid_short_edge",
    "a61_long_long_edge",
    "a61_long_short_edge",
    "a61_primary_adverse_quality_gap",
    "a61_primary_sam_quality_gap",
    "a61_risk_opposition",
    "a61_active_model_count",
]

VARIANTS = {
    "no_legacy": [],
    "sticky_current": STICKY_CURRENT,
    "sticky_alpha61": STICKY_CURRENT + [ALPHA61_META_MARKER],
    "sticky_alpha61_derived": STICKY_CURRENT + [ALPHA61_DERIVED_META_MARKER],
    "factor_core": FACTOR_CORE,
    "risk_transition": RISK_TRANSITION,
    "semantic_probs": SEMANTIC_PROBS,
    "cluster_state": CLUSTER_STATE,
    "all_legacy": FACTOR_CORE + RISK_TRANSITION + SEMANTIC_PROBS + CLUSTER_STATE,
}


def _feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame, variant_cols: list[str], *, feature_basis: str) -> list[str]:
    parent_path = ALPHA4_3_PARENT if feature_basis == "alpha4_3" else v31.DEFAULT_PARENT
    parent_ref = joblib.load(parent_path)
    common = set(train.columns) & set(eval_df.columns)
    cols = [c for c in list(parent_ref["feature_cols"]) if not c.startswith(OLD_CLEAN_PREFIX)]
    if "tp_sl_action_score" in common and "tp_sl_action_score" not in cols:
        cols.append("tp_sl_action_score")
    if ALPHA61_META_MARKER in variant_cols:
        for col in sorted(c for c in common if c.startswith("a61_")):
            if col not in cols:
                cols.append(col)
    if ALPHA61_DERIVED_META_MARKER in variant_cols:
        for col in ALPHA61_DERIVED_META_COLS:
            if col in common and col not in cols:
                cols.append(col)
    for col in variant_cols:
        if col in {ALPHA61_META_MARKER, ALPHA61_DERIVED_META_MARKER}:
            continue
        if col in common and col not in cols:
            cols.append(col)
    return cols


def _run_variant(
    *,
    name: str,
    variant_cols: list[str],
    train_all: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    stride: int,
    seed: int,
    feature_basis: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    base_parent_ref = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent_ref["config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    feature_cols = _feature_cols(train_all, eval_df, variant_cols, feature_basis=feature_basis)
    selected_legacy = [c for c in feature_cols if c.startswith(OLD_CLEAN_PREFIX)]
    selected_sticky = [c for c in feature_cols if c.startswith(STICKY_CURRENT_PREFIX)]

    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, train_meta = build_training_set(
        train_df,
        cfg=label_cfg,
        stride_bars=int(stride),
        batch_size=512,
        feature_cols=feature_cols,
    )
    parent = train_policy(x_train, y_train, cfg=label_cfg, random_state=int(seed), feature_cols=feature_cols)
    parent_path = variant_dir / "parent.pkl"
    joblib.dump(parent, parent_path)
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = list(feature_cols)

    base_train_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    base_val_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    base_eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    experiments: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    raw_result, raw_rows = _select_runner(
        name=f"{name}__parent_direct_raw_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=base_train_dec,
        val_dec=base_val_dec,
        eval_dec=base_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=variant_dir / "runners",
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
        val_metrics = _metrics(
            val_df,
            parent_for_features=parent_for_features,
            runner=noop_runner,
            runner_cfg=noop_cfg,
            dec=val_dec,
            fee=fee,
            slip=slip,
        )
        score = _score(val_metrics)
        grid_rows.append(
            {
                "candidate": f"{name}__parent_direct_scaled_no_teacher",
                "stage": "scale_runtime",
                **rt.__dict__,
                "score": score,
                "val_cost1_pnl": val_metrics["cost1"]["pnl"],
                "val_cost1_mdd": val_metrics["cost1"]["mdd"],
                "val_cost2_pnl": val_metrics["cost2"]["pnl"],
                "val_cost3_pnl": val_metrics["cost3"]["pnl"],
                "val_trades": val_metrics["cost1"]["trades"],
            }
        )
        if best_scale is None or score > best_scale[1]["score"]:
            best_scale = (rt, {"score": score, "metrics": val_metrics}, train_dec, val_dec, eval_dec)
    assert best_scale is not None
    scale_rt, scale_selection, scale_train_dec, scale_val_dec, scale_eval_dec = best_scale
    scaled_result, scaled_rows = _select_runner(
        name=f"{name}__parent_direct_scaled_no_teacher",
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        parent_for_features=parent_for_features,
        train_dec=scale_train_dec,
        val_dec=scale_val_dec,
        eval_dec=scale_eval_dec,
        fee=fee,
        slip=slip,
        out_dir=variant_dir / "runners",
    )
    scaled_result["selected_parent_scale_runtime"] = dict(scale_rt.__dict__)
    scaled_result["scale_selection_metrics"] = scale_selection["metrics"]
    experiments.append(scaled_result)
    grid_rows.extend(scaled_rows)

    best = max(experiments, key=lambda e: float(e["selection_score"]))
    result = {
        "variant": name,
        "variant_cols": variant_cols,
        "feature_basis": feature_basis,
        "selected_legacy_cols": selected_legacy,
        "selected_sticky_cols": selected_sticky,
        "feature_count": int(len(feature_cols)),
        "legacy_feature_count": int(len(selected_legacy)),
        "sticky_feature_count": int(len(selected_sticky)),
        "best_by_selection": best["name"],
        "selection_score": float(best["selection_score"]),
        "selected_metrics": _compact_costs(best["metrics"]),
        "selected_validation_metrics": _compact_costs(best["validation_metrics"]),
        "experiments": experiments,
        "artifacts": {
            "parent": str(parent_path),
            "variant_dir": str(variant_dir),
        },
        "train_meta": train_meta,
    }
    pd.DataFrame(grid_rows).sort_values("score", ascending=False).to_csv(variant_dir / f"{name}_grid.csv", index=False)
    (variant_dir / f"{name}_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result, grid_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate Alpha4.3 legacy clean-regime block subsets under the same no-teacher/no-deep training loop.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--seed", type=int, default=4317)
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    p.add_argument("--feature-basis", choices=("alpha4_3", "alpha5"), default="alpha4_3")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    results: list[dict[str, Any]] = []
    for offset, name in enumerate(args.variants):
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name}; choices={sorted(VARIANTS)}")
        result, _ = _run_variant(
            name=name,
            variant_cols=VARIANTS[name],
            train_all=train_all,
            eval_df=eval_df,
            out_dir=args.out_dir,
            stride=int(args.stride),
            seed=int(args.seed) + offset,
            feature_basis=str(args.feature_basis),
        )
        results.append(result)
        print(json.dumps({"variant": name, "best": result["best_by_selection"], "metrics": result["selected_metrics"]}, ensure_ascii=False, default=_json_default), flush=True)

    summary = {
        "model_id": MODEL_ID,
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "split": {
            "train": "2025-01-01..2025-09-30",
            "selection": "2025-10-01..2025-12-31",
            "oos": "2026 fixed OOS",
        },
        "selection_policy": "original Alpha5 alpha2._score",
        "feature_basis": str(args.feature_basis),
        "changed_surface": "legacy clean_regime_2024_unsup_v4 subset only",
        "teacher_layer": "disabled",
        "deep_scout": "disabled",
        "variants": results,
    }
    report_path = args.out_dir / "alpha4_3_legacy_regime_block_ablation_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    rows = []
    for r in results:
        row = {
            "variant": r["variant"],
            "best_by_selection": r["best_by_selection"],
            "selection_score": r["selection_score"],
            "feature_count": r["feature_count"],
            "legacy_feature_count": r["legacy_feature_count"],
        }
        for cost in ("cost1", "cost2", "cost3"):
            for key, value in r["selected_metrics"][cost].items():
                row[f"{cost}_{key}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "alpha4_3_legacy_regime_block_ablation_results.csv", index=False)
    print(json.dumps({"report": str(report_path), "results_csv": str(args.out_dir / "alpha4_3_legacy_regime_block_ablation_results.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
