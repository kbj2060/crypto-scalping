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

from ensemble.fully_learned_governor_policy import build_training_set, predict_policy_frame, prepare_features
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1
from scripts import eval_hf_v13_v31_parent_swap_v40 as ps
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig

MODEL_ID = "alpha1_parent_only_swap_20260513"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_parent_only_swap_20260513"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha1_parent_only_swap_20260513_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/alpha1_parent_only_swap_20260513_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/alpha1_parent_only_swap_20260513_grid.csv"


PARENT_VARIANTS: dict[str, dict[str, Any]] = {
    "alpha1_original_parent": {
        "kind": "raw",
        "bundle": DEFAULT_PARENT,
        "report": ROOT / "data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_summary.json",
    },
    "v40_6_hgb_pls_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512",
        "nan_missing": False,
    },
    "v40_7_lgbm_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_v40_7_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_v40_7_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_v40_7_20260512",
        "nan_missing": False,
    },
    "v40_8_lgbm_native_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_native_v40_8_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_native_v40_8_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_native_v40_8_20260512",
        "nan_missing": False,
    },
    "v40_9_lgbm_native_quant_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512",
        "nan_missing": True,
    },
    "v40_10_lgbm_tradefloor_parent": {
        "kind": "encoded",
        "bundle": ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_tradefloor_v40_10_20260512/target_aware_full_bundle.pkl",
        "report": ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_tradefloor_v40_10_20260512_summary.json",
        "cache_consumer": "hf_v13_tree_vs_foundation_target_aware_lgbm_native_quant_v40_9_20260512",
        "nan_missing": True,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alpha1 parent-only swap: replace parent decisions, keep V21.2/V27/V31 alpha1 contract fixed.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] loading shared alpha1 artifacts", flush=True)
    original_parent = ps._load_bundle(DEFAULT_PARENT)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = alpha1.v31._load_v27(args.v27_model)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    cfg = ps._parent_cfg()
    print(f"[{MODEL_ID}] preparing train-only projection labels and V27 utilities", flush=True)
    x_train_base, y, training_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx) != len(x_train_base):
        raise RuntimeError(f"train_idx/x mismatch: {len(train_idx)} vs {len(x_train_base)}")
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    proj_targets = ps._projection_targets(y)
    train_feat_zero = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat_zero = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat_zero = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    train_feat_nan = ps._preserve_source_missing_as_nan(train_feat_zero, train_df, feature_cols)
    val_feat_nan = ps._preserve_source_missing_as_nan(val_feat_zero, val_df, feature_cols)
    eval_feat_nan = ps._preserve_source_missing_as_nan(eval_feat_zero, eval_df, feature_cols)
    val_q = alpha1.v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = alpha1.v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name, variant in PARENT_VARIANTS.items():
        print(f"[{MODEL_ID}] evaluating parent={name}", flush=True)
        parent_bundle = ps._load_bundle(Path(variant["bundle"]))
        if variant["kind"] == "raw":
            val_frame = val_df.reset_index(drop=True).copy()
            eval_frame = eval_df.reset_index(drop=True).copy()
            parent_meta = {"kind": "raw"}
        else:
            with Path(variant["report"]).open(encoding="utf-8") as f:
                parent_report = json.load(f)
            val_frame, eval_frame, parent_meta = ps._build_encoded_frames(
                variant_name=name,
                variant=variant,
                bundle=parent_bundle,
                report=parent_report,
                args=args,
                train_df=train_df,
                val_df=val_df,
                eval_df=eval_df,
                train_feat_zero=train_feat_zero,
                val_feat_zero=val_feat_zero,
                eval_feat_zero=eval_feat_zero,
                train_feat_nan=train_feat_nan,
                val_feat_nan=val_feat_nan,
                eval_feat_nan=eval_feat_nan,
                train_idx=train_idx,
                val_idx=val_idx,
                eval_idx=eval_idx,
                proj_targets=proj_targets,
            )
        val_dec = predict_policy_frame(parent_bundle, val_frame, close=_close(val_frame))
        eval_dec = predict_policy_frame(parent_bundle, eval_frame, close=_close(eval_frame))
        base = dict(parent_bundle.get("config", {}))
        fee = float(base.get("fee", alpha1.ALPHA1_CFG.base_tp * 0.0 + 0.0005))
        slip = float(base.get("slip", 0.0002))
        val_metrics = {
            f"cost{mult}": alpha1.backtest_alpha1(
                val_frame,
                original_parent,
                jackpot_model,
                add_cfg,
                val_q,
                fee=fee,
                slip=slip,
                cost_mult=float(mult),
                decisions=val_dec,
            )
            for mult in (1, 2, 3)
        }
        metrics = {
            f"cost{mult}": alpha1.backtest_alpha1(
                eval_frame,
                original_parent,
                jackpot_model,
                add_cfg,
                eval_q,
                fee=fee,
                slip=slip,
                cost_mult=float(mult),
                decisions=eval_dec,
            )
            for mult in (1, 2, 3)
        }
        feature_audit_cols = [c for c in list(parent_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
        feature_audit = _audit_contract(train_all, eval_df, feature_audit_cols)
        score = alpha1._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])
        rows.append(
            {
                "variant": name,
                "score": score,
                "val_pnl": val_metrics["cost1"]["pnl"],
                "val_mdd": val_metrics["cost1"]["mdd"],
                "val_c2_pnl": val_metrics["cost2"]["pnl"],
                "val_c3_pnl": val_metrics["cost3"]["pnl"],
                "cost1_pnl": metrics["cost1"]["pnl"],
                "cost1_mdd": metrics["cost1"]["mdd"],
                "cost1_trades": metrics["cost1"]["trades"],
                "cost1_deep_entries": metrics["cost1"].get("deep_entries", 0),
                "cost2_pnl": metrics["cost2"]["pnl"],
                "cost3_pnl": metrics["cost3"]["pnl"],
            }
        )
        results[name] = {
            "parent_model": str(variant["bundle"]),
            "parent_meta": parent_meta,
            "validation": val_metrics,
            "metrics": metrics,
            "feature_audit": feature_audit,
        }
        (args.out_dir / f"{name}_manifest.json").write_text(
            json.dumps({"model_id": MODEL_ID, "variant": name, "parent_model": str(variant["bundle"]), "metrics": metrics}, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    grid = pd.DataFrame(rows).sort_values("cost1_pnl", ascending=False)
    grid.to_csv(args.grid_out, index=False)
    best = grid.iloc[0].to_dict() if len(grid) else {}
    blocking: list[str] = []
    warnings: list[str] = []
    for name, item in results.items():
        fa = item["feature_audit"]
        blocking.extend([f"{name}:{x}" for x in fa.get("blocking", [])])
        warnings.extend([f"{name}:{x}" for x in fa.get("warnings", [])])
        if item["metrics"]["cost2"]["pnl"] <= 0.0:
            warnings.append(f"{name}:cost2_not_survived")
        if item["metrics"]["cost3"]["pnl"] <= 0.0:
            warnings.append(f"{name}:cost3_not_survived")
    if best.get("variant") == "alpha1_original_parent":
        warnings.append("no_parent_swap_beat_alpha1_original_parent")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best.get("variant") != "alpha1_original_parent" and float(best.get("cost1_pnl", 0.0)) > alpha1.ALPHA1_BASELINE["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS",
        "alpha1_contract_fixed": True,
        "parent_only_swap": True,
        "deep_notional": float(alpha1.ALPHA1_CFG.notional),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Parent-only swap under current alpha1 contract. V21.2, V27, V31 exit, deep notional=2.0, and accounting are fixed; only parent decisions are replaced.",
        "training_meta": training_meta,
        "summary": rows,
        "best_by_cost1": best,
        "variants": results,
        "audit": audit,
        "artifacts": {"out_dir": str(args.out_dir), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out)},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "best_by_cost1": best, "audit": audit}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
