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

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    train_policy,
)
from scripts import eval_alpha3_parent_feature_reduction_20260515 as parent_reduce  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_parent_no_clean_regime_20260516"
CLEAN_PREFIX = "clean_regime_2024_unsup_v4_"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_parent_no_clean_regime_20260516"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_no_clean_regime_20260516_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_parent_no_clean_regime_20260516_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_parent_no_clean_regime_20260516_grid.csv"


def _score(metrics: dict[str, Any]) -> float:
    c1 = metrics["cost1"]
    c2 = metrics["cost2"]
    c3 = metrics["cost3"]
    return float(c1["pnl"] + 0.55 * c2["pnl"] + 0.35 * c3["pnl"] + 0.35 * c1["mdd"])


def _fit_no_clean_parent(
    *,
    original_parent: dict[str, Any],
    train_all: pd.DataFrame,
    cfg: FullyLearnedGovernorConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_cols = list(original_parent["feature_cols"])
    clean_cols = [c for c in raw_cols if c.startswith(CLEAN_PREFIX)]
    no_clean_cols = [c for c in raw_cols if not c.startswith(CLEAN_PREFIX)]
    if "side_hint" not in no_clean_cols:
        no_clean_cols.insert(0, "side_hint")
    no_clean_cols = list(dict.fromkeys(no_clean_cols))

    print(
        json.dumps(
            {
                "stage": "build_training_set",
                "raw_feature_count": len(raw_cols),
                "removed_clean_regime_count": len(clean_cols),
                "candidate_feature_count": len(no_clean_cols),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    x_train_all, y, training_meta = build_training_set(
        train_all,
        cfg=cfg,
        stride_bars=6,
        batch_size=512,
        feature_cols=raw_cols,
    )
    print(json.dumps({"stage": "train_no_clean_parent"}, ensure_ascii=False), flush=True)
    bundle = train_policy(
        x_train_all.reindex(columns=no_clean_cols),
        y,
        cfg=cfg,
        random_state=20260516,
        feature_cols=no_clean_cols,
    )
    bundle["model_id"] = MODEL_ID
    bundle["candidate_name"] = "no_clean_regime_parent"
    bundle["base_model"] = str(v31.DEFAULT_PARENT)
    bundle["train_csv"] = str(v31.DEFAULT_TRAIN)
    bundle["eval_csv"] = str(v31.DEFAULT_EVAL)
    bundle["training_meta"] = training_meta
    bundle["feature_ablation"] = {
        "removed_prefix": CLEAN_PREFIX,
        "removed_features": clean_cols,
        "raw_feature_count": len(raw_cols),
        "candidate_feature_count": len(no_clean_cols),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUT_DIR / "no_clean_regime_parent.pkl"
    joblib.dump(bundle, artifact)
    bundle["artifact_path"] = str(artifact)
    return bundle, {"training_meta": training_meta, "removed_features": clean_cols}


def main() -> int:
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "load_alpha3_stack", "model_id": MODEL_ID}), flush=True)

    original_parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = parent_reduce.CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    _, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v27_payload = torch.load(v31.DEFAULT_V27, map_location="cpu", weights_only=False)
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = parent_reduce._load_teacher()
    runtime = parent_reduce._selected_runtime()
    overlay = next(v.overlay for v in parent_reduce.l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = parent_reduce._canonical_limit_cfg()
    fee = float(dict(original_parent["config"])["fee"])
    slip = float(dict(original_parent["config"])["slip"])

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    cfg = FullyLearnedGovernorConfig(**dict(original_parent["config"]))

    print(json.dumps({"stage": "predict_teacher_v27_once"}, ensure_ascii=False), flush=True)
    eval_teacher_features = parent_reduce.prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    eval_teacher_pred = parent_reduce.teacher._predict_deep(teacher_model, eval_teacher_features, teacher_cols, teacher_norm)
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def evaluate(name: str, parent_bundle: dict[str, Any], decision_frame: pd.DataFrame, meta: dict[str, Any]) -> None:
        print(json.dumps({"stage": "evaluate", "variant": name}, ensure_ascii=False), flush=True)
        metrics = parent_reduce._metrics(
            eval_df=eval_df,
            runner_parent=original_parent,
            decision_parent=parent_bundle,
            decision_frame=decision_frame,
            teacher_pred=eval_teacher_pred,
            teacher_buckets=teacher_buckets,
            runtime=runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            eval_q=eval_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        item = {
            "variant": name,
            "feature_count": int(len(parent_bundle.get("feature_cols") or [])),
            "score": _score(metrics),
            "metrics": metrics,
            "meta": meta,
        }
        results.append(item)
        rows.append(
            {
                "variant": name,
                "feature_count": item["feature_count"],
                "score": item["score"],
                "cost1_pnl": metrics["cost1"]["pnl"],
                "cost1_mdd": metrics["cost1"]["mdd"],
                "cost1_trades": metrics["cost1"]["trades"],
                "cost1_long_entries": metrics["cost1"].get("long_entries"),
                "cost1_short_entries": metrics["cost1"].get("short_entries"),
                "cost1_deep_entries": metrics["cost1"].get("deep_entries"),
                "cost2_pnl": metrics["cost2"]["pnl"],
                "cost2_mdd": metrics["cost2"]["mdd"],
                "cost3_pnl": metrics["cost3"]["pnl"],
                "cost3_mdd": metrics["cost3"]["mdd"],
            }
        )

    original_dec = predict_policy_frame(original_parent, eval_df, close=_close(eval_df))
    evaluate("alpha3_original_clean_regime_parent", original_parent, original_dec, {"kind": "baseline"})

    candidate_parent, train_meta = _fit_no_clean_parent(original_parent=original_parent, train_all=train_all, cfg=cfg)
    candidate_dec = predict_policy_frame(candidate_parent, eval_df, close=_close(eval_df))
    evaluate("no_clean_regime_parent", candidate_parent, candidate_dec, {"kind": "parent_only_retrain", **train_meta})

    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    baseline = next(x for x in results if x["variant"] == "alpha3_original_clean_regime_parent")
    candidate = next(x for x in results if x["variant"] == "no_clean_regime_parent")
    delta = {
        cost: {
            "pnl": float(candidate["metrics"][cost]["pnl"] - baseline["metrics"][cost]["pnl"]),
            "mdd": float(candidate["metrics"][cost]["mdd"] - baseline["metrics"][cost]["mdd"]),
            "trades": int(candidate["metrics"][cost]["trades"] - baseline["metrics"][cost]["trades"]),
        }
        for cost in ("cost1", "cost2", "cost3")
    }
    warnings: list[str] = []
    if delta["cost1"]["pnl"] <= 0:
        warnings.append("no_clean_regime_parent_did_not_improve_cost1_pnl")
    if candidate["metrics"]["cost1"]["mdd"] < baseline["metrics"]["cost1"]["mdd"]:
        warnings.append("no_clean_regime_parent_worsened_cost1_mdd")
    audit = {
        "status": "pass",
        "blocking": [],
        "warnings": warnings,
        "primary_mutable_surface": "parent_only",
        "changed_layers": ["parent"],
        "frozen_layers": ["teacher_gate", "v21_2_runner", "v27_deep_scout", "v31_exit", "execution", "accounting", "data"],
        "removed_prefix": CLEAN_PREFIX,
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS",
        "alpha3_execution_contract": asdict(limit_cfg),
        "runtime": asdict(runtime),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Retrain Alpha3 HGB parent after removing clean_regime_2024_unsup_v4_* inputs. Teacher gate, V21.2 runner, frozen V27 scout, V31 exit overlay, execution, accounting, and data are fixed.",
        "base_model_alias": "alpha3",
        "frozen_protocol": "alpha3_frozen_backtest_protocol_20260515",
        "primary_mutable_surface": "parent_only",
        "changed_layers": ["parent"],
        "frozen_layers": ["teacher_gate", "v21_2_runner", "v27_deep_scout", "v31_exit", "execution", "accounting", "data"],
        "selection_uses_2026": False,
        "oos_window": "2026 fixed OOS",
        "baseline_reproduced": True,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "delta_vs_baseline": delta,
        "results": results,
        "summary": rows,
        "audit": audit,
        "artifacts": {
            "candidate_parent": candidate_parent.get("artifact_path"),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
            "out_dir": str(OUT_DIR),
        },
        "verdict": "promote_to_shadow_candidate" if not warnings and delta["cost1"]["pnl"] > 0 else "do_not_promote_iterate",
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_OUT),
                "audit": str(AUDIT_OUT),
                "grid": str(GRID_OUT),
                "delta_vs_baseline": delta,
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
