#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha2_1_teacher_arch_ablation_20260514 as ab  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import _seq_tensor  # noqa: E402


MODEL_ID = "alpha2_1_teacher_arch_ablation_extra_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha2_1_teacher_arch_ablation_extra_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_extra_20260514_grid.csv"


def _run_variant(stack: dict[str, Any], variant: ab.ArchVariant) -> dict[str, Any]:
    buckets = stack["buckets"]
    if variant.hgb_meta:
        train_features, cols = ab._augment_hgb_meta(stack["train_features"], stack["train_dec"])
        val_features, _ = ab._augment_hgb_meta(stack["val_features"], stack["val_dec"])
        eval_features, _ = ab._augment_hgb_meta(stack["eval_features"], stack["eval_dec"])
    else:
        cols = list(stack["feature_cols"])
        train_features = stack["train_features"].reindex(columns=cols, fill_value=0.0)
        val_features = stack["val_features"].reindex(columns=cols, fill_value=0.0)
        eval_features = stack["eval_features"].reindex(columns=cols, fill_value=0.0)
    train_seq = _seq_tensor(train_features, np.arange(len(train_features), dtype=np.int64), cols)
    val_seq = _seq_tensor(val_features, np.arange(len(val_features), dtype=np.int64), cols)
    train_y_action = stack["train_dec"]["action"].astype(int).to_numpy(dtype=np.int64)
    val_y_action = stack["val_dec"]["action"].astype(int).to_numpy(dtype=np.int64)
    train_y_quality = pd.to_numeric(stack["train_dec"]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    val_y_quality = pd.to_numeric(stack["val_dec"]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    train_y_notional = ab._bucket_labels(stack["train_dec"], buckets)
    val_y_notional = ab._bucket_labels(stack["val_dec"], buckets)
    model, meta = ab._train_model(
        variant,
        train_seq,
        val_seq,
        train_y_action,
        val_y_action,
        train_y_quality,
        val_y_quality,
        train_y_notional,
        val_y_notional,
        n_buckets=len(buckets),
    )
    val_pred = ab._predict_model(model, val_features, cols, meta["norm"])
    eval_pred = ab._predict_model(model, eval_features, cols, meta["norm"])
    row = ab._eval_decisions(variant.name, stack["val_dec"], stack["eval_dec"], val_pred, eval_pred, buckets, variant, stack)
    row["train_meta"] = {k: v for k, v in meta.items() if k != "norm"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": asdict(variant),
            "state_dict": model.state_dict(),
            "feature_cols": cols,
            "train_meta": meta,
            "buckets": buckets,
        },
        OUT_DIR / f"{variant.name}.pt",
    )
    del train_seq, val_seq, model, val_pred, eval_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> int:
    print(f"[{MODEL_ID}] start", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = ab._load_stack()
    variants = [
        ab.ArchVariant("baseline_retrain_focal", "baseline", train_epochs=48),
        ab.ArchVariant("rope_task_attention_focal", "rope_task_attn", train_epochs=48),
        ab.ArchVariant("hgb_meta_task_attention_focal", "task_attn", hgb_meta=True, train_epochs=48),
    ]
    experiments = []
    for variant in variants:
        print(f"[{MODEL_ID}] variant {variant.name}", flush=True)
        row = _run_variant(stack, variant)
        experiments.append(row)
        print(
            f"[{MODEL_ID}] {variant.name} OOS cost1={row['metrics']['cost1']['pnl']:.2f} "
            f"mdd={row['metrics']['cost1']['mdd']:.2f} cost2={row['metrics']['cost2']['pnl']:.2f} "
            f"cost3={row['metrics']['cost3']['pnl']:.2f}",
            flush=True,
        )
    grid_rows = [
        {
            "name": exp["name"],
            "selection_score": exp["selection_score"],
            "score": exp["score"],
            "val_cost1_pnl": exp["val_metrics"]["cost1"]["pnl"],
            "val_cost1_mdd": exp["val_metrics"]["cost1"]["mdd"],
            "val_cost2_pnl": exp["val_metrics"]["cost2"]["pnl"],
            "val_cost3_pnl": exp["val_metrics"]["cost3"]["pnl"],
            "cost1_pnl": exp["metrics"]["cost1"]["pnl"],
            "cost1_mdd": exp["metrics"]["cost1"]["mdd"],
            "cost1_trades": exp["metrics"]["cost1"]["trades"],
            "cost2_pnl": exp["metrics"]["cost2"]["pnl"],
            "cost3_pnl": exp["metrics"]["cost3"]["pnl"],
        }
        for exp in experiments
    ]
    pd.DataFrame(grid_rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    selected = max(experiments, key=lambda x: float(x["selection_score"]))
    audit = {
        "status": "pass" if not stack["audit"].get("blocking") else "fail",
        "verdict": "iterate",
        "blocking": list(stack["audit"].get("blocking", [])),
        "warnings": list(stack["audit"].get("warnings", [])),
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection only",
        "selected": selected["name"],
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Lower-risk follow-up ablation after GRN variants underperformed: focal-only retrain, RoPE without GRN, HGB meta-state without GRN.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "out_dir": str(OUT_DIR)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "selected": selected["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
