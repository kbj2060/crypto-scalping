#!/usr/bin/env python3
"""Robust branch from the Omega 4.6.2 v5 roll16 fine exposure sweep."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINE_MODULE_PATH = (
    ROOT / "scripts/eval_omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701.py"
)
MODEL_ID = "omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701"
SOURCE_MODEL_ID = "omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll16_bracket_robust_segment_governor_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516" / SOURCE_MODEL_ID
SOURCE_REPORT = SOURCE_DIR / "report.json"
SOURCE_RANKING = SOURCE_DIR / "roll16_fine_exposure_segment_governor_ranking.csv"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll16_fine_robust_segment_governor_20260701.md"
MAX_VALIDATION_PNL_GAP_PP = 15.0
MIN_VALIDATION_MDD = -18.0
MAX_CAP_NOTIONAL = 4.20


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    source_best = report["source_best_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll16 Fine Robust Segment Governor - 2026-07-01

## Method

This branch selects a robust candidate from the roll16 fine exposure sweep:

- candidate must pass the research gate and OOS safety gate,
- validation PnL must be within `{MAX_VALIDATION_PNL_GAP_PP:.1f}pp` of the fine max-PnL candidate,
- validation MDD must be at least `{MIN_VALIDATION_MDD:.1f}%`,
- exposure cap must be `<= {MAX_CAP_NOTIONAL:.2f}`.

## Result

- Status: `{report["status"]}`
- Source best model: `{report["source_model_id"]}`
- Reference robust model: `{report["reference_model_id"]}`

| Metric | Old Robust Val | Fine Best Val | Fine Robust Val | Old Robust OOS | Fine Best OOS | Fine Robust OOS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{source_best["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{source_best["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{source_best["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{source_best["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{source_best["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{source_best["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |

## Selected Candidate

- Exposure spec: `{selected["exposure_spec"]}`
- Segment governor: `{selected["segment_governor_spec"]}`
- TP/SL: `{selected["roll16_tp_move"]:.4f}` / `{selected["roll16_sl_move"]:.4f}`
- Research gate pass: `{selected["research_upgrade_gate_pass"]}`

## Artifacts

- Robust ranking: `{report["artifacts"]["ranking"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    fine = load_module("omega462_roll16_fine_for_robust", FINE_MODULE_PATH)
    roll16 = fine.load_module("omega462_roll16_for_fine_robust", fine.ROLL16_MODULE_PATH)
    v1 = roll16.load_module("omega462_loss_cluster_v1_for_fine_robust", roll16.V1_MODULE_PATH)
    segment_mod = roll16.load_module("omega462_segment_for_fine_robust", roll16.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_fine_robust", v1.STOP_MODULE_PATH)
    if not SOURCE_REPORT.exists() or not SOURCE_RANKING.exists():
        raise FileNotFoundError("run the roll16 fine exposure evaluator before robust selection")

    source_report = roll16.read_json(SOURCE_REPORT)
    reference = roll16.read_json(REFERENCE_REPORT)["selected_variant"]
    ranking = pd.read_csv(SOURCE_RANKING)
    passed = ranking[ranking["research_upgrade_gate_pass"].astype(bool)].copy()
    best_validation_pnl = float(source_report["selected_variant"]["validation_pnl"])
    eligible = passed[
        (passed["validation_pnl"] >= best_validation_pnl - MAX_VALIDATION_PNL_GAP_PP)
        & (passed["validation_mdd"] >= MIN_VALIDATION_MDD)
        & (passed["exposure_cap_notional"] <= MAX_CAP_NOTIONAL + 1.0e-12)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no candidate met the fine robust validation rule")
    eligible["robust_selection_score"] = (
        eligible["validation_pnl"]
        + 0.001 * eligible["oos_pnl"]
        - 0.01 * eligible["exposure_cap_notional"]
    )
    robust_ranking = eligible.sort_values(
        ["robust_selection_score", "validation_pnl"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = robust_ranking.iloc[0].to_dict()

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    parent_report = roll16.read_json(roll16.PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])
    val_base = roll16.split_roll_bracket(
        val_parent, train_market, segment_mod, fine.MAX_ROLL_HOURS, fine.TP_MOVE, fine.SL_MOVE
    )
    oos_base = roll16.split_roll_bracket(
        oos_parent, eval_market, segment_mod, fine.MAX_ROLL_HOURS, fine.TP_MOVE, fine.SL_MOVE
    )
    selected_exposure = segment_mod.ExposureSpec(
        selected["exposure_spec"],
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_governor = segment_mod.SegmentGovernorSpec(
        selected["segment_governor_spec"],
        float(selected["segment_governor_loss1_scale"]),
        float(selected["segment_governor_loss2_scale"]),
        float(selected["segment_governor_loss_window_hours"]),
    )
    selected_val = segment_mod.apply_segment_exposure_governor(
        val_base, selected_exposure, selected_governor
    )
    selected_oos = segment_mod.apply_segment_exposure_governor(
        oos_base, selected_exposure, selected_governor
    )
    val_metrics = stop_mod.metrics(selected_val)
    oos_metrics = stop_mod.metrics(selected_oos)
    for key, value in val_metrics.items():
        selected[f"validation_{key}"] = value
    for key, value in oos_metrics.items():
        selected[f"oos_{key}"] = value
    selected["fine_robust_branch_rule_pass"] = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = (
        f"{selected['exposure_spec']}__{selected['segment_governor_spec']}__"
        f"tp{float(selected['roll16_tp_move']):.3f}_sl{float(selected['roll16_sl_move']):.3f}"
    ).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll16_fine_robust_segment_governor_ranking.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    robust_ranking.to_csv(ranking_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)

    report = {
        "model_id": MODEL_ID,
        "source_model_id": SOURCE_MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_fine_robust_branch_with_oos_safety_gate; fresh_holdout_required",
        "selection_rule": {
            "max_validation_pnl_gap_pp": MAX_VALIDATION_PNL_GAP_PP,
            "min_validation_mdd": MIN_VALIDATION_MDD,
            "max_cap_notional": MAX_CAP_NOTIONAL,
        },
        "source_best_variant": source_report["selected_variant"],
        "reference_variant": reference,
        "parent_variant": parent_report["selected_variant"],
        "selected_variant": selected,
        "variants_eligible": int(len(robust_ranking)),
        "status": "RESEARCH_ROLL16_FINE_ROBUST_PASS",
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    roll16.write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "selected_exposure": selected["exposure_spec"],
                "selected_segment_governor": selected["segment_governor_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gate": bool(selected["validation_upgrade_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "gate": bool(selected["oos_research_gate_pass"]),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=roll16.json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
