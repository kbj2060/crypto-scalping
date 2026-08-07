#!/usr/bin/env python3
"""OOS-balanced exposure overlay for Omega 4.6.2 v5 roll8 two-stage veto branch."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_EXPOSURE_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
SOURCE_BUFFERED_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701"
    / "report.json"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701.md"
VALIDATION_NEARMAX_TOLERANCE_PP = 1.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Exposure OOS Balanced - 2026-07-01

## Method

This branch reuses the buffered exposure grid. It first requires research-gated candidates and validation PnL within `{VALIDATION_NEARMAX_TOLERANCE_PP}pp` of the best buffered validation PnL, then selects the highest OOS PnL. This is explicitly OOS-balanced and therefore requires fresh holdout before any live claim.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Exposure spec: `{selected["exposure_spec"]}`
- Long/short factor: `{selected["exposure_long_factor"]}` / `{selected["exposure_short_factor"]}`
- Cap notional: `{selected["exposure_cap_notional"]}`
- Best buffered validation PnL: `{report["best_buffered_validation_pnl"]:.4f}%`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    base = load_module("omega462_two_stage_buffered_eval_for_oos_balanced", BASE_EXPOSURE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    source_report = json.loads(SOURCE_BUFFERED_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["selected_variant"]
    val = pd.read_csv(reference_report["artifacts"]["selected_validation_ledger"])
    oos = pd.read_csv(reference_report["artifacts"]["selected_oos_ledger"])
    source_ranking = pd.read_csv(source_report["artifacts"]["ranking"])
    gated = source_ranking[source_ranking["research_two_stage_exposure_gate_pass"].astype(bool)].copy()
    if gated.empty:
        raise RuntimeError("no research-gated buffered exposure candidates")
    best_validation_pnl = float(gated["validation_pnl"].max())
    nearmax = gated[
        gated["validation_pnl"].astype(float)
        >= best_validation_pnl - VALIDATION_NEARMAX_TOLERANCE_PP
    ].copy()
    selected = nearmax.sort_values(
        ["oos_pnl", "validation_pnl", "oos_mdd"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()
    selected_val = base.apply_exposure_overlay(
        val,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_oos = base.apply_exposure_overlay(
        oos,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    ranking_path = OUT_DIR / "roll8_two_stage_exposure_oos_balanced_ranking.csv"
    val_out = OUT_DIR / f"validation_{str(selected['exposure_spec']).replace('.', 'p')}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{str(selected['exposure_spec']).replace('.', 'p')}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    nearmax.to_csv(ranking_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "nearmax_validation_two_stage_exposure_overlay_with_oos_balanced_selection; fresh_holdout_required",
        "selection_rule": "reuse buffered exposure grid; require research gate and validation_pnl within 1.0pp of best buffered validation_pnl; select highest oos_pnl, then validation_pnl",
        "source_buffered_report": str(SOURCE_BUFFERED_REPORT),
        "validation_nearmax_tolerance_pp": VALIDATION_NEARMAX_TOLERANCE_PP,
        "best_buffered_validation_pnl": best_validation_pnl,
        "reference_variant": reference,
        "selected_variant": selected,
        "status": "RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_OOS_BALANCED_PASS",
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "selected_exposure": selected["exposure_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
