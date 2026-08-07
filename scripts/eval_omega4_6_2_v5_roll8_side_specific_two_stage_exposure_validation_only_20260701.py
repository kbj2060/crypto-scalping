#!/usr/bin/env python3
"""Validation-only exposure overlay for repaired Omega 4.6.2 v5 roll8 two-stage branch."""

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
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
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
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701.md"
VALIDATION_MDD_FLOOR = -17.50


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
    text = f"""# Omega 4.6.2 v5 Roll8 Two-Stage Exposure Validation-Only - 2026-07-01

## Method

This branch reuses the repaired buffered exposure grid, but removes OOS from selection. It selects only by validation gates and validation metrics, including a validation MDD floor of `{VALIDATION_MDD_FLOOR:.2f}%`. OOS is read out after selection and is not a filter, ordering key, or tie-breaker.

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
- Validation MDD floor: `{VALIDATION_MDD_FLOOR:.2f}%`
- OOS used in selection: `False`

## Artifacts

- Source buffered report: `{SOURCE_BUFFERED_REPORT}`
- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    base = load_module("omega462_two_stage_buffered_eval_for_validation_only", BASE_EXPOSURE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(SOURCE_BUFFERED_REPORT.read_text(encoding="utf-8"))
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    reference = source_report["reference_variant"]
    source_ranking = pd.read_csv(source_report["artifacts"]["ranking"])
    validation_only = source_ranking[
        source_ranking["validation_two_stage_exposure_gate_pass"].astype(bool)
        & (source_ranking["validation_mdd"].astype(float) >= VALIDATION_MDD_FLOOR)
    ].copy()
    if validation_only.empty:
        raise RuntimeError("no validation-gated exposure candidates")
    selected = validation_only.sort_values(
        ["validation_pnl", "validation_mdd", "validation_avg_hold_hours"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    selected["oos_used_in_selection"] = False
    selected["research_validation_only_gate_pass"] = True

    ref_val = pd.read_csv(reference_report["artifacts"]["selected_validation_ledger"])
    ref_oos = pd.read_csv(reference_report["artifacts"]["selected_oos_ledger"])
    selected_val = base.apply_exposure_overlay(
        ref_val,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_oos = base.apply_exposure_overlay(
        ref_oos,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    safe = str(selected["exposure_spec"]).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll8_two_stage_exposure_validation_only_ranking.csv"
    top20_path = OUT_DIR / "roll8_two_stage_exposure_validation_only_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    validation_only.to_csv(ranking_path, index=False)
    validation_only.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only_two_stage_exposure_overlay_with_validation_mdd_floor; oos_readout_after_selection",
        "selection_rule": f"require validation_two_stage_exposure_gate_pass and validation_mdd >= {VALIDATION_MDD_FLOOR:.2f}; sort by validation_pnl, validation_mdd, validation_avg_hold_hours; OOS is not used as filter, ordering key, or tie-breaker",
        "validation_mdd_floor": VALIDATION_MDD_FLOOR,
        "reference_report": str(REFERENCE_REPORT),
        "source_buffered_report": str(SOURCE_BUFFERED_REPORT),
        "reference_variant": reference,
        "selected_variant": selected,
        "top20": validation_only.head(20).to_dict(orient="records"),
        "status": "RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_VALIDATION_ONLY_PASS",
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "source_buffered_report": str(SOURCE_BUFFERED_REPORT),
            "ranking": str(ranking_path),
            "top20": str(top20_path),
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
