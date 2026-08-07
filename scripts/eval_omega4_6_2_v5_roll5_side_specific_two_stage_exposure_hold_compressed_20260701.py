#!/usr/bin/env python3
"""5h hold-compressed two-stage exposure branch for Omega 4.6.2 v5."""

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
ROLL6_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll5_side_specific_two_stage_exposure_hold_compressed_20260701.md"
MAX_ROLL_HOURS = 5.0


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


def load_roll6() -> Any:
    roll6 = load_module("omega462_roll6_for_roll5_eval", ROLL6_PATH)
    roll6.MAX_ROLL_HOURS = MAX_ROLL_HOURS
    return roll6


def build_selected_ledgers(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    roll6 = load_roll6()
    return roll6.build_selected_ledgers(selected)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll5 Side-Specific Two-Stage Exposure Hold Compressed - 2026-07-01

## Method

This branch reuses the roll6 hold-compressed construction, but sets max roll hold to `5h` and searches a lower exposure grid. Selection is validation-primary with OOS as a safety gate.

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
- Max roll hold: `{selected["roll5_max_hours"]}h`
- Research gate pass: `{selected["research_roll5_hold_compressed_gate_pass"]}`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    roll6 = load_roll6()
    roll7 = roll6.load_module("omega462_roll7_for_roll5_eval", roll6.ROLL7_PATH)
    roll7.MAX_ROLL_HOURS = MAX_ROLL_HOURS
    mods = roll7.modules()
    helper = mods["helper"]
    exposure = mods["exposure"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))["selected_variant"]
    val_parent, oos_parent, train_market, eval_market = roll7.load_parent_and_markets(mods)
    val_pre = roll7.build_pre_exposure(val_parent, train_market, mods)
    oos_pre = roll7.build_pre_exposure(oos_parent, eval_market, mods)

    rows: list[dict[str, Any]] = []
    for long_factor in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        for short_factor in [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.15, 1.20]:
            for cap_notional in [2.80, 3.20, 3.60, 4.00, 4.40, 4.80, 5.00]:
                val_work = exposure.apply_exposure_overlay(val_pre, long_factor, short_factor, cap_notional)
                oos_work = exposure.apply_exposure_overlay(oos_pre, long_factor, short_factor, cap_notional)
                val_metrics = helper.metrics(val_work)
                oos_metrics = helper.metrics(oos_work)
                row = {
                    "exposure_spec": f"lf{long_factor:.3f}_sf{short_factor:.3f}_cap{cap_notional:.2f}",
                    "exposure_long_factor": float(long_factor),
                    "exposure_short_factor": float(short_factor),
                    "exposure_cap_notional": float(cap_notional),
                    "roll5_max_hours": MAX_ROLL_HOURS,
                    "roll5_long_tp_move": 0.0200,
                    "roll5_long_sl_move": 0.0300,
                    "roll5_short_tp_move": 0.0250,
                    "roll5_short_sl_move": 0.0385,
                    **roll7.flatten("validation", val_metrics),
                    **roll7.flatten("oos", oos_metrics),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) >= 100.0
                    and float(row["validation_mdd"]) >= -20.0
                    and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and float(row["validation_avg_hold_hours"]) < float(reference["validation_avg_hold_hours"])
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["validation_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                oos_gate = bool(
                    float(row["oos_pnl"]) >= 100.0
                    and float(row["oos_mdd"]) >= -20.0
                    and float(row["oos_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and float(row["oos_avg_hold_hours"]) < float(reference["oos_avg_hold_hours"])
                    and int(row["oos_overlap_count"]) == 0
                    and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["oos_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_roll5_hold_compressed_gate_pass"] = validation_gate
                row["oos_safety_gate_pass"] = oos_gate
                row["research_roll5_hold_compressed_gate_pass"] = validation_gate and oos_gate
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_roll5_hold_compressed_gate_pass",
            "validation_pnl",
            "oos_pnl",
            "validation_avg_hold_hours",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_val, selected_oos = build_selected_ledgers(selected)
    status = (
        "RESEARCH_ROLL5_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASS"
        if bool(selected["research_roll5_hold_compressed_gate_pass"])
        else "NO_ROLL5_TWO_STAGE_EXPOSURE_HOLD_COMPRESSED_PASSING_CANDIDATE"
    )
    safe = str(selected["exposure_spec"]).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll5_two_stage_exposure_hold_compressed_ranking.csv"
    top20_path = OUT_DIR / "roll5_two_stage_exposure_hold_compressed_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_primary_roll5_two_stage_veto_exposure_overlay_with_oos_safety_gate; fresh_holdout_required",
        "selection_rule": "regenerate 5h roll path with current two-stage veto rules; among research-gated exposure overlays, sort by validation_pnl, oos_pnl, validation_avg_hold",
        "reference_variant": reference,
        "selected_variant": selected,
        "top20": ranking.head(20).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(OUT_DIR),
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
                "status": status,
                "selected_exposure": selected["exposure_spec"],
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "gate": bool(selected["validation_roll5_hold_compressed_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "gate": bool(selected["oos_safety_gate_pass"]),
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
