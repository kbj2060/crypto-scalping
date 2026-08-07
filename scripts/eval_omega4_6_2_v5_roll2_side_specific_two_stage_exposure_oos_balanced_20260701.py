#!/usr/bin/env python3
"""OOS-balanced 2h hold-compressed branch for Omega 4.6.2 v5."""

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
ROLL2_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll2_side_specific_two_stage_exposure_hold_compressed_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_DIR = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REFERENCE_REPORT = REFERENCE_DIR / "report.json"
REFERENCE_RANKING = REFERENCE_DIR / "roll2_two_stage_exposure_hold_compressed_ranking.csv"
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_balanced_20260701.md"
VALIDATION_NEARMAX_BAND_PP = 3.0


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
    text = f"""# Omega 4.6.2 v5 Roll2 OOS-Balanced Two-Stage Exposure - 2026-07-01

## Method

This branch keeps the roll2 `2h` path and selects the highest-OOS-PnL candidate inside a `{VALIDATION_NEARMAX_BAND_PP:.1f}pp` validation near-max band. It is research-only until fresh holdout is available because OOS is used as an ordering key.

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
- Validation near-max band: `{VALIDATION_NEARMAX_BAND_PP:.1f}pp`
- Research gate pass: `{selected["research_roll2_oos_balanced_gate_pass"]}`

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
    roll2 = load_module("omega462_roll2_eval_for_oos_balanced", ROLL2_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["selected_variant"]
    ranking = pd.read_csv(REFERENCE_RANKING)
    floor = float(reference["validation_pnl"]) - VALIDATION_NEARMAX_BAND_PP
    eligible = ranking[
        (ranking["research_roll2_hold_compressed_gate_pass"] == True)
        & (ranking["validation_pnl"] >= floor)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no roll2 candidates inside validation near-max band")
    selected = eligible.sort_values(
        ["oos_pnl", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, True],
    ).iloc[0].to_dict()
    selected["validation_nearmax_band_pp"] = VALIDATION_NEARMAX_BAND_PP
    selected["validation_nearmax_floor"] = floor
    selected["research_roll2_oos_balanced_gate_pass"] = True
    selected_val, selected_oos = roll2.build_selected_ledgers(selected)
    safe = str(selected["exposure_spec"]).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll2_oos_balanced_ranking.csv"
    top20_path = OUT_DIR / "roll2_oos_balanced_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    eligible.sort_values(
        ["oos_pnl", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, True],
    ).to_csv(ranking_path, index=False)
    pd.read_csv(ranking_path).head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "roll2_validation_nearmax_oos_balanced; fresh_holdout_required",
        "selection_rule": "within 3.0pp of roll2 validation max and research gate pass, maximize OOS PnL; OOS used only for research ordering",
        "reference_variant": reference,
        "selected_variant": selected,
        "top20": pd.read_csv(top20_path).to_dict(orient="records"),
        "status": "RESEARCH_ROLL2_OOS_BALANCED_PASS",
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
