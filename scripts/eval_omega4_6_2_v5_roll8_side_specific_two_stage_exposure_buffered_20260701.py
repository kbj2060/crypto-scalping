#!/usr/bin/env python3
"""Buffered exposure overlay for Omega 4.6.2 v5 roll8 two-stage veto branch."""

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
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_exposure_buffered_20260701.md"
VALIDATION_MDD_BUFFER_FLOOR = -19.50
LEVERAGE_CAP = 5.0
MAX_MARGIN_FRACTION = 1.0
EPS = 1.0e-12


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


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def apply_exposure_overlay(df: pd.DataFrame, long_factor: float, short_factor: float, cap_notional: float) -> pd.DataFrame:
    out = df.copy()
    active = out["notional"].astype(float) > EPS
    side = out["side"].astype(int)
    factor = side.map(lambda value: long_factor if value > 0 else short_factor).astype(float)
    base_notional = out["notional"].astype(float)
    capped_notional = np.minimum(base_notional * factor, cap_notional)
    capped_notional = pd.Series(capped_notional, index=out.index).where(active, 0.0)
    leverage = LEVERAGE_CAP
    margin = capped_notional / leverage
    spec = f"lf{long_factor:.3f}_sf{short_factor:.3f}_cap{cap_notional:.2f}"
    out["two_stage_exposure_spec"] = spec
    out["two_stage_exposure_long_factor"] = float(long_factor)
    out["two_stage_exposure_short_factor"] = float(short_factor)
    out["two_stage_exposure_cap_notional"] = float(cap_notional)
    out["two_stage_exposure_base_notional"] = base_notional
    out.loc[active, "notional"] = capped_notional[active]
    out.loc[active, "leverage"] = leverage
    out.loc[active, "margin_fraction"] = margin[active]
    for col in ["risk_notional", "exit_input_notional", "exit_input_exposure"]:
        if col in out.columns:
            out.loc[active, col] = capped_notional[active]
    for col in ["risk_leverage", "exit_input_leverage"]:
        if col in out.columns:
            out.loc[active, col] = leverage
    if "risk_margin_fraction" in out.columns:
        out.loc[active, "risk_margin_fraction"] = margin[active]
    out.loc[active, "trade_return"] = (
        out.loc[active, "net_per_notional"].astype(float) * capped_notional[active]
    )
    out.loc[active, "win"] = (out.loc[active, "trade_return"].astype(float) > 0.0).astype(int)
    return out


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Exposure Buffered - 2026-07-01

## Method

This branch starts from `{REFERENCE_MODEL_ID}` and applies a ledger-level exposure overlay to already selected entries/exits. It does not change hold time. Selection is validation-primary with a validation MDD buffer floor of `{VALIDATION_MDD_BUFFER_FLOOR}%`; OOS is a safety gate only.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Exposure spec: `{selected["exposure_spec"]}`
- Long/short factor: `{selected["exposure_long_factor"]}` / `{selected["exposure_short_factor"]}`
- Cap notional: `{selected["exposure_cap_notional"]}`
- Research gate pass: `{selected["research_two_stage_exposure_gate_pass"]}`

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
    helper = load_module("omega462_two_stage_exposure_helper", AUDIT_HELPER_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["selected_variant"]
    val = pd.read_csv(reference_report["artifacts"]["selected_validation_ledger"])
    oos = pd.read_csv(reference_report["artifacts"]["selected_oos_ledger"])

    rows: list[dict[str, Any]] = []
    for long_factor in [0.90, 0.95, 1.00, 1.03, 1.05, 1.08, 1.10, 1.12, 1.15]:
        for short_factor in [0.90, 0.95, 1.00, 1.03, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.25]:
            for cap_notional in [4.20, 4.40, 4.60, 4.80, 5.00]:
                cap = min(float(cap_notional), LEVERAGE_CAP * MAX_MARGIN_FRACTION)
                val_work = apply_exposure_overlay(val, long_factor, short_factor, cap)
                oos_work = apply_exposure_overlay(oos, long_factor, short_factor, cap)
                val_metrics = helper.metrics(val_work)
                oos_metrics = helper.metrics(oos_work)
                row = {
                    "exposure_spec": f"lf{long_factor:.3f}_sf{short_factor:.3f}_cap{cap:.2f}",
                    "exposure_long_factor": float(long_factor),
                    "exposure_short_factor": float(short_factor),
                    "exposure_cap_notional": float(cap),
                    "exposure_leverage_cap": LEVERAGE_CAP,
                    "exposure_max_margin_fraction": MAX_MARGIN_FRACTION,
                    "validation_mdd_buffer_floor": VALIDATION_MDD_BUFFER_FLOOR,
                    **flatten("validation", val_metrics),
                    **flatten("oos", oos_metrics),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) > float(reference["validation_pnl"])
                    and float(row["validation_mdd"]) >= VALIDATION_MDD_BUFFER_FLOOR
                    and float(row["validation_max_hold_hours"]) <= float(reference["validation_max_hold_hours"]) + 1.0e-9
                    and float(row["validation_avg_hold_hours"]) <= float(reference["validation_avg_hold_hours"]) + 1.0e-9
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["validation_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                oos_gate = bool(
                    float(row["oos_pnl"]) > float(reference["oos_pnl"])
                    and float(row["oos_mdd"]) >= -20.0
                    and float(row["oos_max_hold_hours"]) <= float(reference["oos_max_hold_hours"]) + 1.0e-9
                    and float(row["oos_avg_hold_hours"]) <= float(reference["oos_avg_hold_hours"]) + 1.0e-9
                    and int(row["oos_overlap_count"]) == 0
                    and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["oos_max_margin_fraction"]) <= 1.0 + 1.0e-9
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_two_stage_exposure_gate_pass"] = validation_gate
                row["oos_safety_gate_pass"] = oos_gate
                row["research_two_stage_exposure_gate_pass"] = validation_gate and oos_gate
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_two_stage_exposure_gate_pass",
            "validation_pnl",
            "oos_pnl",
            "validation_mdd",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("no two-stage exposure variants evaluated")
    selected = ranking.iloc[0].to_dict()
    selected_val = apply_exposure_overlay(
        val,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    selected_oos = apply_exposure_overlay(
        oos,
        float(selected["exposure_long_factor"]),
        float(selected["exposure_short_factor"]),
        float(selected["exposure_cap_notional"]),
    )
    status = (
        "RESEARCH_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASS"
        if bool(selected["research_two_stage_exposure_gate_pass"])
        else "NO_ROLL8_TWO_STAGE_EXPOSURE_BUFFERED_PASSING_CANDIDATE"
    )
    safe = str(selected["exposure_spec"]).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll8_two_stage_exposure_buffered_ranking.csv"
    top20_path = OUT_DIR / "roll8_two_stage_exposure_buffered_top20.csv"
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
        "selection_scope": "validation_primary_two_stage_exposure_overlay_with_validation_mdd_buffer_and_oos_safety_gate; fresh_holdout_required",
        "selection_rule": "among research-gated exposure overlays, sort by validation_pnl, oos_pnl, validation_mdd; OOS is a safety/tie-break key only after validation gate and MDD buffer",
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
                    "max_notional": selected["validation_max_notional"],
                    "gate": bool(selected["validation_two_stage_exposure_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "max_notional": selected["oos_max_notional"],
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
