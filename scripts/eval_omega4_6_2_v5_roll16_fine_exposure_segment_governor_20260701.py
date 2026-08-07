#!/usr/bin/env python3
"""Fine exposure sweep for Omega 4.6.2 v5 roll16 bracket segment governor."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLL16_MODULE_PATH = (
    ROOT / "scripts/eval_omega4_6_2_v5_roll16_bracket_segment_governor_20260701.py"
)
MODEL_ID = "omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll16_bracket_segment_governor_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll16_fine_exposure_segment_governor_20260701.md"
MAX_ROLL_HOURS = 16.0
TP_MOVE = 0.045
SL_MOVE = 0.045


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll16 Fine Exposure Segment Governor - 2026-07-01

## Method

This branch keeps the roll16 TP/SL bracket fixed at `4.5%/4.5%` and fine-tunes long/short exposure around the prior roll16 winner. Selection is validation-primary with an OOS safety gate.

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
- Segment governor: `{selected["segment_governor_spec"]}`
- TP/SL: `{selected["roll16_tp_move"]:.4f}` / `{selected["roll16_sl_move"]:.4f}`
- Research gate pass: `{selected["research_upgrade_gate_pass"]}`

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
    roll16 = load_module("omega462_roll16_for_fine", ROLL16_MODULE_PATH)
    v1 = roll16.load_module("omega462_loss_cluster_v1_for_roll16_fine", roll16.V1_MODULE_PATH)
    segment_mod = roll16.load_module("omega462_segment_for_roll16_fine", roll16.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll16_fine", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference = roll16.read_json(REFERENCE_REPORT)["selected_variant"]
    parent_report = roll16.read_json(roll16.PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])
    val_base = roll16.split_roll_bracket(
        val_parent, train_market, segment_mod, MAX_ROLL_HOURS, TP_MOVE, SL_MOVE
    )
    oos_base = roll16.split_roll_bracket(
        oos_parent, eval_market, segment_mod, MAX_ROLL_HOURS, TP_MOVE, SL_MOVE
    )

    governors = [
        segment_mod.SegmentGovernorSpec("none", 1.00, 1.00, 0.0),
        segment_mod.SegmentGovernorSpec("loss1_95_win12", 0.95, 1.00, 12.0),
        segment_mod.SegmentGovernorSpec("loss1_90_win12", 0.90, 1.00, 12.0),
        segment_mod.SegmentGovernorSpec("streak85_60_win12", 0.85, 0.60, 12.0),
        segment_mod.SegmentGovernorSpec("streak90_70_win12", 0.90, 0.70, 12.0),
    ]
    rows: list[dict[str, Any]] = []
    for long_factor in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]:
        for short_factor in [0.95, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
            for cap_notional in [4.00, 4.10, 4.20, 4.30, 4.40]:
                exposure = segment_mod.ExposureSpec(
                    f"lf{long_factor:.2f}_sf{short_factor:.2f}_cap{cap_notional:.2f}",
                    long_factor,
                    short_factor,
                    cap_notional,
                )
                for governor in governors:
                    val_work = segment_mod.apply_segment_exposure_governor(
                        val_base, exposure, governor
                    )
                    oos_work = segment_mod.apply_segment_exposure_governor(
                        oos_base, exposure, governor
                    )
                    row = {
                        "exposure_spec": exposure.name,
                        "segment_governor_spec": governor.name,
                        "roll16_max_hours": MAX_ROLL_HOURS,
                        "roll16_tp_move": TP_MOVE,
                        "roll16_sl_move": SL_MOVE,
                        **{f"exposure_{k}": value for k, value in asdict(exposure).items()},
                        **{f"segment_governor_{k}": value for k, value in asdict(governor).items()},
                        **flatten("validation", stop_mod.metrics(val_work)),
                        **flatten("oos", stop_mod.metrics(oos_work)),
                    }
                    validation_gate = bool(
                        float(row["validation_pnl"]) > float(reference["validation_pnl"])
                        and float(row["validation_mdd"]) >= -20.0
                        and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                        and int(row["validation_overlap_count"]) == 0
                        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
                        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                    )
                    oos_gate = bool(
                        float(row["oos_pnl"]) > float(reference["oos_pnl"])
                        and float(row["oos_mdd"]) >= -20.0
                        and float(row["oos_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                        and int(row["oos_overlap_count"]) == 0
                        and float(row["oos_max_leverage"]) <= 5.0 + 1.0e-9
                        and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                        and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                        and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                    )
                    row["validation_upgrade_gate_pass"] = validation_gate
                    row["oos_research_gate_pass"] = oos_gate
                    row["research_upgrade_gate_pass"] = validation_gate and oos_gate
                    row["selection_score"] = (
                        float(row["validation_pnl"])
                        - 0.01 * float(row["exposure_cap_notional"])
                        + 0.001 * (float(row["oos_pnl"]) if oos_gate else 0.0)
                    )
                    rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["research_upgrade_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
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
    status = (
        "RESEARCH_ROLL16_FINE_EXPOSURE_UPGRADE_PASS"
        if bool(selected["research_upgrade_gate_pass"])
        else "NO_ROLL16_FINE_EXPOSURE_UPGRADE_PASSING_CANDIDATE"
    )
    safe = (
        f"{selected['exposure_spec']}__{selected['segment_governor_spec']}__"
        f"tp{float(selected['roll16_tp_move']):.3f}_sl{float(selected['roll16_sl_move']):.3f}"
    ).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll16_fine_exposure_segment_governor_ranking.csv"
    top20_path = OUT_DIR / "roll16_fine_exposure_segment_governor_top20.csv"
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
        "selection_scope": "validation_primary_with_oos_safety_gate; fresh_holdout_required",
        "reference_variant": reference,
        "parent_variant": parent_report["selected_variant"],
        "variants_evaluated": int(len(ranking)),
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
    roll16.write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
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
