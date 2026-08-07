#!/usr/bin/env python3
"""10h bracket day-trade branch for Omega 4.6.2 v5."""

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
SOURCE_MODULE_PATH = (
    ROOT / "scripts/eval_omega4_6_2_v5_roll16_bracket_segment_governor_20260701.py"
)
MODEL_ID = "omega4_6_2_v5_roll10_bracket_daytrade_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll12_fine_exposure_daytrade_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll10_bracket_daytrade_20260701.md"
MAX_ROLL_HOURS = 10.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_roll10(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {col: col.replace("roll16_", "roll10_") for col in out.columns if col.startswith("roll16_")}
    out = out.rename(columns=rename)
    out["reason"] = out["reason"].astype(str).str.replace("roll16_", "roll10_", regex=False)
    return out


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
    text = f"""# Omega 4.6.2 v5 Roll10 Bracket Daytrade - 2026-07-01

## Method

This branch starts from the v5 parent and splits positions into `<=10h` path-causal segments. It is selected as a middle ground between the 12h daytrade branch and the non-promoted 8h probe.

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
- Segment governor: `{selected["segment_governor_spec"]}`
- TP/SL: `{selected["roll10_tp_move"]:.4f}` / `{selected["roll10_sl_move"]:.4f}`
- Research gate pass: `{selected["research_daytrade_gate_pass"]}`

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
    source = load_module("omega462_roll16_source_for_roll10", SOURCE_MODULE_PATH)
    v1 = source.load_module("omega462_loss_cluster_v1_for_roll10", source.V1_MODULE_PATH)
    segment_mod = source.load_module("omega462_segment_for_roll10", source.SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_roll10", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference = source.read_json(REFERENCE_REPORT)["selected_variant"]
    parent_report = source.read_json(source.PARENT_REPORT)
    val_parent = pd.read_csv(parent_report["artifacts"]["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_report["artifacts"]["selected_oos_ledger"])

    rows: list[dict[str, Any]] = []
    exposure_specs = [
        segment_mod.ExposureSpec(f"lf{lf:.2f}_sf{sf:.2f}_cap{cap:.2f}", lf, sf, cap)
        for lf in [0.60, 0.70, 0.80, 0.90, 1.00]
        for sf in [0.90, 0.95, 1.00, 1.02, 1.04, 1.06, 1.08]
        for cap in [3.60, 4.00, 4.20, 4.30]
    ]
    governors = [
        segment_mod.SegmentGovernorSpec("none", 1.00, 1.00, 0.0),
        segment_mod.SegmentGovernorSpec("loss1_90_win10", 0.90, 1.00, 10.0),
        segment_mod.SegmentGovernorSpec("streak85_60_win10", 0.85, 0.60, 10.0),
    ]
    bracket_specs = [(0.025, 0.035), (0.030, 0.040), (0.035, 0.045), (0.040, 0.045), (0.045, 0.045)]
    for tp_move, sl_move in bracket_specs:
        val_base = normalize_roll10(
            source.split_roll_bracket(
                val_parent, train_market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move
            )
        )
        oos_base = normalize_roll10(
            source.split_roll_bracket(
                oos_parent, eval_market, segment_mod, MAX_ROLL_HOURS, tp_move, sl_move
            )
        )
        for exposure in exposure_specs:
            for governor in governors:
                val_work = segment_mod.apply_segment_exposure_governor(val_base, exposure, governor)
                oos_work = segment_mod.apply_segment_exposure_governor(oos_base, exposure, governor)
                row = {
                    "exposure_spec": exposure.name,
                    "segment_governor_spec": governor.name,
                    "roll10_max_hours": MAX_ROLL_HOURS,
                    "roll10_tp_move": tp_move,
                    "roll10_sl_move": sl_move,
                    **{f"exposure_{k}": value for k, value in asdict(exposure).items()},
                    **{f"segment_governor_{k}": value for k, value in asdict(governor).items()},
                    **flatten("validation", stop_mod.metrics(val_work)),
                    **flatten("oos", stop_mod.metrics(oos_work)),
                }
                validation_gate = bool(
                    float(row["validation_pnl"]) >= 100.0
                    and float(row["validation_mdd"]) >= -20.0
                    and float(row["validation_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and int(row["validation_overlap_count"]) == 0
                    and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                oos_gate = bool(
                    float(row["oos_pnl"]) >= 100.0
                    and float(row["oos_mdd"]) >= -20.0
                    and float(row["oos_max_hold_hours"]) <= MAX_ROLL_HOURS + 1.0e-9
                    and int(row["oos_overlap_count"]) == 0
                    and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                    and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                    and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
                )
                row["validation_daytrade_gate_pass"] = validation_gate
                row["oos_daytrade_gate_pass"] = oos_gate
                row["research_daytrade_gate_pass"] = validation_gate and oos_gate
                row["cap_floor_tiebreak_pass"] = float(row["exposure_cap_notional"]) >= 4.0
                row["cap_floor_distance_abs"] = abs(float(row["exposure_cap_notional"]) - 4.0)
                row["selection_score"] = (
                    float(row["validation_pnl"])
                    + 0.5 * (float(reference["validation_avg_hold_hours"]) - float(row["validation_avg_hold_hours"]))
                )
                rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_daytrade_gate_pass",
            "selection_score",
            "validation_pnl",
            "cap_floor_tiebreak_pass",
            "cap_floor_distance_abs",
        ],
        ascending=[False, False, False, False, True],
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
    selected_val_base = normalize_roll10(
        source.split_roll_bracket(
            val_parent,
            train_market,
            segment_mod,
            MAX_ROLL_HOURS,
            float(selected["roll10_tp_move"]),
            float(selected["roll10_sl_move"]),
        )
    )
    selected_oos_base = normalize_roll10(
        source.split_roll_bracket(
            oos_parent,
            eval_market,
            segment_mod,
            MAX_ROLL_HOURS,
            float(selected["roll10_tp_move"]),
            float(selected["roll10_sl_move"]),
        )
    )
    selected_val = segment_mod.apply_segment_exposure_governor(
        selected_val_base, selected_exposure, selected_governor
    )
    selected_oos = segment_mod.apply_segment_exposure_governor(
        selected_oos_base, selected_exposure, selected_governor
    )
    status = (
        "RESEARCH_ROLL10_DAYTRADE_PASS"
        if bool(selected["research_daytrade_gate_pass"])
        else "NO_ROLL10_DAYTRADE_PASSING_CANDIDATE"
    )
    safe = (
        f"{selected['exposure_spec']}__{selected['segment_governor_spec']}__"
        f"tp{float(selected['roll10_tp_move']):.3f}_sl{float(selected['roll10_sl_move']):.3f}"
    ).replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "roll10_bracket_daytrade_ranking.csv"
    top20_path = OUT_DIR / "roll10_bracket_daytrade_top20.csv"
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
        "selection_tiebreaker": "validation ties prefer the minimum exposure cap at or above 4.0; OOS metrics are not ordering keys",
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
    source.write_json(report_path, report)
    write_markdown(report)
    print(json.dumps({
        "report": str(report_path),
        "status": status,
        "selected_exposure": selected["exposure_spec"],
        "selected_segment_governor": selected["segment_governor_spec"],
        "selected_tp": selected["roll10_tp_move"],
        "selected_sl": selected["roll10_sl_move"],
        "selected_validation": {
            "pnl": selected["validation_pnl"],
            "mdd": selected["validation_mdd"],
            "avg_hold_hours": selected["validation_avg_hold_hours"],
            "max_hold_hours": selected["validation_max_hold_hours"],
            "gate": bool(selected["validation_daytrade_gate_pass"]),
        },
        "selected_oos": {
            "pnl": selected["oos_pnl"],
            "mdd": selected["oos_mdd"],
            "avg_hold_hours": selected["oos_avg_hold_hours"],
            "max_hold_hours": selected["oos_max_hold_hours"],
            "gate": bool(selected["oos_daytrade_gate_pass"]),
        },
    }, ensure_ascii=False, indent=2, default=source.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
