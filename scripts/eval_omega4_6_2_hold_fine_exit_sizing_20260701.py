#!/usr/bin/env python3
"""Fine hold-time compression sweep after the Omega 4.6.2 paper exit+sizing candidate."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_STOP_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_paper_exit_stopping_20260701.py"
SIZING_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_paper_exit_sizing_stopping_20260701.py"
MODEL_ID = "omega4_6_2_hold_fine_exit_sizing_overlay_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_paper_optstop_exit_sizing_overlay_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
DEFAULT_RUNTIME = ROOT / "tmp/causal_regen_20260516" / BASE_MODEL_ID / "runtime_contract.json"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_hold_fine_exit_sizing_overlay_20260701.md"
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def stop_specs(stop_mod: Any) -> list[Any]:
    loss_specs = [
        ("loss36_3p5", 36.0, -0.035),
        ("loss48_4p5", 48.0, -0.045),
        ("loss60_5p0", 60.0, -0.050),
    ]
    trail_specs = [
        ("trail48_6p5_gap2p0", 48.0, 0.065, 0.020, 0.020),
        ("trail60_6p8_gap2p3", 60.0, 0.068, 0.023, 0.022),
        ("trail72_7p0_gap2p5", 72.0, 0.070, 0.025, 0.025),
    ]
    stall_specs = [
        ("stall60_lb12_min5p0", 60.0, 12.0, 0.050, 0.0020),
        ("stall72_lb24_min5p5", 72.0, 24.0, 0.055, 0.0020),
        ("stall84_lb24_min6p0", 84.0, 24.0, 0.060, 0.0025),
        ("stall96_lb24_min6p5", 96.0, 24.0, 0.065, 0.0025),
    ]
    specs = []
    for hard in [72.0, 78.0, 84.0, 90.0, 96.0]:
        for loss_name, loss_after, loss_stop in loss_specs:
            for trail_name, trail_after, trail_arm, trail_gap, trail_floor in trail_specs:
                for stall_name, stall_after, stall_lb, stall_min, stall_slope in stall_specs:
                    name = f"hard{int(hard)}__{loss_name}__{trail_name}__{stall_name}"
                    specs.append(
                        stop_mod.StopSpec(
                            name=name,
                            hard_stop_hours=hard,
                            loss_after_hours=loss_after,
                            loss_stop_move=loss_stop,
                            trail_after_hours=trail_after,
                            trail_arm_move=trail_arm,
                            trail_giveback_move=trail_gap,
                            trail_floor_move=trail_floor,
                            stall_after_hours=stall_after,
                            stall_lookback_hours=stall_lb,
                            stall_min_profit_move=stall_min,
                            stall_slope_max=stall_slope,
                        )
                    )
    return specs


def exposure_specs(sizing_mod: Any) -> list[Any]:
    specs = []
    for factor in [1.34, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52, 1.55, 1.58]:
        cap = min(5.0, round(2.30 * factor, 2))
        name = f"balanced{int(round(factor * 100)):03d}_cap{int(round(cap * 100)):03d}"
        specs.append(sizing_mod.ExposureSpec(name, factor, factor, cap))
    for short_factor in [1.45, 1.48, 1.52, 1.55, 1.58, 1.62]:
        cap = min(5.0, round(2.15 * short_factor, 2))
        name = f"short{int(round(short_factor * 100)):03d}_long100_cap{int(round(cap * 100)):03d}"
        specs.append(sizing_mod.ExposureSpec(name, 1.00, short_factor, cap))
    for long_factor in [0.80, 1.00, 1.20]:
        for short_factor in [1.48, 1.55]:
            cap = min(5.0, round(2.20 * max(long_factor, short_factor), 2))
            name = (
                f"long{int(round(long_factor * 100)):03d}_"
                f"short{int(round(short_factor * 100)):03d}_cap{int(round(cap * 100)):03d}"
            )
            specs.append(sizing_mod.ExposureSpec(name, long_factor, short_factor, cap))
    return specs


def gate_and_score(row: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, float]:
    pnl_gain = float(row["validation_pnl"]) - float(reference["validation_pnl"])
    avg_hold_drop = float(reference["validation_avg_hold_hours"]) - float(row["validation_avg_hold_hours"])
    max_hold_drop = float(reference["validation_max_hold_hours"]) - float(row["validation_max_hold_hours"])
    mdd_abs = abs(float(row["validation_mdd"]))
    gate = bool(
        pnl_gain > 0.0
        and avg_hold_drop > 0.0
        and max_hold_drop > 0.0
        and mdd_abs <= 20.0
        and int(row["validation_overlap_count"]) == 0
        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
    )
    score = pnl_gain + 0.50 * avg_hold_drop + 0.15 * max_hold_drop
    if mdd_abs > 20.0:
        score -= 35.0 * (mdd_abs - 20.0)
    return gate, float(score)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 Hold Fine Exit + Sizing Overlay - 2026-07-01

## Method

This sweep keeps the Omega 4.6.2 cap220 source ledgers and the paper exit+sizing framework, then searches finer hard-stop horizons between 72h and 96h plus validation-only exposure rescaling.

## Result

- Status: `{report["status"]}`
- Selection scope: `{report["selection_scope"]}`
- Reference model: `{report["reference_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Stop spec: `{selected["stop_spec"]}`
- Exposure spec: `{selected["exposure_spec"]}`
- Validation upgrade gate pass: `{selected["validation_upgrade_gate_pass"]}`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    stop_mod = load_module("omega462_stop_mod", BASE_STOP_MODULE_PATH)
    sizing_mod = load_module("omega462_sizing_mod", SIZING_MODULE_PATH)
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    source_dir = stop_mod.resolve_path(runtime["source_report"]).parent
    val_path, oos_path = stop_mod.source_variant_ledgers(source_dir)
    val = stop_mod.ensure_time_columns(pd.read_csv(val_path))
    oos = stop_mod.ensure_time_columns(pd.read_csv(oos_path))
    reference_report = read_json(REFERENCE_REPORT)
    reference = reference_report["selected_variant"]

    rows: list[dict[str, Any]] = []
    for stop_spec in stop_specs(stop_mod):
        val_stop = stop_mod.apply_stop_spec(val, train_market, stop_spec)
        oos_stop = stop_mod.apply_stop_spec(oos, eval_market, stop_spec)
        for exp_spec in exposure_specs(sizing_mod):
            val_work = sizing_mod.apply_exposure(val_stop, exp_spec)
            oos_work = sizing_mod.apply_exposure(oos_stop, exp_spec)
            row = {
                "stop_spec": stop_spec.name,
                "exposure_spec": exp_spec.name,
                **{f"stop_{k}": v for k, v in asdict(stop_spec).items()},
                **{f"exposure_{k}": v for k, v in asdict(exp_spec).items()},
                **flatten("validation", stop_mod.metrics(val_work)),
                **flatten("oos", stop_mod.metrics(oos_work)),
            }
            gate, score = gate_and_score(row, reference)
            row["validation_upgrade_gate_pass"] = gate
            row["selection_score"] = score
            rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_upgrade_gate_pass", "selection_score", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_stop = next(spec for spec in stop_specs(stop_mod) if spec.name == selected["stop_spec"])
    selected_exp = next(spec for spec in exposure_specs(sizing_mod) if spec.name == selected["exposure_spec"])
    selected_val = sizing_mod.apply_exposure(
        stop_mod.apply_stop_spec(val, train_market, selected_stop), selected_exp
    )
    selected_oos = sizing_mod.apply_exposure(
        stop_mod.apply_stop_spec(oos, eval_market, selected_stop), selected_exp
    )

    status = (
        "VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_AND_HOLD"
        if bool(selected["validation_upgrade_gate_pass"])
        else "NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_AND_HOLD"
    )
    safe = f"{selected['stop_spec']}__{selected['exposure_spec']}".replace(".", "p").replace("/", "_")
    ranking_path = out_dir / "hold_fine_ranking.csv"
    top20_path = out_dir / "hold_fine_top20.csv"
    val_out = out_dir / f"validation_{safe}_ledger.csv"
    oos_out = out_dir / f"oos_{safe}_ledger.csv"
    report_path = out_dir / "report.json"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)

    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only; OOS readout only",
        "reference_variant": reference,
        "variants_evaluated": int(len(ranking)),
        "selected_variant": selected,
        "top20": ranking.head(20).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(ranking_path),
            "top20": str(top20_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(AUDIT_MD, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_stop": selected["stop_spec"],
                "selected_exposure": selected["exposure_spec"],
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
