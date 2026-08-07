#!/usr/bin/env python3
"""Third loss-cluster sweep: earlier trail/stall exits for lower average hold."""

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
V1_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_loss_cluster_governor_20260701.py"
MODEL_ID = "omega4_6_2_loss_cluster_governor_v3_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_loss_cluster_governor_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_loss_cluster_governor_v3_20260701.md"


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
    specs = []
    loss_specs = [("loss48_4p5", 48.0, -0.045), ("loss60_5p0", 60.0, -0.050)]
    trail_specs = [
        ("trail48_5p5_gap1p8", 48.0, 0.055, 0.018, 0.015),
        ("trail60_6p5_gap2p0", 60.0, 0.065, 0.020, 0.020),
        ("trail72_7p0_gap2p5", 72.0, 0.070, 0.025, 0.025),
    ]
    stall_specs = [
        ("stall48_lb12_min4p5", 48.0, 12.0, 0.045, 0.0015),
        ("stall60_lb12_min5p0", 60.0, 12.0, 0.050, 0.0020),
        ("stall72_lb24_min5p5", 72.0, 24.0, 0.055, 0.0020),
        ("stall84_lb24_min6p0", 84.0, 24.0, 0.060, 0.0025),
    ]
    for hard in [84.0, 90.0]:
        for loss_name, loss_after, loss_stop in loss_specs:
            for trail_name, trail_after, trail_arm, trail_gap, trail_floor in trail_specs:
                for stall_name, stall_after, stall_lb, stall_min, stall_slope in stall_specs:
                    specs.append(
                        stop_mod.StopSpec(
                            name=f"hard{int(hard)}__{loss_name}__{trail_name}__{stall_name}",
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


def exposure_specs(v1: Any) -> list[Any]:
    specs = []
    for long_factor in [0.75, 1.00, 1.20]:
        for short_factor in [1.70, 1.78, 1.86, 1.95, 2.05, 2.15, 2.25]:
            cap = min(5.0, round(2.15 * short_factor, 2))
            specs.append(
                v1.ExposureSpec(
                    name=(
                        f"long{int(round(long_factor * 100)):03d}_"
                        f"short{int(round(short_factor * 100)):03d}_cap{int(round(cap * 100)):03d}"
                    ),
                    long_factor=long_factor,
                    short_factor=short_factor,
                    cap_notional=cap,
                )
            )
    for factor in [1.70, 1.78, 1.86, 1.95, 2.05]:
        cap = min(5.0, round(2.30 * factor, 2))
        specs.append(
            v1.ExposureSpec(
                name=f"balanced{int(round(factor * 100)):03d}_cap{int(round(cap * 100)):03d}",
                long_factor=factor,
                short_factor=factor,
                cap_notional=cap,
            )
        )
    return specs


def governor_specs(v1: Any) -> list[Any]:
    return [
        v1.GovernorSpec("loss1_75_win12", 0.75, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        v1.GovernorSpec("loss1_70_win12", 0.70, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        v1.GovernorSpec("loss1_65_win12", 0.65, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        v1.GovernorSpec("loss1_60_win12", 0.60, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        v1.GovernorSpec("loss1_55_win12", 0.55, 1.00, -1.0, 1.00, -1.0, 1.00, 12.0),
        v1.GovernorSpec("loss1_65_win24", 0.65, 1.00, -1.0, 1.00, -1.0, 1.00, 24.0),
    ]


def gate_and_score(row: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, float]:
    pnl_gain = float(row["validation_pnl"]) - float(reference["validation_pnl"])
    avg_hold_drop = float(reference["validation_avg_hold_hours"]) - float(row["validation_avg_hold_hours"])
    max_hold_ok = float(row["validation_max_hold_hours"]) <= float(reference["validation_max_hold_hours"])
    mdd_abs = abs(float(row["validation_mdd"]))
    gate = bool(
        pnl_gain > 0.0
        and avg_hold_drop > 0.0
        and max_hold_ok
        and mdd_abs <= 20.0
        and int(row["validation_overlap_count"]) == 0
        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
    )
    score = pnl_gain + 0.60 * avg_hold_drop
    if mdd_abs > 20.0:
        score -= 60.0 * (mdd_abs - 20.0)
    return gate, float(score)


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 Loss-Cluster Governor v3 - 2026-07-01

## Method

This sweep keeps max hold at or below v1 but moves trail/stall exits earlier to reduce average hold. Selection remains validation-only; OOS is readout.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Selection scope: `{report["selection_scope"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Stop spec: `{selected["stop_spec"]}`
- Exposure spec: `{selected["exposure_spec"]}`
- Governor spec: `{selected["governor_spec"]}`
- Validation upgrade gate pass: `{selected["validation_upgrade_gate_pass"]}`

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
    v1 = load_module("omega462_loss_cluster_v1_for_v3", V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_v3", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    source_dir = stop_mod.resolve_path(runtime["source_report"]).parent
    val_path, oos_path = stop_mod.source_variant_ledgers(source_dir)
    val = stop_mod.ensure_time_columns(pd.read_csv(val_path))
    oos = stop_mod.ensure_time_columns(pd.read_csv(oos_path))
    reference = read_json(REFERENCE_REPORT)["selected_variant"]

    stop_list = stop_specs(stop_mod)
    exposure_list = exposure_specs(v1)
    governor_list = governor_specs(v1)
    rows: list[dict[str, Any]] = []
    for stop_spec in stop_list:
        val_stop = stop_mod.apply_stop_spec(val, train_market, stop_spec)
        oos_stop = stop_mod.apply_stop_spec(oos, eval_market, stop_spec)
        for exposure in exposure_list:
            for governor in governor_list:
                val_work = v1.apply_exposure_governor(val_stop, exposure, governor)
                oos_work = v1.apply_exposure_governor(oos_stop, exposure, governor)
                row = {
                    "stop_spec": stop_spec.name,
                    "exposure_spec": exposure.name,
                    "governor_spec": governor.name,
                    **{f"stop_{k}": value for k, value in asdict(stop_spec).items()},
                    **{f"exposure_{k}": value for k, value in asdict(exposure).items()},
                    **{f"governor_{k}": value for k, value in asdict(governor).items()},
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
    selected_stop = next(spec for spec in stop_list if spec.name == selected["stop_spec"])
    selected_exposure = next(spec for spec in exposure_list if spec.name == selected["exposure_spec"])
    selected_governor = next(spec for spec in governor_list if spec.name == selected["governor_spec"])
    selected_val = v1.apply_exposure_governor(
        stop_mod.apply_stop_spec(val, train_market, selected_stop),
        selected_exposure,
        selected_governor,
    )
    selected_oos = v1.apply_exposure_governor(
        stop_mod.apply_stop_spec(oos, eval_market, selected_stop),
        selected_exposure,
        selected_governor,
    )
    status = (
        "VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_AND_AVG_HOLD"
        if bool(selected["validation_upgrade_gate_pass"])
        else "NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_AND_AVG_HOLD"
    )
    safe = (
        f"{selected['stop_spec']}__{selected['exposure_spec']}__{selected['governor_spec']}"
        .replace(".", "p")
        .replace("/", "_")
    )
    ranking_path = OUT_DIR / "loss_cluster_governor_v3_ranking.csv"
    top20_path = OUT_DIR / "loss_cluster_governor_v3_top20.csv"
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
        "base_model_id": BASE_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only; OOS readout only",
        "reference_variant": reference,
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
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_stop": selected["stop_spec"],
                "selected_exposure": selected["exposure_spec"],
                "selected_governor": selected["governor_spec"],
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
