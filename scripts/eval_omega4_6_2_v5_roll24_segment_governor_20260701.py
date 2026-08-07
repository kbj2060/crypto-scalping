#!/usr/bin/env python3
"""Segment exposure/governor sweep for the Omega 4.6.2 v5 roll24 branch."""

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
SEGMENT_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_roll24_segment_governor_sweep_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll24_segment_governor_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll24_daytrade_overlay_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
PARENT_REPORT = ROOT / "tmp/causal_regen_20260516" / PARENT_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll24_segment_governor_20260701.md"


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


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    text = f"""# Omega 4.6.2 v5 Roll24 Segment Governor - 2026-07-01

## Method

This sweep starts from the v5 parent, splits trades into fixed 24h roll segments, and tunes only segment-level exposure plus a path-causal segment loss governor. Selection is validation-primary with an OOS safety gate; fresh holdout is required before any live claim.

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
    v1 = load_module("omega462_loss_cluster_v1_for_v5_segment", V1_MODULE_PATH)
    segment_mod = load_module("omega462_roll24_segment_for_v5", SEGMENT_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_v5_segment", v1.STOP_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference = read_json(REFERENCE_REPORT)["selected_variant"]
    parent_report = read_json(PARENT_REPORT)
    parent_artifacts = parent_report["artifacts"]
    val_parent = pd.read_csv(parent_artifacts["selected_validation_ledger"])
    oos_parent = pd.read_csv(parent_artifacts["selected_oos_ledger"])
    val_base = segment_mod.split_roll24_base(val_parent, train_market)
    oos_base = segment_mod.split_roll24_base(oos_parent, eval_market)

    rows: list[dict[str, Any]] = []
    governors = [
        segment_mod.SegmentGovernorSpec("none", 1.00, 1.00, 0.0),
        segment_mod.SegmentGovernorSpec("loss1_90_win12", 0.90, 1.00, 12.0),
        segment_mod.SegmentGovernorSpec("loss1_80_win12", 0.80, 1.00, 12.0),
        segment_mod.SegmentGovernorSpec("streak90_70_win12", 0.90, 0.70, 12.0),
        segment_mod.SegmentGovernorSpec("streak85_60_win12", 0.85, 0.60, 12.0),
    ]
    for long_factor in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        for short_factor in [1.00, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10]:
            for cap in [3.70, 3.80, 3.90, 4.00, 4.05, 4.10, 4.20, 4.30, 4.40, 4.60]:
                exposure = segment_mod.ExposureSpec(
                    name=f"long{int(round(long_factor * 100)):03d}_short{int(round(short_factor * 100)):03d}_cap{int(round(cap * 100)):03d}",
                    long_factor=long_factor,
                    short_factor=short_factor,
                    cap_notional=cap,
                )
                for governor in governors:
                    val_work = segment_mod.apply_segment_exposure_governor(val_base, exposure, governor)
                    oos_work = segment_mod.apply_segment_exposure_governor(oos_base, exposure, governor)
                    row = {
                        "exposure_spec": exposure.name,
                        "segment_governor_spec": governor.name,
                        **{f"exposure_{k}": value for k, value in asdict(exposure).items()},
                        **{f"segment_governor_{k}": value for k, value in asdict(governor).items()},
                        **flatten("validation", stop_mod.metrics(val_work)),
                        **flatten("oos", stop_mod.metrics(oos_work)),
                    }
                    val_mdd_abs = abs(float(row["validation_mdd"]))
                    validation_gate = bool(
                        float(row["validation_pnl"]) > float(reference["validation_pnl"])
                        and float(row["validation_mdd"]) >= -20.0
                        and float(row["validation_max_hold_hours"]) <= 24.0 + 1.0e-9
                        and int(row["validation_overlap_count"]) == 0
                        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
                        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                    )
                    oos_gate = bool(
                        float(row["oos_pnl"]) > float(reference["oos_pnl"])
                        and float(row["oos_mdd"]) >= -20.0
                        and float(row["oos_max_hold_hours"]) <= 24.0 + 1.0e-9
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
                        - float(reference["validation_pnl"])
                        - max(0.0, val_mdd_abs - 19.75) * 10.0
                    )
                    rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["research_upgrade_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_exposure = segment_mod.ExposureSpec(
        name=selected["exposure_spec"],
        long_factor=float(selected["exposure_long_factor"]),
        short_factor=float(selected["exposure_short_factor"]),
        cap_notional=float(selected["exposure_cap_notional"]),
    )
    selected_governor = segment_mod.SegmentGovernorSpec(
        name=selected["segment_governor_spec"],
        loss1_scale=float(selected["segment_governor_loss1_scale"]),
        loss2_scale=float(selected["segment_governor_loss2_scale"]),
        loss_window_hours=float(selected["segment_governor_loss_window_hours"]),
    )
    selected_val = segment_mod.apply_segment_exposure_governor(val_base, selected_exposure, selected_governor)
    selected_oos = segment_mod.apply_segment_exposure_governor(oos_base, selected_exposure, selected_governor)
    status = (
        "RESEARCH_DAYTRADE_UPGRADE_IMPROVES_REFERENCE_WITH_OOS_SAFETY_GATE"
        if bool(selected["research_upgrade_gate_pass"])
        else "NO_RESEARCH_DAYTRADE_UPGRADE_IMPROVED_REFERENCE_WITH_OOS_SAFETY_GATE"
    )

    safe = f"{selected['exposure_spec']}__{selected['segment_governor_spec']}".replace(".", "p").replace("/", "_")
    ranking_path = OUT_DIR / "v5_roll24_segment_governor_ranking.csv"
    top20_path = OUT_DIR / "v5_roll24_segment_governor_top20.csv"
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
    write_json(report_path, report)
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
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
