#!/usr/bin/env python3
"""V5 fine exposure sweep around the Omega 4.6.2 v4 governor candidate."""

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
MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_loss_cluster_governor_v4_fine_exposure_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701.md"


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
    text = f"""# Omega 4.6.2 Loss-Cluster Governor v5 Fine Exposure - 2026-07-01

## Method

This sweep freezes the v4 stop design and performs a narrow validation-only search around the high-PnL exposure boundary. It keeps the loss-window governor path-causal and only changes long/short exposure factors, notional cap, and the first-loss scale.

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
    v1 = load_module("omega462_loss_cluster_v1_for_v5", V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_v5", v1.STOP_MODULE_PATH)
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

    stop_spec = stop_mod.StopSpec(
        name="hard90__loss48_4p5__trail72_7p0_gap2p5__stall72_lb24_min5p5",
        hard_stop_hours=90.0,
        loss_after_hours=48.0,
        loss_stop_move=-0.045,
        trail_after_hours=72.0,
        trail_arm_move=0.070,
        trail_giveback_move=0.025,
        trail_floor_move=0.025,
        stall_after_hours=72.0,
        stall_lookback_hours=24.0,
        stall_min_profit_move=0.055,
        stall_slope_max=0.0020,
    )
    val_stop = stop_mod.apply_stop_spec(val, train_market, stop_spec)
    oos_stop = stop_mod.apply_stop_spec(oos, eval_market, stop_spec)

    rows: list[dict[str, Any]] = []
    for long_factor in [1.20, 1.25, 1.30]:
        for short_factor in [1.945, 1.950, 1.955]:
            for cap_multiplier in [2.10, 2.12, 2.14, 2.15, 2.16, 2.18, 2.20]:
                cap = round(cap_multiplier * short_factor, 3)
                for loss1_scale in [0.50, 0.525, 0.55]:
                    exposure = v1.ExposureSpec(
                        name=(
                            f"long{int(round(long_factor * 1000)):04d}_"
                            f"short{int(round(short_factor * 1000)):04d}_"
                            f"cap{int(round(cap * 1000)):04d}"
                        ),
                        long_factor=long_factor,
                        short_factor=short_factor,
                        cap_notional=cap,
                    )
                    governor = v1.GovernorSpec(
                        name=f"loss1_{int(round(loss1_scale * 1000)):03d}_win12",
                        loss1_scale=loss1_scale,
                        loss2_scale=1.00,
                        dd1_threshold=-1.0,
                        dd1_scale=1.00,
                        dd2_threshold=-1.0,
                        dd2_scale=1.00,
                        loss_window_hours=12.0,
                    )
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
                    pnl_gain = float(row["validation_pnl"]) - float(reference["validation_pnl"])
                    mdd_abs = abs(float(row["validation_mdd"]))
                    gate = bool(
                        pnl_gain > 0.0
                        and float(row["validation_avg_hold_hours"])
                        <= float(reference["validation_avg_hold_hours"]) + 1.0e-9
                        and float(row["validation_max_hold_hours"])
                        <= float(reference["validation_max_hold_hours"]) + 1.0e-9
                        and mdd_abs <= 20.0
                        and int(row["validation_overlap_count"]) == 0
                        and float(row["validation_max_leverage"]) <= 5.0 + 1.0e-9
                        and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                        and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                        and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
                    )
                    row["validation_upgrade_gate_pass"] = gate
                    row["selection_score"] = pnl_gain - max(0.0, mdd_abs - 19.95) * 20.0
                    rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_upgrade_gate_pass", "selection_score", "validation_pnl"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_exposure = v1.ExposureSpec(
        name=selected["exposure_spec"],
        long_factor=float(selected["exposure_long_factor"]),
        short_factor=float(selected["exposure_short_factor"]),
        cap_notional=float(selected["exposure_cap_notional"]),
    )
    selected_governor = v1.GovernorSpec(
        name=selected["governor_spec"],
        loss1_scale=float(selected["governor_loss1_scale"]),
        loss2_scale=float(selected["governor_loss2_scale"]),
        dd1_threshold=float(selected["governor_dd1_threshold"]),
        dd1_scale=float(selected["governor_dd1_scale"]),
        dd2_threshold=float(selected["governor_dd2_threshold"]),
        dd2_scale=float(selected["governor_dd2_scale"]),
        loss_window_hours=float(selected["governor_loss_window_hours"]),
    )
    selected_val = v1.apply_exposure_governor(val_stop, selected_exposure, selected_governor)
    selected_oos = v1.apply_exposure_governor(oos_stop, selected_exposure, selected_governor)
    status = (
        "VALIDATION_UPGRADE_IMPROVES_REFERENCE_PNL_WITH_HOLD_NOT_WORSE"
        if bool(selected["validation_upgrade_gate_pass"])
        else "NO_VALIDATION_UPGRADE_IMPROVED_REFERENCE_PNL_WITH_HOLD_NOT_WORSE"
    )

    safe = (
        f"{selected['stop_spec']}__{selected['exposure_spec']}__{selected['governor_spec']}"
        .replace(".", "p")
        .replace("/", "_")
    )
    ranking_path = OUT_DIR / "loss_cluster_governor_v5_fine_exposure_ranking.csv"
    top20_path = OUT_DIR / "loss_cluster_governor_v5_fine_exposure_top20.csv"
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
