#!/usr/bin/env python3
"""24h roll overlay for the Omega 4.6.2 v5 loss-cluster candidate."""

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
V1_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_loss_cluster_governor_20260701.py"
ROLL24_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_roll24_daytrade_overlay_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll24_daytrade_overlay_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_roll24_daytrade_overlay_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
PARENT_REPORT = ROOT / "tmp/causal_regen_20260516" / PARENT_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll24_daytrade_overlay_20260701.md"
MAX_ROLL_HOURS = 24.0


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


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    parent = report["parent_variant"]
    text = f"""# Omega 4.6.2 v5 Roll24 Daytrade Overlay - 2026-07-01

## Method

This overlay applies the same 24h roll segmentation used by the prior roll24 branch to the v5 loss-cluster parent ledger. The roll transformation is fixed; OOS is readout only.

## Result

- Status: `{report["status"]}`
- Parent model: `{report["parent_model_id"]}`
- Reference daytrade model: `{report["reference_model_id"]}`

| Metric | Reference Roll24 Val | v5 Roll24 Val | Reference Roll24 OOS | v5 Roll24 OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Parent Context

- Parent validation PnL: `{parent["validation_pnl"]:.4f}%`
- Parent OOS PnL: `{parent["oos_pnl"]:.4f}%`
- Parent max hold: `{parent["validation_max_hold_hours"]:.4f}h` / `{parent["oos_max_hold_hours"]:.4f}h`

## Artifacts

- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    v1 = load_module("omega462_loss_cluster_v1_for_v5_roll24", V1_MODULE_PATH)
    stop_mod = v1.load_module("omega462_stop_mod_for_v5_roll24", v1.STOP_MODULE_PATH)
    roll24 = load_module("omega462_roll24_for_v5", ROLL24_MODULE_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runtime = stop_mod.read_json(stop_mod.resolve_path(v1.DEFAULT_RUNTIME))
    train_csv, eval_csv = stop_mod.component_train_eval_paths(runtime)
    train_market = stop_mod.load_market(train_csv)
    eval_market = stop_mod.load_market(eval_csv)
    reference_report = read_json(REFERENCE_REPORT)
    reference = reference_report["selected_variant"]
    parent_report = read_json(PARENT_REPORT)
    parent = parent_report["selected_variant"]
    parent_artifacts = parent_report["artifacts"]
    val = pd.read_csv(parent_artifacts["selected_validation_ledger"])
    oos = pd.read_csv(parent_artifacts["selected_oos_ledger"])

    roll_val = roll24.split_roll24(val, train_market)
    roll_oos = roll24.split_roll24(oos, eval_market)
    val_metrics = stop_mod.metrics(roll_val)
    oos_metrics = stop_mod.metrics(roll_oos)
    selected = {
        "overlay": "v5_roll24_daytrade",
        "parent_model_id": PARENT_MODEL_ID,
        "max_roll_hours": MAX_ROLL_HOURS,
        **{f"validation_{k}": v for k, v in val_metrics.items() if not isinstance(v, (dict, list))},
        "validation_reason_counts": json.dumps(val_metrics["reason_counts"], ensure_ascii=False, sort_keys=True),
        **{f"oos_{k}": v for k, v in oos_metrics.items() if not isinstance(v, (dict, list))},
        "oos_reason_counts": json.dumps(oos_metrics["reason_counts"], ensure_ascii=False, sort_keys=True),
    }
    daytrade_pass = bool(
        val_metrics["pnl"] >= 100.0
        and oos_metrics["pnl"] >= 100.0
        and val_metrics["mdd"] >= -20.0
        and oos_metrics["mdd"] >= -20.0
        and val_metrics["max_hold_hours"] <= 24.0
        and oos_metrics["max_hold_hours"] <= 24.0
    )
    pnl_upgrade_vs_reference = bool(
        val_metrics["pnl"] > reference["validation_pnl"]
        and oos_metrics["pnl"] > reference["oos_pnl"]
    )
    status = (
        "DAYTRADE_HOLD_AND_PNL_PASS"
        if daytrade_pass and pnl_upgrade_vs_reference
        else "DAYTRADE_HOLD_PASS_PNL_LOWER_THAN_REFERENCE"
        if daytrade_pass
        else "DAYTRADE_HOLD_FAIL"
    )
    val_out = OUT_DIR / "validation_v5_roll24_daytrade_ledger.csv"
    oos_out = OUT_DIR / "oos_v5_roll24_daytrade_ledger.csv"
    report_path = OUT_DIR / "report.json"
    roll_val.to_csv(val_out, index=False)
    roll_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "fixed_v5_overlay; OOS readout only",
        "reference_variant": reference,
        "parent_variant": parent,
        "selected_variant": selected,
        "status": status,
        "daytrade_pass": daytrade_pass,
        "pnl_upgrade_vs_reference": pnl_upgrade_vs_reference,
        "artifacts": {
            "out_dir": str(OUT_DIR),
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
                "daytrade_pass": daytrade_pass,
                "pnl_upgrade_vs_reference": pnl_upgrade_vs_reference,
                "validation": val_metrics,
                "oos": oos_metrics,
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
