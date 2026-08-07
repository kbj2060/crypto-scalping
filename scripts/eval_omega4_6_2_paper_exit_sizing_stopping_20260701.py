#!/usr/bin/env python3
"""Cost-sensitive exit + margin sizing overlay for Omega 4.6.2 cap220."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = ROOT / "scripts/eval_omega4_6_2_paper_exit_stopping_20260701.py"
MODEL_ID = "omega4_6_2_paper_optstop_exit_sizing_overlay_20260701"
BASE_MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
DEFAULT_RUNTIME = ROOT / "tmp/causal_regen_20260516" / BASE_MODEL_ID / "runtime_contract.json"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
EPS = 1.0e-12


@dataclass(frozen=True)
class ExposureSpec:
    name: str
    long_factor: float
    short_factor: float
    cap_notional: float
    leverage_cap: float = 5.0
    max_margin_fraction: float = 1.0


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("omega462_optstop", BASE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {BASE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
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


def exposure_specs() -> list[ExposureSpec]:
    return [
        ExposureSpec("keep_cap220", 1.00, 1.00, 2.20),
        ExposureSpec("balanced110_cap260", 1.10, 1.10, 2.60),
        ExposureSpec("balanced125_cap300", 1.25, 1.25, 3.00),
        ExposureSpec("balanced140_cap320", 1.40, 1.40, 3.20),
        ExposureSpec("balanced142_cap325", 1.42, 1.42, 3.25),
        ExposureSpec("balanced145_cap330", 1.45, 1.45, 3.30),
        ExposureSpec("balanced147_cap335", 1.47, 1.47, 3.35),
        ExposureSpec("balanced148_cap340", 1.48, 1.48, 3.40),
        ExposureSpec("balanced150_cap350", 1.50, 1.50, 3.50),
        ExposureSpec("balanced175_cap400", 1.75, 1.75, 4.00),
        ExposureSpec("balanced200_cap500", 2.00, 2.00, 5.00),
        ExposureSpec("short125_long100_cap260", 1.00, 1.25, 2.60),
        ExposureSpec("short140_long100_cap300", 1.00, 1.40, 3.00),
        ExposureSpec("short145_long100_cap310", 1.00, 1.45, 3.10),
        ExposureSpec("short147_long100_cap315", 1.00, 1.47, 3.15),
        ExposureSpec("short148_long100_cap318", 1.00, 1.48, 3.18),
        ExposureSpec("short150_long100_cap320", 1.00, 1.50, 3.20),
        ExposureSpec("short175_long100_cap400", 1.00, 1.75, 4.00),
        ExposureSpec("long125_short110_cap300", 1.25, 1.10, 3.00),
        ExposureSpec("long150_short115_cap350", 1.50, 1.15, 3.50),
    ]


def apply_exposure(df: pd.DataFrame, spec: ExposureSpec) -> pd.DataFrame:
    out = df.copy()
    active = out["notional"].astype(float) > EPS
    side = out["side"].astype(int).to_numpy()
    factor = np.where(side > 0, float(spec.long_factor), float(spec.short_factor))
    old_notional = out["notional"].astype(float).to_numpy(dtype=np.float64)
    max_notional = float(spec.leverage_cap) * float(spec.max_margin_fraction)
    target = np.minimum(old_notional * factor, min(float(spec.cap_notional), max_notional))
    target = np.where(active.to_numpy(), target, 0.0)
    leverage = np.where(active.to_numpy(), float(spec.leverage_cap), 0.0)
    margin = np.divide(target, leverage, out=np.zeros_like(target), where=leverage > EPS)
    out["notional"] = target
    out["leverage"] = leverage
    out["margin_fraction"] = margin
    out["risk_notional"] = target
    out["risk_leverage"] = leverage
    out["risk_margin_fraction"] = margin
    for col, values in {
        "exit_input_notional": target,
        "exit_input_leverage": leverage,
        "exit_input_exposure": target,
    }.items():
        if col in out.columns:
            out[col] = values
    out["paper_sizing_spec"] = spec.name
    out["paper_sizing_long_factor"] = float(spec.long_factor)
    out["paper_sizing_short_factor"] = float(spec.short_factor)
    out["paper_sizing_cap_notional"] = float(spec.cap_notional)
    out["trade_return"] = out["net_per_notional"].astype(float) * out["notional"].astype(float)
    return out


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def gate_and_score(row: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, float]:
    pnl_gain = float(row["validation_pnl"]) - float(baseline["pnl"])
    avg_hold_drop = float(baseline["avg_hold_hours"]) - float(row["validation_avg_hold_hours"])
    max_hold_drop = float(baseline["max_hold_hours"]) - float(row["validation_max_hold_hours"])
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
    score = pnl_gain + 0.30 * avg_hold_drop + 0.08 * max_hold_drop
    if mdd_abs > 20.0:
        score -= 25.0 * (mdd_abs - 20.0)
    return gate, float(score)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    base = report["baseline"]
    text = f"""# Omega 4.6.2 Paper Exit + Sizing Overlay - 2026-07-01

## Method

HF paper scan pointed to exit-time stochastic control plus cost/risk-sensitive rewards. The first exit-only sweep proved that early stopping alone cuts TP-runner convexity, so this second sweep jointly tests:

- optimal-stopping style lifecycle compression,
- margin_fraction/notional rescaling under leverage cap 5,
- validation-only selection with OOS readout.

## Selected Candidate

- Stop spec: `{selected["stop_spec"]}`
- Exposure spec: `{selected["exposure_spec"]}`
- Status: `{report["status"]}`

| Metric | Baseline Val | Selected Val | Baseline OOS | Selected OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{base["validation"]["pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{base["oos"]["pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{base["validation"]["mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{base["oos"]["mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Avg hold h | `{base["validation"]["avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{base["oos"]["avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{base["validation"]["max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{base["oos"]["max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |
| Max notional | `{base["validation"]["max_notional"]:.4f}` | `{selected["validation_max_notional"]:.4f}` | `{base["oos"]["max_notional"]:.4f}` | `{selected["oos_max_notional"]:.4f}` |

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
- Report: `{report["artifacts"]["report"]}`

This remains research-only until runtime-native replay and fresh holdout are done.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    m = load_base_module()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = m.read_json(m.resolve_path(args.runtime_contract))
    train_csv, eval_csv = m.component_train_eval_paths(runtime)
    train_market = m.load_market(train_csv)
    eval_market = m.load_market(eval_csv)
    source_dir = m.resolve_path(runtime["source_report"]).parent
    val_path, oos_path = m.source_variant_ledgers(source_dir)
    val = m.ensure_time_columns(pd.read_csv(val_path))
    oos = m.ensure_time_columns(pd.read_csv(oos_path))
    baseline_val = m.metrics(val)
    baseline_oos = m.metrics(oos)
    specs = m.stop_specs()
    exposures = exposure_specs()

    rows: list[dict[str, Any]] = []
    best_ledgers: tuple[pd.DataFrame, pd.DataFrame] | None = None
    for stop_spec in specs:
        val_stop = m.apply_stop_spec(val, train_market, stop_spec)
        oos_stop = m.apply_stop_spec(oos, eval_market, stop_spec)
        for exp_spec in exposures:
            val_work = apply_exposure(val_stop, exp_spec)
            oos_work = apply_exposure(oos_stop, exp_spec)
            val_metrics = m.metrics(val_work)
            oos_metrics = m.metrics(oos_work)
            row = {
                "stop_spec": stop_spec.name,
                "exposure_spec": exp_spec.name,
                **{f"stop_{k}": v for k, v in asdict(stop_spec).items()},
                **{f"exposure_{k}": v for k, v in asdict(exp_spec).items()},
                **flatten("validation", val_metrics),
                **flatten("oos", oos_metrics),
            }
            gate, score = gate_and_score(row, baseline_val)
            row["validation_gate_pass"] = gate
            row["selection_score"] = score
            rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_gate_pass", "selection_score", "validation_pnl", "validation_avg_hold_hours"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_stop = next(spec for spec in specs if spec.name == selected["stop_spec"])
    selected_exp = next(spec for spec in exposures if spec.name == selected["exposure_spec"])
    selected_val = apply_exposure(m.apply_stop_spec(val, train_market, selected_stop), selected_exp)
    selected_oos = apply_exposure(m.apply_stop_spec(oos, eval_market, selected_stop), selected_exp)
    status = (
        "VALIDATION_SELECTED_CANDIDATE_IMPROVES_PNL_AND_HOLD_TIME"
        if bool(selected["validation_gate_pass"])
        else "NO_VALIDATION_CANDIDATE_IMPROVED_BOTH_PNL_AND_HOLD_TIME_WITH_MDD_LE_20"
    )

    safe = f"{selected['stop_spec']}__{selected['exposure_spec']}".replace(".", "p").replace("/", "_")
    ranking_path = out_dir / "paper_exit_sizing_ranking.csv"
    top20_path = out_dir / "paper_exit_sizing_top20.csv"
    val_out = out_dir / f"validation_{safe}_ledger.csv"
    oos_out = out_dir / f"oos_{safe}_ledger.csv"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)

    report = {
        "model_id": MODEL_ID,
        "base_model_id": BASE_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_only; OOS readout only",
        "paper_sources": [
            {"paper_id": "2302.07320", "url": "https://hf.co/papers/2302.07320"},
            {"paper_id": "2003.03051", "url": "https://hf.co/papers/2003.03051"},
            {"paper_id": "2505.04553", "url": "https://hf.co/papers/2505.04553"},
        ],
        "baseline": {"validation": baseline_val, "oos": baseline_oos},
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
            "report": str(out_dir / "report.json"),
            "audit_md": str(ROOT / "docs/audits/omega4_6_2_paper_exit_sizing_overlay_20260701.md"),
        },
        "live_promotion_note": "Research-only: requires native replay and fresh holdout.",
    }
    write_json(out_dir / "report.json", report)
    write_markdown(ROOT / "docs/audits/omega4_6_2_paper_exit_sizing_overlay_20260701.md", report)
    print(
        json.dumps(
            {
                "report": str(out_dir / "report.json"),
                "status": status,
                "selected_stop": selected["stop_spec"],
                "selected_exposure": selected["exposure_spec"],
                "baseline_validation": baseline_val,
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "max_notional": selected["validation_max_notional"],
                },
                "baseline_oos": baseline_oos,
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "max_notional": selected["oos_max_notional"],
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
