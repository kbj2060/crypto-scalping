#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import full_replay_omega4_4_v18_short_aged_profit_overlays_20260625 as v18  # noqa: E402
import full_replay_omega44_v18_omega3_exposure_fine_sweep_20260626 as sweep  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


BASE_MODEL_ID = "omega4_4_v18_baseline_20260624"
SOURCE_MODEL_ID = "omega3_aggressive_compensated_scale200_cap090_20260618"
FINE_SWEEP_DIR = ROOT / "tmp/causal_regen_20260516/omega44_v18_omega3_exposure_fine_sweep_20260626"
FINE_SWEEP_REPORT = FINE_SWEEP_DIR / "report.json"
FINE_SWEEP_GRID = FINE_SWEEP_DIR / "fine_exposure_grid.csv"

BASE_RUN_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
)
BASE_REPORT = BASE_RUN_DIR / "report.json"
BASE_RUNTIME_CONTRACT = BASE_RUN_DIR / "runtime_contract.json"
BASE_CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{BASE_MODEL_ID}/candidate_manifest.json"
BASE_REDTEAM_JSON = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(obj: Any) -> Any:
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


def _slug(variant: str) -> str:
    return re.sub(r"(?<=\d)p(?=\d)", "", variant)


def _model_id(variant: str) -> str:
    return f"omega4_4_v18_omega3_{_slug(variant)}_20260626"


def _row_for_variant(grid: pd.DataFrame, variant: str) -> dict[str, Any]:
    rows = grid.loc[grid["variant"].eq(variant)]
    if rows.empty:
        raise ValueError(f"variant not found in fine sweep grid: {variant}")
    return rows.iloc[0].to_dict()


def _spec_from_row(row: dict[str, Any]) -> sweep.ExposureSpec:
    partial_value = row.get("short_partial", None)
    if partial_value is None or (isinstance(partial_value, float) and math.isnan(partial_value)):
        partial = "shortpartial" in str(row["variant"]) or float(row.get("partial_fraction", 0.0) or 0.0) > 0.0
    else:
        partial = bool(partial_value)
    return sweep.ExposureSpec(
        variant=str(row["variant"]),
        mode=str(row.get("mode", "side_scaled")),
        scale=float(row.get("scale", 1.0) or 1.0),
        cap=float(row.get("cap", 0.0) or 0.0),
        fixed_notional=float(row.get("fixed_notional", 0.0) or 0.0),
        long_scale=float(row.get("long_scale", 1.0) or 1.0),
        short_scale=float(row.get("short_scale", 1.0) or 1.0),
        leverage=float(row.get("leverage", 2.0) or 2.0),
        short_partial=partial,
    )


def _metric_subset(row: dict[str, Any], split: str) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "wr",
        "trades_per_day",
        "avg_notional",
        "avg_margin_fraction",
        "avg_leverage",
        "long_entries",
        "short_entries",
        "overlay_hits",
        "exit_reasons",
        "log_growth_sum",
        "tail_excess_sum",
        "liquidation_excess_sum",
        "log_risk_utility",
    )
    return {key: row[f"{split}_{key}"] for key in keys if f"{split}_{key}" in row}


def _delta_subset(candidate: dict[str, Any], baseline: dict[str, Any], split: str) -> dict[str, float]:
    return {
        "pnl": float(candidate[f"{split}_pnl"]) - float(baseline[f"{split}_pnl"]),
        "mdd": float(candidate[f"{split}_mdd"]) - float(baseline[f"{split}_mdd"]),
        "log_risk_utility": float(candidate[f"{split}_log_risk_utility"]) - float(baseline[f"{split}_log_risk_utility"]),
        "trades": float(candidate[f"{split}_trades"]) - float(baseline[f"{split}_trades"]),
        "wr": float(candidate[f"{split}_wr"]) - float(baseline[f"{split}_wr"]),
    }


def _close(a: Any, b: Any, tol: float = 1.0e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _runtime_overlay_contract(spec: sweep.ExposureSpec) -> dict[str, Any]:
    if not spec.short_partial:
        return {"enabled": False}
    return {
        "enabled": True,
        "source_family": "omega1_2_1_horizon_short_cap_fine_20260612",
        "source_idea": "short_cap1760_min0.035 aged profitable short lifecycle guard",
        "mode": "short_aged_profit_partial_deleverage",
        "side": "short",
        "side_value": -1,
        "cap_bars": 1152,
        "bar_interval_minutes_assumption": 5,
        "cap_duration_days_assumption": 4.0,
        "min_unrealized_price_move": 0.035,
        "partial_fraction": 0.50,
        "fires_once_per_position": True,
        "execution_timing": (
            "On each in-position bar after MFE/MAE update and before standard TP/SL/exit-head checks, "
            "if a short has held at least cap_bars and current directional price move is at least "
            "min_unrealized_price_move, execute one partial close for partial_fraction of current notional; "
            "the remaining notional continues under the existing exit head and ATR safety TP/SL."
        ),
        "cash_update_contract": "closed_fraction_cash *= 1 + raw_partial_price_move * closed_notional, less execution fees",
        "remaining_position_contract": "remaining_notional = prior_notional * (1 - partial_fraction)",
        "sltp_contract": "TP/SL price-move barrier locations are unchanged by the partial de-risk event",
    }


def _risk_remap_contract(spec: sweep.ExposureSpec) -> dict[str, Any]:
    return {
        "enabled": spec.mode != "sidecar",
        "source_idea": "borrow Omega3 aggressive exposure while preserving Omega4.4 risk score ordering",
        "mode": spec.mode,
        "scale": spec.scale,
        "cap_notional": spec.cap,
        "fixed_notional": spec.fixed_notional,
        "long_scale": spec.long_scale,
        "short_scale": spec.short_scale,
        "leverage": spec.leverage,
        "notional_math": "notional = margin_fraction * leverage",
        "side_scaled_formula": "notional = min(base_margin_fraction * base_leverage * side_scale, cap_notional)",
        "margin_formula": "margin_fraction = notional / leverage",
        "sltp_contract": "ATR safety TP/SL remains a price-move barrier before PnL conversion; leverage is not multiplied twice.",
        "runtime_must_fail_on_missing_contract": True,
    }


def _audit_fields(audit_json: Path, audit_doc: Path) -> dict[str, Any]:
    if not audit_json.exists():
        return {
            "redteam_report": rel(audit_doc),
            "redteam_json": rel(audit_json),
            "redteam_verdict": "REDTEAM_PENDING",
            "redteam_pass": False,
            "promotion_pass": False,
        }
    audit = read_json(audit_json)
    return {
        "redteam_report": rel(audit_doc),
        "redteam_json": rel(audit_json),
        "redteam_verdict": audit.get("verdict"),
        "redteam_pass": audit.get("redteam_pass"),
        "research_reproduction_pass": audit.get("research_reproduction_pass"),
        "clean_oos_promotion_pass": audit.get("clean_oos_promotion_pass"),
        "promotion_pass": audit.get("promotion_pass"),
    }


def _replay_candidate(spec: sweep.ExposureSpec, run_dir: Path) -> dict[str, Any]:
    report = read_json(v18.REPORT_PATH)
    device = parent._device("cuda")
    payload, extra = v18._prepare_payload(report, device)
    fee, slip = v18.omega._load_fee_slip()
    out: dict[str, Any] = {}
    for split, (frame, base_x, dec, base_margin, base_leverage) in payload.items():
        margin, leverage = sweep._risk_arrays(spec, dec, base_margin, base_leverage)
        metrics, ledger = v18._replay_overlay(
            frame,
            base_x,
            dec,
            extra["loaded"],
            margin,
            leverage,
            sweep._overlay_spec(spec),
            report=report,
            fee=fee,
            slip=slip,
            device=device,
        )
        for key, value in metrics.items():
            out[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
        ledger.to_csv(run_dir / f"{split}_{spec.variant}_ledger.csv", index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="side_l0p90_s1p35_cap1p00_shortpartial")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--sweep-report", type=Path, default=FINE_SWEEP_REPORT)
    parser.add_argument("--sweep-grid", type=Path, default=FINE_SWEEP_GRID)
    parser.add_argument(
        "--selection-policy",
        default="strict validation/OOS diagnostic winner from Omega3 exposure fine sweep",
    )
    args = parser.parse_args()

    sweep_report = args.sweep_report.resolve()
    sweep_grid = args.sweep_grid.resolve()
    fine_report = read_json(sweep_report)
    grid = pd.read_csv(sweep_grid)
    row = _row_for_variant(grid, args.variant)
    spec = _spec_from_row(row)
    model_id = args.model_id or _model_id(args.variant)

    run_dir = ROOT / f"tmp/causal_regen_20260516/{model_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = ROOT / f"data/ensemble/supervised/{model_id}"
    candidate_manifest = candidate_dir / "candidate_manifest.json"
    promotion_manifest = run_dir / "promotion_manifest.json"
    runtime_contract = run_dir / "runtime_contract.json"
    redteam_json = run_dir / "redteam_report.json"
    redteam_doc = ROOT / f"docs/audits/{model_id}_redteam.md"
    contract_doc = ROOT / f"docs/model_contracts/{model_id}_contract.md"

    base_report = read_json(BASE_REPORT)
    base_runtime = read_json(BASE_RUNTIME_CONTRACT)
    base_manifest = read_json(BASE_CANDIDATE_MANIFEST)
    base_redteam = read_json(BASE_REDTEAM_JSON)
    baseline = fine_report["baseline"]
    replayed = _replay_candidate(spec, run_dir)
    for split in ("validation", "oos"):
        for key in ("pnl", "mdd", "trades", "wr", "avg_notional", "avg_margin_fraction", "avg_leverage", "overlay_hits", "log_risk_utility"):
            if not _close(replayed[f"{split}_{key}"], row[f"{split}_{key}"]):
                raise ValueError(
                    f"replay metric mismatch {split}_{key}: replay={replayed[f'{split}_{key}']} grid={row[f'{split}_{key}']}"
                )

    audit = _audit_fields(redteam_json, redteam_doc)
    runtime = copy.deepcopy(base_runtime)
    runtime.update(
        {
            "model_id": model_id,
            "display_version": f"Omega 4.4 v18 + Omega3 exposure transfer ({args.variant})",
            "role": "omega4_4_v18_omega3_exposure_transfer_candidate",
            "base_model_id": BASE_MODEL_ID,
            "source_model_id": SOURCE_MODEL_ID,
            "source_report_model_id": base_report.get("model_id"),
            "base_runtime_contract": rel(BASE_RUNTIME_CONTRACT),
            "base_candidate_manifest": rel(BASE_CANDIDATE_MANIFEST),
            "fine_sweep_report": rel(sweep_report),
            "fine_sweep_grid": rel(sweep_grid),
            "risk_remap": _risk_remap_contract(spec),
            "lifecycle_overlay": _runtime_overlay_contract(spec),
            "candidate_selection": {
                "policy": args.selection_policy,
                "selection_oos_informed": True,
                "clean_oos_holdout_available_for_this_candidate": False,
                "note": "This candidate may pass contract/reproduction audit but cannot claim clean OOS promotion without a fresh holdout or walk-forward confirmation.",
            },
        }
    )
    runtime.setdefault("execution_contract", {})
    runtime["execution_contract"].update(
        {
            "risk_remap_enabled": spec.mode != "sidecar",
            "lifecycle_overlay_enabled": spec.short_partial,
            "runtime_must_fail_on_missing_risk_remap_contract": True,
        }
    )
    runtime["fail_fast_required"] = True

    manifest = {
        "model_id": model_id,
        "display_version": runtime["display_version"],
        "status": "omega4_4_v18_omega3_exposure_transfer_research_candidate_not_live_wired",
        "created_kst": "2026-06-26",
        "contract_created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": BASE_MODEL_ID,
        "base_display_version": base_manifest.get("display_version"),
        "source_model_id": SOURCE_MODEL_ID,
        "base_redteam_verdict": base_redteam.get("verdict"),
        "base_redteam_pass": base_redteam.get("redteam_pass"),
        "promotion_type": "runtime_risk_remap_and_lifecycle_overlay_on_omega4_4_v18_baseline",
        "weights_retrained": False,
        "risk_sidecar_trained": False,
        "runtime_overlay_only": True,
        "variant": args.variant,
        "source_report": rel(BASE_REPORT),
        "base_runtime_contract": rel(BASE_RUNTIME_CONTRACT),
        "base_candidate_manifest": rel(BASE_CANDIDATE_MANIFEST),
        "fine_sweep_report": rel(sweep_report),
        "fine_sweep_grid": rel(sweep_grid),
        "manifest": rel(candidate_manifest),
        "runtime_contract": rel(runtime_contract),
        "promotion_manifest": rel(promotion_manifest),
        "contract": rel(contract_doc),
        **audit,
        "selection_policy": runtime["candidate_selection"],
        "risk_remap": runtime["risk_remap"],
        "lifecycle_overlay": runtime["lifecycle_overlay"],
        "baseline_validation": _metric_subset(baseline, "validation"),
        "baseline_oos_readout": _metric_subset(baseline, "oos"),
        "candidate_validation": _metric_subset(replayed, "validation"),
        "candidate_oos_readout": _metric_subset(replayed, "oos"),
        "candidate_validation_delta_vs_v18": _delta_subset(replayed, baseline, "validation"),
        "candidate_oos_delta_vs_v18": _delta_subset(replayed, baseline, "oos"),
        "base_runtime_template": runtime,
        "base_model_artifacts": {
            "parent_bundle": base_manifest.get("parent_bundle"),
            "risk_sidecar_artifact": base_manifest.get("risk_sidecar_artifact"),
            "risk_report": base_manifest.get("risk_report"),
            "parent_report": base_manifest.get("parent_report"),
        },
        "ledgers": {
            "validation": rel(run_dir / f"validation_{spec.variant}_ledger.csv"),
            "oos": rel(run_dir / f"oos_{spec.variant}_ledger.csv"),
        },
    }

    write_json(runtime_contract, runtime)
    write_json(promotion_manifest, manifest)
    write_json(candidate_manifest, manifest)

    val = manifest["candidate_validation"]
    oos = manifest["candidate_oos_readout"]
    bval = manifest["baseline_validation"]
    boos = manifest["baseline_oos_readout"]
    contract_doc.parent.mkdir(parents=True, exist_ok=True)
    contract_doc.write_text(
        "\n".join(
            [
                f"# {runtime['display_version']} Contract - 2026-06-26",
                "",
                "## Status",
                "",
                f"- Model id: `{model_id}`",
                f"- Base model: `{BASE_MODEL_ID}`",
                f"- Source model: `{SOURCE_MODEL_ID}`",
                f"- Variant: `{args.variant}`",
                "- Status: `omega4_4_v18_omega3_exposure_transfer_research_candidate_not_live_wired`",
                f"- Red-team verdict: `{audit['redteam_verdict']}`",
                "",
                "## Risk Remap Contract",
                "",
                "```json",
                json.dumps(runtime["risk_remap"], ensure_ascii=False, indent=2),
                "```",
                "",
                "## Lifecycle Overlay Contract",
                "",
                "```json",
                json.dumps(runtime["lifecycle_overlay"], ensure_ascii=False, indent=2),
                "```",
                "",
                "## Selection Caveat",
                "",
                "This candidate was selected after a validation/OOS diagnostic fine sweep.",
                "It cannot claim clean-OOS promotion until a fresh holdout or walk-forward confirmation is run.",
                "",
                "## Metrics",
                "",
                "| Split | PnL | MDD | WR | Trades | Avg notional | Overlay hits | Log-risk utility |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| Baseline validation | `{bval['pnl']:+.4f}%` | `{bval['mdd']:.4f}%` | `{bval['wr']:.4f}` | `{bval['trades']}` | `{bval['avg_notional']:.4f}` | `{bval['overlay_hits']}` | `{bval['log_risk_utility']:.6f}` |",
                f"| Candidate validation | `{val['pnl']:+.4f}%` | `{val['mdd']:.4f}%` | `{val['wr']:.4f}` | `{val['trades']}` | `{val['avg_notional']:.4f}` | `{val['overlay_hits']}` | `{val['log_risk_utility']:.6f}` |",
                f"| Baseline OOS readout | `{boos['pnl']:+.4f}%` | `{boos['mdd']:.4f}%` | `{boos['wr']:.4f}` | `{boos['trades']}` | `{boos['avg_notional']:.4f}` | `{boos['overlay_hits']}` | `{boos['log_risk_utility']:.6f}` |",
                f"| Candidate OOS readout | `{oos['pnl']:+.4f}%` | `{oos['mdd']:.4f}%` | `{oos['wr']:.4f}` | `{oos['trades']}` | `{oos['avg_notional']:.4f}` | `{oos['overlay_hits']}` | `{oos['log_risk_utility']:.6f}` |",
                "",
                "## Artifacts",
                "",
                f"- Runtime contract: `{rel(runtime_contract)}`",
                f"- Promotion manifest: `{rel(promotion_manifest)}`",
                f"- Candidate manifest: `{rel(candidate_manifest)}`",
                f"- Fine sweep report: `{rel(sweep_report)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_id": model_id,
                "candidate_manifest": str(candidate_manifest),
                "runtime_contract": str(runtime_contract),
                "promotion_manifest": str(promotion_manifest),
                "contract_doc": str(contract_doc),
                "redteam_verdict": audit["redteam_verdict"],
                "validation_pnl": val["pnl"],
                "validation_mdd": val["mdd"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
