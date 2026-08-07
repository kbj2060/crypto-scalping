#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "omega4_4_v18_short_partial_cap1152_u0035_p050_20260626"
DISPLAY_VERSION = "Omega 4.4 v18 short aged-profit partial de-risk overlay"
BASE_MODEL_ID = "omega4_4_v18_baseline_20260624"
SOURCE_VARIANT = "short_partial_cap1152_u0.035_p0.50"

BASE_RUN_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
)
BASE_REPORT = BASE_RUN_DIR / "report.json"
BASE_RUNTIME_CONTRACT = BASE_RUN_DIR / "runtime_contract.json"
BASE_CANDIDATE_MANIFEST = ROOT / f"data/ensemble/supervised/{BASE_MODEL_ID}/candidate_manifest.json"
BASE_REDTEAM_JSON = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json"

OVERLAY_DIR = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_short_aged_profit_overlay_full_replay_20260625"
OVERLAY_REPORT = OVERLAY_DIR / "report.json"
OVERLAY_GRID = OVERLAY_DIR / "full_replay_overlay_results.csv"

RUN_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
PROMOTION_MANIFEST = RUN_DIR / "promotion_manifest.json"
RUNTIME_CONTRACT = RUN_DIR / "runtime_contract.json"
AUDIT_JSON = RUN_DIR / "redteam_report.json"
AUDIT_DOC = ROOT / f"docs/audits/{MODEL_ID}_redteam.md"

CANDIDATE_DIR = ROOT / f"data/ensemble/supervised/{MODEL_ID}"
CANDIDATE_MANIFEST = CANDIDATE_DIR / "candidate_manifest.json"
CONTRACT_DOC = ROOT / f"docs/model_contracts/{MODEL_ID}_contract.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_result(report: dict[str, Any], variant: str) -> dict[str, Any]:
    for row in report["results"]:
        if row.get("variant") == variant:
            return row
    raise KeyError(f"variant not found in overlay report: {variant}")


def metric_subset(row: dict[str, Any], split: str) -> dict[str, Any]:
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
    out: dict[str, Any] = {}
    for key in keys:
        full_key = f"{split}_{key}"
        if full_key in row:
            out[key] = row[full_key]
    return out


def delta_subset(candidate: dict[str, Any], baseline: dict[str, Any], split: str) -> dict[str, float]:
    return {
        "pnl": float(candidate[f"{split}_pnl"]) - float(baseline[f"{split}_pnl"]),
        "mdd": float(candidate[f"{split}_mdd"]) - float(baseline[f"{split}_mdd"]),
        "log_risk_utility": float(candidate[f"{split}_log_risk_utility"]) - float(baseline[f"{split}_log_risk_utility"]),
        "trades": float(candidate[f"{split}_trades"]) - float(baseline[f"{split}_trades"]),
        "wr": float(candidate[f"{split}_wr"]) - float(baseline[f"{split}_wr"]),
    }


def audit_fields() -> dict[str, Any]:
    if not AUDIT_JSON.exists():
        return {
            "redteam_report": rel(AUDIT_DOC),
            "redteam_json": rel(AUDIT_JSON),
            "redteam_verdict": "REDTEAM_PENDING",
            "redteam_pass": False,
            "promotion_pass": False,
        }
    audit = read_json(AUDIT_JSON)
    return {
        "redteam_report": rel(AUDIT_DOC),
        "redteam_json": rel(AUDIT_JSON),
        "redteam_verdict": audit.get("verdict"),
        "redteam_pass": audit.get("redteam_pass"),
        "research_reproduction_pass": audit.get("research_reproduction_pass"),
        "clean_oos_promotion_pass": audit.get("clean_oos_promotion_pass"),
        "promotion_pass": audit.get("promotion_pass"),
    }


def build_lifecycle_overlay() -> dict[str, Any]:
    return {
        "enabled": True,
        "source_family": "omega1_2_1_horizon_short_cap_fine_20260612",
        "source_idea": "short_cap1760_min0.035 aged profitable short lifecycle guard",
        "source_variant": SOURCE_VARIANT,
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


def main() -> int:
    base_report = read_json(BASE_REPORT)
    base_runtime = read_json(BASE_RUNTIME_CONTRACT)
    base_manifest = read_json(BASE_CANDIDATE_MANIFEST)
    base_redteam = read_json(BASE_REDTEAM_JSON)
    overlay_report = read_json(OVERLAY_REPORT)
    baseline = overlay_report["baseline"]
    candidate = find_result(overlay_report, SOURCE_VARIANT)

    expected = {
        "mode": "partial_deleverage",
        "side": -1,
        "cap_bars": 1152,
        "min_unreal": 0.035,
        "partial_fraction": 0.5,
    }
    for key, value in expected.items():
        actual = candidate.get(key)
        if actual != value:
            raise ValueError(f"{SOURCE_VARIANT} contract mismatch: {key}={actual!r}, expected {value!r}")

    lifecycle_overlay = build_lifecycle_overlay()
    audit = audit_fields()

    runtime = copy.deepcopy(base_runtime)
    runtime.update(
        {
            "model_id": MODEL_ID,
            "display_version": DISPLAY_VERSION,
            "role": "omega4_4_v18_short_aged_profit_partial_derisk_candidate",
            "base_model_id": BASE_MODEL_ID,
            "source_report_model_id": base_report.get("model_id"),
            "base_runtime_contract": rel(BASE_RUNTIME_CONTRACT),
            "base_candidate_manifest": rel(BASE_CANDIDATE_MANIFEST),
            "overlay_source_report": rel(OVERLAY_REPORT),
            "overlay_source_grid": rel(OVERLAY_GRID),
            "lifecycle_overlay": lifecycle_overlay,
            "candidate_selection": {
                "policy": "user-approved balanced validation/OOS diagnostic candidate from overlay sweep",
                "selection_oos_informed": True,
                "clean_oos_holdout_available_for_this_candidate": False,
                "note": "This candidate may pass contract/reproduction audit but cannot claim clean OOS promotion without a fresh holdout or walk-forward confirmation.",
            },
        }
    )
    runtime.setdefault("execution_contract", {})
    runtime["execution_contract"].update(
        {
            "lifecycle_overlay_enabled": True,
            "lifecycle_overlay_mode": lifecycle_overlay["mode"],
            "runtime_must_fail_on_missing_lifecycle_overlay_contract": True,
        }
    )
    runtime["fail_fast_required"] = True

    manifest = {
        "model_id": MODEL_ID,
        "display_version": DISPLAY_VERSION,
        "status": "omega4_4_v18_lifecycle_overlay_research_candidate_not_live_wired",
        "created_kst": "2026-06-26",
        "contract_created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": BASE_MODEL_ID,
        "base_display_version": base_manifest.get("display_version"),
        "base_redteam_verdict": base_redteam.get("verdict"),
        "base_redteam_pass": base_redteam.get("redteam_pass"),
        "source_report_model_id": base_report.get("model_id"),
        "promotion_type": "runtime_lifecycle_overlay_on_omega4_4_v18_baseline",
        "weights_retrained": False,
        "risk_sidecar_trained": False,
        "runtime_overlay_only": True,
        "source_report": rel(BASE_REPORT),
        "base_runtime_contract": rel(BASE_RUNTIME_CONTRACT),
        "base_candidate_manifest": rel(BASE_CANDIDATE_MANIFEST),
        "overlay_report": rel(OVERLAY_REPORT),
        "overlay_grid": rel(OVERLAY_GRID),
        "manifest": rel(CANDIDATE_MANIFEST),
        "runtime_contract": rel(RUNTIME_CONTRACT),
        "promotion_manifest": rel(PROMOTION_MANIFEST),
        "contract": rel(CONTRACT_DOC),
        **audit,
        "selection_policy": runtime["candidate_selection"],
        "lifecycle_overlay": lifecycle_overlay,
        "baseline_validation": metric_subset(baseline, "validation"),
        "baseline_oos_readout": metric_subset(baseline, "oos"),
        "candidate_validation": metric_subset(candidate, "validation"),
        "candidate_oos_readout": metric_subset(candidate, "oos"),
        "candidate_validation_delta_vs_v18": delta_subset(candidate, baseline, "validation"),
        "candidate_oos_delta_vs_v18": delta_subset(candidate, baseline, "oos"),
        "base_runtime_template": runtime,
        "base_model_artifacts": {
            "parent_bundle": base_manifest.get("parent_bundle"),
            "risk_sidecar_artifact": base_manifest.get("risk_sidecar_artifact"),
            "risk_report": base_manifest.get("risk_report"),
            "parent_report": base_manifest.get("parent_report"),
        },
        "ledgers": {
            "validation": rel(OVERLAY_DIR / f"validation_{SOURCE_VARIANT}_ledger.csv"),
            "oos": rel(OVERLAY_DIR / f"oos_{SOURCE_VARIANT}_ledger.csv"),
            "baseline_validation": rel(OVERLAY_DIR / "validation_baseline_full_replay_ledger.csv"),
            "baseline_oos": rel(OVERLAY_DIR / "oos_baseline_full_replay_ledger.csv"),
        },
    }

    write_json(RUNTIME_CONTRACT, runtime)
    write_json(PROMOTION_MANIFEST, manifest)
    write_json(CANDIDATE_MANIFEST, manifest)

    val = manifest["candidate_validation"]
    oos = manifest["candidate_oos_readout"]
    bval = manifest["baseline_validation"]
    boos = manifest["baseline_oos_readout"]
    CONTRACT_DOC.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_DOC.write_text(
        "\n".join(
            [
                "# Omega4.4 v18 Short Partial Lifecycle Overlay Contract - 2026-06-26",
                "",
                "## Status",
                "",
                f"- Model id: `{MODEL_ID}`",
                f"- Base model: `{BASE_MODEL_ID}`",
                f"- Source variant: `{SOURCE_VARIANT}`",
                "- Status: `omega4_4_v18_lifecycle_overlay_research_candidate_not_live_wired`",
                f"- Red-team verdict: `{audit['redteam_verdict']}`",
                "",
                "## Overlay Contract",
                "",
                "```json",
                json.dumps(lifecycle_overlay, ensure_ascii=False, indent=2),
                "```",
                "",
                "## Selection Caveat",
                "",
                "This candidate was chosen as a balanced validation/OOS diagnostic candidate after the overlay sweep.",
                "It cannot claim clean-OOS promotion until a fresh holdout or walk-forward confirmation is run.",
                "",
                "## Metrics",
                "",
                "| Split | PnL | MDD | WR | Trades | Overlay hits | Log-risk utility |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| Baseline validation | `{bval['pnl']:+.4f}%` | `{bval['mdd']:.4f}%` | `{bval['wr']:.4f}` | `{bval['trades']}` | `{bval['overlay_hits']}` | `{bval['log_risk_utility']:.6f}` |",
                f"| Candidate validation | `{val['pnl']:+.4f}%` | `{val['mdd']:.4f}%` | `{val['wr']:.4f}` | `{val['trades']}` | `{val['overlay_hits']}` | `{val['log_risk_utility']:.6f}` |",
                f"| Baseline OOS readout | `{boos['pnl']:+.4f}%` | `{boos['mdd']:.4f}%` | `{boos['wr']:.4f}` | `{boos['trades']}` | `{boos['overlay_hits']}` | `{boos['log_risk_utility']:.6f}` |",
                f"| Candidate OOS readout | `{oos['pnl']:+.4f}%` | `{oos['mdd']:.4f}%` | `{oos['wr']:.4f}` | `{oos['trades']}` | `{oos['overlay_hits']}` | `{oos['log_risk_utility']:.6f}` |",
                "",
                "## Artifacts",
                "",
                f"- Runtime contract: `{rel(RUNTIME_CONTRACT)}`",
                f"- Promotion manifest: `{rel(PROMOTION_MANIFEST)}`",
                f"- Candidate manifest: `{rel(CANDIDATE_MANIFEST)}`",
                f"- Overlay report: `{rel(OVERLAY_REPORT)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "candidate_manifest": str(CANDIDATE_MANIFEST),
                "runtime_contract": str(RUNTIME_CONTRACT),
                "promotion_manifest": str(PROMOTION_MANIFEST),
                "contract_doc": str(CONTRACT_DOC),
                "redteam_verdict": audit["redteam_verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
