#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega4_4_v18_baseline_20260624"
DISPLAY_VERSION = "Omega 4.4 v18 live-like dynamic leverage baseline"
RUN_DIR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
REPORT = RUN_DIR / "report.json"
PROMOTION_MANIFEST = RUN_DIR / "promotion_manifest.json"
RUNTIME_CONTRACT = RUN_DIR / "runtime_contract.json"
CANDIDATE_DIR = ROOT / f"data/ensemble/supervised/{MODEL_ID}"
CANDIDATE_MANIFEST = CANDIDATE_DIR / "candidate_manifest.json"
CONTRACT_DOC = ROOT / f"docs/model_contracts/{MODEL_ID}_contract.md"
AUDIT_JSON = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_redteam_audit_20260625/report.json"
AUDIT_DOC = ROOT / "docs/audits/omega4_4_v18_redteam_audit_20260625.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "wr",
        "trades_per_day",
        "avg_notional",
        "avg_margin_fraction",
        "avg_leverage",
        "log_growth_sum",
        "tail_excess_sum",
        "liquidation_excess_sum",
        "log_risk_utility",
        "long_entries",
        "short_entries",
        "exit_reasons",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def audit_fields() -> dict[str, Any]:
    if not AUDIT_JSON.exists():
        return {
            "redteam_report": rel(AUDIT_DOC),
            "redteam_json": rel(AUDIT_JSON),
            "redteam_verdict": "REDTEAM_PENDING",
            "redteam_pass": False,
        }
    audit = read_json(AUDIT_JSON)
    return {
        "redteam_report": rel(AUDIT_DOC),
        "redteam_json": rel(AUDIT_JSON),
        "redteam_verdict": audit.get("verdict"),
        "redteam_pass": audit.get("redteam_pass"),
        "research_reproduction_pass": audit.get("research_reproduction_pass"),
        "promotion_pass": audit.get("promotion_pass"),
    }


def main() -> int:
    report = read_json(REPORT)
    selected = report["selected"]
    risk_model = report["risk_model"]
    contract = report["contract"]
    audit = audit_fields()

    parent_bundle = Path(report["baseline_bundle"])
    runtime = {
        "model_id": MODEL_ID,
        "display_version": DISPLAY_VERSION,
        "role": "omega4_4_v18_live_like_dynamic_leverage_baseline",
        "base_model_id": report["base_model"],
        "source_report_model_id": report["model_id"],
        "parent_bundle": rel(parent_bundle),
        "parent_report": rel(parent_bundle.parent / "report.json"),
        "risk_sidecar_artifact": rel(RUN_DIR / "risk_sidecar.pkl"),
        "risk_report": rel(REPORT),
        "quality_threshold": contract["quality_threshold"],
        "exit_threshold": contract["exit_threshold"],
        "max_hold_bars": 0,
        "cooldown_bars": 0,
        "topdown_sequence": {
            "parent_recipe": "same_as_direction + terminal_giveback + epochs2 + train15k + exit15k + q0.70",
            "exit_threshold": contract["exit_threshold"],
            "risk_sidecar": "HGB side-split parent_outputs live_exposure_grid dynamic_leverage validation-only logrisk tail050 minavg075",
        },
        "atr_safety_sltp": {
            "atr_window_bars": contract["atr_window"],
            "take_profit_atr_multiple": contract["take_profit_atr_multiple"],
            "stop_loss_atr_multiple": contract["stop_loss_atr_multiple"],
            "floor_take_profit_price_move": contract["floor_take_profit_price_move"],
            "floor_stop_loss_price_move": contract["floor_stop_loss_price_move"],
            "cap_take_profit_price_move": contract["cap_take_profit_price_move"],
            "cap_stop_loss_price_move": contract["cap_stop_loss_price_move"],
            "sltp_hit_contract": contract["sltp"],
        },
        "price_move_take_profit_expr": "clip(max(0.075, atr_pct_192 * 12.0), 0.0, 0.22)",
        "price_move_stop_loss_expr": "clip(max(0.040, atr_pct_192 * 6.0), 0.0, 0.12)",
        "risk_sidecar": {
            "mode": "full_replay_trade_level_overlay",
            "model_kind": risk_model["model_kind"],
            "feature_mode": risk_model["risk_feature_mode"],
            "side_split_model": risk_model["side_split_model"],
            "dynamic_leverage": risk_model["dynamic_leverage"],
            "selection_objective": risk_model["selection_objective"],
            "selection_scope": risk_model["selection_scope"],
            "live_exposure_grid": risk_model["live_exposure_grid"],
            "min_validation_avg_notional": risk_model["min_validation_avg_notional"],
            "max_validation_avg_notional": risk_model["max_validation_avg_notional"],
            "log_risk_params": risk_model["log_risk_params"],
            "selected_variant": selected["variant"],
            "selected_mapping": selected["mapping"],
            "outputs": ["margin_fraction", "leverage"],
            "notional_contract": contract["risk_sizing"],
        },
        "pnl_contract": "PnL = realized_price_move * notional, after fee/slippage cost multiplier",
        "sltp_contract": contract["sltp"],
        "notional_scaled_sltp": contract["notional_scaled_sltp"],
        "execution_contract": {
            "parent_model_owns": [
                "direction",
                "quality_gate",
                "exit_head",
                "entry_time_atr_sltp_barrier_timing",
            ],
            "risk_sidecar_owns": [
                "entry_time_margin_fraction",
                "entry_time_leverage",
                "trade_pnl_sizing",
            ],
            "full_replay_dynamic_exit_enabled": True,
            "exit_sizing_input_mode": risk_model.get("exit_sizing_input_mode", "actual"),
            "runtime_must_fail_on_missing_sidecar_or_contract_mismatch": True,
        },
        "fail_fast_required": True,
    }

    common_manifest = {
        "model_id": MODEL_ID,
        "display_version": DISPLAY_VERSION,
        "status": "omega4_4_v18_research_baseline_not_live_wired",
        "created_kst": "2026-06-24",
        "contract_created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_role": "Omega4.4 v18 high-exposure live-like dynamic leverage research baseline; not live wired.",
        "promotion_type": "topdown_parent_plus_validation_only_logrisk_live_exposure_margin_leverage_sidecar_full_replay_contract",
        "weights_retrained": True,
        "risk_sidecar_trained": True,
        "base_model_id": report["base_model"],
        "source_report_model_id": report["model_id"],
        "source_report": rel(REPORT),
        "contract": rel(CONTRACT_DOC),
        "manifest": rel(CANDIDATE_MANIFEST),
        "runtime_contract": rel(RUNTIME_CONTRACT),
        "promotion_manifest": rel(PROMOTION_MANIFEST),
        "parent_report": rel(parent_bundle.parent / "report.json"),
        "parent_bundle": rel(parent_bundle),
        "risk_report": rel(REPORT),
        "risk_mapping_ranking": rel(RUN_DIR / "risk_mapping_ranking.csv"),
        "risk_sidecar_artifact": rel(RUN_DIR / "risk_sidecar.pkl"),
        **audit,
        "selected_variant": selected["variant"],
        "selection_policy": selected["selection_rule"],
        "selection_scope": risk_model["selection_scope"],
        "oos_usage_policy": "OOS excluded from mapping filter/sort/tie-break; OOS is selected-row readout only.",
        "promoted_readout": "full_replay",
        "topdown_sequence": runtime["topdown_sequence"],
        "risk_model": {
            "model_kind": risk_model["model_kind"],
            "risk_feature_mode": risk_model["risk_feature_mode"],
            "side_split_model": risk_model["side_split_model"],
            "score_quality_blend": risk_model["score_quality_blend"],
            "dynamic_leverage": risk_model["dynamic_leverage"],
            "require_dynamic_leverage_mapping": risk_model["require_dynamic_leverage_mapping"],
            "selection_objective": risk_model["selection_objective"],
            "selection_scope": risk_model["selection_scope"],
            "live_exposure_grid": risk_model["live_exposure_grid"],
            "min_validation_avg_notional": risk_model["min_validation_avg_notional"],
            "max_validation_avg_notional": risk_model["max_validation_avg_notional"],
            "log_risk_params": risk_model["log_risk_params"],
            "notional_scaled_sltp": risk_model["notional_scaled_sltp"],
        },
        "risk_label": report["risk_label"],
        "selected_mapping": selected["mapping"],
        "runtime_template": runtime,
        "sizing_only_validation": metric_subset(selected["validation"]),
        "sizing_only_oos_readout": metric_subset(selected["oos"]),
        "full_replay_validation": metric_subset(selected["selected_full_replay"]["validation"]),
        "full_replay_oos_readout": metric_subset(selected["selected_full_replay"]["oos"]),
        "charts": {
            "validation": rel(RUN_DIR / "charts/omega44_live_like_dynamic_leverage_v18_validation_trade_chart.png"),
            "oos": rel(RUN_DIR / "charts/omega44_live_like_dynamic_leverage_v18_oos_trade_chart.png"),
        },
    }

    write_json(RUNTIME_CONTRACT, runtime)
    write_json(PROMOTION_MANIFEST, common_manifest)
    write_json(CANDIDATE_MANIFEST, common_manifest)

    full_val = common_manifest["full_replay_validation"]
    full_oos = common_manifest["full_replay_oos_readout"]
    sizing_val = common_manifest["sizing_only_validation"]
    sizing_oos = common_manifest["sizing_only_oos_readout"]
    CONTRACT_DOC.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_DOC.write_text(
        "\n".join(
            [
                "# Omega4.4 v18 Baseline Contract - 2026-06-24",
                "",
                "## Status",
                "",
                f"- Model id: `{MODEL_ID}`",
                f"- Display version: `{DISPLAY_VERSION}`",
                "- Status: `omega4_4_v18_research_baseline_not_live_wired`",
                "- Role: high-exposure live-like dynamic leverage research baseline",
                f"- Red-team verdict: `{audit['redteam_verdict']}`",
                "",
                "## Lineage",
                "",
                "1. Parent: top-down best parent, q0.70, exit threshold 0.75.",
                "2. Risk sidecar: HGB, side-split, parent-output features, dynamic leverage.",
                "3. Selection: validation-only log-risk with validation MDD >= -16.00 and validation average notional in [0.75, 0.90].",
                "4. OOS is readout only and is not used for mapping selection.",
                "",
                "## Runtime Contract",
                "",
                "```text",
                "notional = margin_fraction * leverage",
                "PnL = realized_price_move * notional",
                "SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines",
                "```",
                "",
                "- Quality threshold: `0.70`",
                "- Exit threshold: `0.75`",
                "- Max hold bars: `0`",
                "- Cooldown bars: `0`",
                "- Full dynamic-risk exit replay: promoted readout, `exit_sizing_input_mode=actual`",
                "",
                "## Selected Risk Mapping",
                "",
                "```json",
                json.dumps(selected["mapping"], ensure_ascii=False, indent=2),
                "```",
                "",
                "## Metrics",
                "",
                "| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Leverage | Log-Risk Utility |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| Validation sizing-only | `{sizing_val['pnl']:+.2f}%` | `{sizing_val['mdd']:.2f}%` | `{sizing_val['wr']:.2%}` | `{sizing_val['trades']}` | `{sizing_val['avg_notional']:.4f}` | `{sizing_val['avg_margin_fraction']:.4f}` | `{sizing_val['avg_leverage']:.4f}` | `{sizing_val['log_risk_utility']:.6f}` |",
                f"| OOS sizing-only readout | `{sizing_oos['pnl']:+.2f}%` | `{sizing_oos['mdd']:.2f}%` | `{sizing_oos['wr']:.2%}` | `{sizing_oos['trades']}` | `{sizing_oos['avg_notional']:.4f}` | `{sizing_oos['avg_margin_fraction']:.4f}` | `{sizing_oos['avg_leverage']:.4f}` | `{sizing_oos['log_risk_utility']:.6f}` |",
                f"| Validation full replay | `{full_val['pnl']:+.2f}%` | `{full_val['mdd']:.2f}%` | `{full_val['wr']:.2%}` | `{full_val['trades']}` | `{full_val['avg_notional']:.4f}` | `{full_val['avg_margin_fraction']:.4f}` | `{full_val['avg_leverage']:.4f}` | `{full_val['log_risk_utility']:.6f}` |",
                f"| OOS full replay readout | `{full_oos['pnl']:+.2f}%` | `{full_oos['mdd']:.2f}%` | `{full_oos['wr']:.2%}` | `{full_oos['trades']}` | `{full_oos['avg_notional']:.4f}` | `{full_oos['avg_margin_fraction']:.4f}` | `{full_oos['avg_leverage']:.4f}` | `{full_oos['log_risk_utility']:.6f}` |",
                "",
                "## Artifacts",
                "",
                f"- Runtime contract: `{rel(RUNTIME_CONTRACT)}`",
                f"- Promotion manifest: `{rel(PROMOTION_MANIFEST)}`",
                f"- Candidate manifest: `{rel(CANDIDATE_MANIFEST)}`",
                f"- Source report: `{rel(REPORT)}`",
                f"- Risk sidecar: `{rel(RUN_DIR / 'risk_sidecar.pkl')}`",
                f"- Parent bundle: `{rel(parent_bundle)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "promotion_manifest": str(PROMOTION_MANIFEST),
                "runtime_contract": str(RUNTIME_CONTRACT),
                "candidate_manifest": str(CANDIDATE_MANIFEST),
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
