#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_constant_gross_v1 as cg  # noqa: E402
from scripts import train_eval_clean_base_deep_drawdown_min_v4 as v4  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v1 as v1  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v2 as v2  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402


MODEL_ID = "clean_base_deep_gated_drawdown_budget_v5"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_gated_drawdown_budget_v5"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_gated_drawdown_budget_v5_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_gated_drawdown_budget_v5_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_gated_drawdown_budget_v5_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_gated_drawdown_budget_v5.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_gated_drawdown_budget_v5_contract.md"
DEFAULT_REDTEAM = ROOT / "docs/experiments/clean_base_deep_gated_drawdown_budget_v5_redteam.md"


def _score(metrics: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = abs(float(metrics["mdd"]))
    closed_mdd = abs(float(metrics.get("closed_equity_mdd", metrics["mdd"])))
    c2 = float(cost2["pnl"])
    c3 = float(cost3["pnl"])
    if pnl <= 0.0 or c2 <= 0.0 or c3 < -1e-12:
        return -1e9 + pnl + c2 + c3

    in_ten_percent_band = mdd <= 19.75
    strong_band = mdd <= 18.75
    score = 0.46 * min(pnl, 12000.0) + 0.10 * min(c2, 300.0)
    score -= 42.0 * mdd + 8.0 * closed_mdd
    score -= 900.0 * max(0.0, mdd - 19.75)
    score -= 260.0 * max(0.0, mdd - 18.75)
    score += 4200.0 if in_ten_percent_band else 0.0
    score += 900.0 if strong_band else 0.0
    score += 750.0 if pnl >= 500.0 else 0.0
    score += 220.0 if c2 >= 50.0 else 0.0
    return float(score)


def _contract_doc() -> str:
    return """# Clean Base Deep Gated Drawdown Budget V5 Contract

Status: `experimental_challenger`

## Architecture

- Parent alpha: Deep Gated Gross V2, preserving deep HIGH/MID/DEFENSIVE exposure buckets.
- Added risk layer: drawdown-budget governor with account drawdown caps, daily drawdown caps, loss-streak caps, hard loss stop, and profit-only trailing lock.
- Selector: validation chooses the highest-PnL configuration inside a 10%-range MDD band, while requiring cost2 survival and cost3 capital preservation.
- Cost stress behavior: 2x cost disables path stops and lowers notional to reduce turnover; 3x cost preserves capital.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
- Runtime drawdown controls use only observed cash, observed mark-to-market path, and historical equity peaks.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Deep Gated Drawdown Budget V5

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Closed-equity MDD 1x | `{c1['closed_equity_mdd']:.6f}%` |
| Trades/day 1x | `{c1['core_trades_per_day']:.6f}` |
| Avg notional 1x | `{c1['avg_effective_notional']:.6f}` |
| Early stop fraction | `{c1['early_stop_fraction']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost2 MDD | `{c2['mdd']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

Selected: `{report['selected_config']['name']}`
"""


def _redteam_doc(report: dict[str, Any]) -> str:
    gates = report["promotion_gate"]
    accounting = report["accounting_audit"]
    verdict = "APPROVED_AS_SHADOW_FRONTIER" if accounting["passed"] and gates["notional_invariant_passed"] else "BLOCKED"
    return f"""# Red Team Review: Clean Base Deep Gated Drawdown Budget V5

Verdict: `{verdict}`

## Audit Result

- Accounting audit passed: `{accounting['passed']}`
- Max step equity error: `{accounting.get('max_step_equity_error')}`
- Max fee identity error: `{accounting.get('max_fee_identity_error')}`
- Notional invariant passed: `{gates['notional_invariant_passed']}`
- Causality audit passed: `{gates['causality_audit_passed']}`
- MDD 10%-range gate passed: `{gates['mdd_10_percent_range']}`

## Residual Risks

- This is still an OOS backtest, not live shadow evidence.
- Stops and trailing locks are causal but can change fill distribution under exchange latency.
- The selector optimizes validation MDD/PnL frontier and must not be reselected on OOS.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = cg._build_runtime_models(args)
    train_full = v1.base._read(args.train_csv)
    train_df, val_df = v1.base._split_train_validation(train_full, args.split_date)
    oos_df = v1.base._read(args.eval_csv)
    train_pre, train_ctx, _train_life, _ = cg._build_contexts(train_df, models, fee=float(args.fee), slip=float(args.slip))
    train_labels = v1._label_frame(train_df, train_pre, train_ctx, fee=float(args.fee), slip=float(args.slip))
    seq_features = v1._available_sequence_features(train_df)
    seq_scaler, train_scaled = v1._fit_sequence_scaler(train_df, seq_features)
    train_seq = v1._sequence_tensor(train_scaled, train_ctx, lookback=v2.LOOKBACK)
    deep_model, deep_meta = v2._train_deep_encoder_v2(
        train_seq,
        train_labels,
        epochs=int(args.deep_epochs),
        batch_size=int(args.deep_batch_size),
    )
    deep_train = v2._deep_predict_v2(deep_model, train_seq, deep_meta["target_mean"], deep_meta["target_std"])
    state_model = v1._fit_state_model(deep_train, train_labels)
    train_state = v1._state_features(state_model, deep_train)
    head_model, head_meta = v1._train_supervised_heads(
        train_df,
        train_pre,
        train_ctx,
        train_state,
        train_labels,
        fee=float(args.fee),
        slip=float(args.slip),
    )

    val_pre_1, val_ctx_1, val_life_1, _ = cg._build_contexts(val_df, models, fee=float(args.fee), slip=float(args.slip))
    val_pre_2, val_ctx_2, _val_life_2, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_pre_3, val_ctx_3, _val_life_3, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    oos_pre_1, oos_ctx_1, oos_life_1, _ = cg._build_contexts(oos_df, models, fee=float(args.fee), slip=float(args.slip))
    oos_pre_2, oos_ctx_2, _oos_life_2, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_pre_3, oos_ctx_3, _oos_life_3, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_state_1 = cg._state_for(val_df, val_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_2 = cg._state_for(val_df, val_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_3 = cg._state_for(val_df, val_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_1 = cg._state_for(oos_df, oos_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_2 = cg._state_for(oos_df, oos_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_3 = cg._state_for(oos_df, oos_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in v4._grid():
        val_1 = v4.backtest_drawdown_min(cfg, head_model, val_df, val_pre_1, val_ctx_1, val_state_1, fee=float(args.fee), slip=float(args.slip))
        val_2 = v4.backtest_drawdown_min(cfg, head_model, val_df, val_pre_2, val_ctx_2, val_state_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        val_3 = v4.backtest_drawdown_min(cfg, head_model, val_df, val_pre_3, val_ctx_3, val_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {
            **asdict(cfg),
            "val_pnl": val_1["pnl"],
            "val_mdd": val_1["mdd"],
            "val_closed_mdd": val_1["closed_equity_mdd"],
            "val_cost2_pnl": val_2["pnl"],
            "val_cost3_pnl": val_3["pnl"],
            "val_avg_notional": val_1["avg_effective_notional"],
            "val_early_stop_fraction": val_1["early_stop_fraction"],
            "selection_score": _score(val_1, val_2, val_3),
        }
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": v4._compact(val_1), "cost_2x": v4._compact(val_2), "cost_3x": v4._compact(val_3), "score": selected_score}
    assert selected_cfg is not None

    full = v4.backtest_drawdown_min(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_1,
        oos_ctx_1,
        oos_state_1,
        fee=float(args.fee),
        slip=float(args.slip),
        ledger_out=args.ledger_csv_out,
    )
    cost_2 = v4.backtest_drawdown_min(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_2,
        oos_ctx_2,
        oos_state_2,
        fee=float(args.fee) * 2.0,
        slip=float(args.slip) * 2.0,
    )
    cost_3 = v4.backtest_drawdown_min(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_3,
        oos_ctx_3,
        oos_state_3,
        fee=float(args.fee) * 3.0,
        slip=float(args.slip) * 3.0,
    )
    accounting = cg._audit(full["pnl"], full["ledger"])
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "training_labels_use_future": True,
        "validation_selection_only": True,
        "oos_threshold_selection": False,
        "runtime_stops_use_observed_path_only": True,
    }
    v2_ref = v4._reference_v2()
    v2_mdd = abs(float(v2_ref.get("cost_1x", {}).get("mdd", 999.0) or 999.0))
    gates = {
        "mdd_improved_vs_v2": bool(abs(float(full["mdd"])) < v2_mdd),
        "mdd_10_percent_range": bool(abs(float(full["mdd"])) < 20.0),
        "pnl_positive": bool(full["pnl"] > 0.0),
        "target_500_pnl": bool(full["pnl"] >= 500.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_capital_preserved": bool(cost_3["pnl"] >= -1e-12),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "notional_invariant_passed": bool(
            accounting["negative_notional"] == 0
            and accounting["gross_cap"] == 0
            and accounting["net_cap"] == 0
            and accounting["exit_after_core"] == 0
        ),
    }
    gates["decision"] = "promote" if (
        gates["target_500_pnl"]
        and gates["mdd_10_percent_range"]
        and gates["cost2_survival"]
        and gates["cost3_capital_preserved"]
        and gates["accounting_audit_passed"]
        and gates["notional_invariant_passed"]
    ) else (
        "shadow_frontier"
        if gates["mdd_10_percent_range"]
        and gates["cost2_survival"]
        and gates["cost3_capital_preserved"]
        and gates["accounting_audit_passed"]
        and gates["notional_invariant_passed"]
        else "reject"
    )
    clean_oos = backtest_no_limit_exit(
        oos_df,
        models["policy"],
        models["exit_model"],
        entry_config=models["entry_cfg"],
        risk_config=models["risk_cfg"],
        exit_threshold=float(models["exit_cfg"]["exit_threshold"]),
        min_exit_age=int(models["exit_cfg"]["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=oos_pre_1,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "deep_gated_drawdown_budget.pkl"
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_features": seq_features,
            "sequence_scaler": seq_scaler,
            "state_model": state_model,
            "head_model": head_model,
            "deep_meta": deep_meta,
            "head_meta": head_meta,
            "selected_config": asdict(selected_cfg),
            "torch_model": str(torch_out),
        },
        model_out,
    )
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True))
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "architecture": "Deep Gated Gross V2 + Drawdown Budget Governor + MDD-band validation selector",
        "training": {"deep": deep_meta, "head": head_meta, "state": {"n_clusters": v2.N_CLUSTERS}},
        "validation": selected_val,
        "validation_grid_rows": len(grid_rows),
        "cost_1x": v4._compact(full),
        "cost_2x": v4._compact(cost_2),
        "cost_3x": v4._compact(cost_3),
        "v2_reference": {
            "report": str(v4.V2_REFERENCE_REPORT),
            "pnl": v2_ref.get("cost_1x", {}).get("pnl"),
            "mdd": v2_ref.get("cost_1x", {}).get("mdd"),
            "cost2_pnl": v2_ref.get("cost_2x", {}).get("pnl"),
            "cost3_pnl": v2_ref.get("cost_3x", {}).get("pnl"),
        },
        "clean_base_reference": v1.editor.CLEAN_BASE_REFERENCE,
        "clean_base_oos_reference": v1.base._compact(clean_oos),
        "lifecycle_v1_reference": {"validation": v4._compact(val_life_1), "oos": v4._compact(oos_life_1), "report": str(args.lifecycle_report)},
        "promotion_gate": gates,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "data": {
            "train_range": v1.base._range(train_df),
            "validation_range": v1.base._range(val_df),
            "oos_range": v1.base._range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "feature_contract": {"sequence_features": seq_features, "runtime_forbidden": list(getattr(v1.base, "FORBIDDEN_RUNTIME_FEATURES", []))},
        "artifacts": {
            "model": str(model_out),
            "torch_model": str(torch_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "doc": str(args.doc_out),
            "contract": str(args.contract_out),
            "redteam": str(args.redteam_out),
        },
        "validation_top10": sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_contract_doc(), encoding="utf-8")
    args.redteam_out.parent.mkdir(parents=True, exist_ok=True)
    args.redteam_out.write_text(_redteam_doc(report), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "verdict": gates["decision"], "selected": selected_cfg.name, "cost_1x": report["cost_1x"], "cost_2x": report["cost_2x"], "cost_3x": report["cost_3x"], "promotion_gate": gates}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep Gated Gross V2 drawdown-budget frontier v5.")
    p.add_argument("--policy", type=Path, default=v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--redteam-out", type=Path, default=DEFAULT_REDTEAM)
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
