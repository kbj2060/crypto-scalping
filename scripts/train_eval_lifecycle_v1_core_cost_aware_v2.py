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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_clean_base_lifecycle_editor_v1 import (  # noqa: E402
    BASE_REFERENCE,
    DEFAULT_AUDIT,
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT,
    DEFAULT_POLICY,
    DEFAULT_TRAIN_CSV,
    LifecycleRuntimeConfig,
    _base_frame,
    _base_trade_plan,
    _compact,
    _preservation_audit,
    _range,
    _read,
    _sha256,
    _split_train_validation,
    backtest_lifecycle_editor,
)
from scripts.train_eval_clean_base_exit_hazard_recalibrator_v1 import train_bucket_recalibrator  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit, collect_exit_samples  # noqa: E402
from scripts.materialize_clean_base_deep_drawdown_min_noregime_v5_inputs import is_forbidden_column  # noqa: E402


MODEL_ID = "lifecycle_v1_core_cost_aware_v2_20260510"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/lifecycle_v1_core_cost_aware_v2_20260510"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/lifecycle_v1_core_cost_aware_v2_20260510.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/lifecycle_v1_core_cost_aware_v2_20260510_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/lifecycle_v1_core_cost_aware_v2_20260510_ledger.csv"
DEFAULT_AUDIT_OUT = ROOT / "data/ensemble/reports/lifecycle_v1_core_cost_aware_v2_20260510_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/lifecycle_v1_core_cost_aware_v2_20260510_contract.md"
V1_REFERENCE = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"


def _grid(max_notional: float) -> list[LifecycleRuntimeConfig]:
    rows: dict[str, LifecycleRuntimeConfig] = {}
    for shift in (-0.01, 0.02, 0.04):
        for scale, max_delta in ((0.0, 0.0), (0.60, 0.08)):
            for min_age_delta in (3, 6):
                for shrink_margin, shrink_mult in ((999.0, 1.0), (0.08, 0.60)):
                    for boost_margin, boost_mult in ((999.0, 1.0), (0.12, 1.08)):
                        for cap_mult in (0.75, 0.90, 1.00):
                            cap = float(max_notional) * float(cap_mult)
                            name = (
                                f"shift{shift:+.2f}_scale{scale:.2f}_maxd{max_delta:.2f}_"
                                f"agep{min_age_delta}_sh{shrink_margin:.2f}x{shrink_mult:.2f}_"
                                f"bo{boost_margin:.2f}x{boost_mult:.2f}_cap{cap_mult:.2f}"
                            )
                            rows[name] = LifecycleRuntimeConfig(
                                name=name,
                                threshold_shift=float(shift),
                                delta_scale=float(scale),
                                max_delta=float(max_delta),
                                min_age_delta=int(min_age_delta),
                                shrink_margin=float(shrink_margin),
                                shrink_mult=float(shrink_mult),
                                boost_margin=float(boost_margin),
                                boost_mult=float(boost_mult),
                                max_notional=cap,
                            )
    return list(rows.values())


def _score(m1: dict[str, Any], m2: dict[str, Any], m3: dict[str, Any]) -> float:
    return float(m1["pnl"]) + 0.28 * float(m2["pnl"]) + 0.22 * float(m3["pnl"]) - 1.6 * abs(float(m1["mdd"])) - 10.0 * max(0.0, 5.5 - float(m1["trades_per_day"]))


def _ledger(path: Path, plan: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in plan for k in row.keys()}) if plan else ["entry_idx"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(plan)


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    feature_hits = [c for c in list(policy.get("feature_cols", [])) if is_forbidden_column(c)] if isinstance(policy, dict) else []

    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    eval_df = _read(args.eval_csv)
    val_pre = _base_frame(val_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)

    x, y, sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=int(args.entry_stride),
        min_age=int(args.min_age),
        max_age=int(args.max_age),
        age_stride=int(args.age_stride),
        future_horizon=int(args.future_horizon),
        exit_edge=float(args.exit_edge),
        adverse_gap=float(args.adverse_gap),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    recalibrator = train_bucket_recalibrator(x, y)

    base_val = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=val_pre)
    base_oos = backtest_no_limit_exit(eval_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=eval_pre)

    val_base = {
        mult: _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
        for mult in (1.0, 2.0, 3.0)
    }
    eval_base = {
        mult: _base_trade_plan(eval_df, exit_model, risk_cfg, exit_cfg, eval_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
        for mult in (1.0, 2.0, 3.0)
    }

    rows: list[dict[str, Any]] = []
    best = None
    best_score = -1e18
    for cfg in _grid(float(risk_cfg.get("max_notional", 3.6))):
        m1 = backtest_lifecycle_editor(val_df, exit_model, recalibrator, cfg, val_base[1.0], exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
        m2 = backtest_lifecycle_editor(val_df, exit_model, recalibrator, cfg, val_base[2.0], exit_cfg, val_pre, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        m3 = backtest_lifecycle_editor(val_df, exit_model, recalibrator, cfg, val_base[3.0], exit_cfg, val_pre, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        score = _score(m1, m2, m3)
        row = {"config": cfg, "validation_1x": _compact(m1), "validation_2x": _compact(m2), "validation_3x": _compact(m3), "score": score}
        rows.append(row)
        if score > best_score:
            best_score = score
            best = row
    assert best is not None
    cfg = best["config"]
    oos_full: dict[float, dict[str, Any]] = {}
    for mult in (1.0, 2.0, 3.0):
        oos_full[mult] = backtest_lifecycle_editor(eval_df, exit_model, recalibrator, cfg, eval_base[mult], exit_cfg, eval_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
    oos1 = _compact(oos_full[1.0])
    oos2 = _compact(oos_full[2.0])
    oos3 = _compact(oos_full[3.0])
    _ledger(args.ledger_out, oos_full[1.0].get("lifecycle_plan", []))
    invariant = _preservation_audit(eval_base[1.0], oos_full[1.0].get("lifecycle_plan", []))

    v1 = json.loads(args.v1_reference.read_text(encoding="utf-8"))
    gates = {
        "pnl_vs_lifecycle_v1": float(oos1["pnl"]) > float(v1["cost_1x"]["pnl"]),
        "mdd_vs_lifecycle_v1": abs(float(oos1["mdd"])) <= abs(float(v1["cost_1x"]["mdd"])),
        "cost2_survival": float(oos2["pnl"]) > 0.0,
        "cost3_survival": float(oos3["pnl"]) > 0.0,
        "trades_per_day_min5_5": float(oos1["trades_per_day"]) >= 5.5,
        "preservation_audit_passed": bool(invariant.get("passed")),
    }
    gates["decision"] = "promote" if all(gates.values()) else ("shadow_candidate" if float(oos1["pnl"]) > 0 and gates["cost2_survival"] else "reject")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "lifecycle_core_cost_aware.pkl"
    joblib.dump(
        {
            "type": MODEL_ID,
            "recalibrator": recalibrator,
            "selected_runtime_config": asdict(cfg),
            "sample_meta": sample_meta,
            "base_policy": str(args.policy),
            "base_exit_governor": str(args.exit_model),
            "entry_config": entry_cfg,
            "risk_config": risk_cfg,
            "exit_config": exit_cfg,
            "validation_selection": "1x/2x/3x cost-aware validation score",
        },
        model_out,
    )

    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "threshold_shift", "delta_scale", "max_delta", "min_age_delta", "shrink_margin", "shrink_mult", "boost_margin", "boost_mult", "max_notional", "score", "val1_pnl", "val1_mdd", "val2_pnl", "val3_pnl", "val1_trades_per_day"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
            c = asdict(row["config"])
            writer.writerow({**{k: c[k] for k in fieldnames if k in c}, "score": row["score"], "val1_pnl": row["validation_1x"]["pnl"], "val1_mdd": row["validation_1x"]["mdd"], "val2_pnl": row["validation_2x"]["pnl"], "val3_pnl": row["validation_3x"]["pnl"], "val1_trades_per_day": row["validation_1x"]["trades_per_day"]})

    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "design": "Lifecycle V1 Core Cost-Aware V2 keeps base entry timing/side/cooldown but reselects lifecycle runtime edits with a validation score that includes 1x/2x/3x fee/slippage survival.",
        "selected_config": asdict(cfg),
        "cost_1x": oos1,
        "cost_2x": oos2,
        "cost_3x": oos3,
        "promotion_gate": gates,
        "lifecycle_v1_reference": {"cost_1x": v1["cost_1x"], "cost_2x": v1["cost_2x"], "cost_3x": v1["cost_3x"]},
        "base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(base_val),
        "clean_base_oos_reference": _compact(base_oos),
        "feature_audit": {"underlying_policy_forbidden_feature_hits": feature_hits, "passed_clean_noregime": not feature_hits},
        "preservation_audit": invariant,
        "split": {"train_range": _range(train_df), "validation_range": _range(val_df), "eval_range": _range(eval_df), "train_rows": len(train_df), "validation_rows": len(val_df), "eval_rows": len(eval_df)},
        "training": {"sample_meta": sample_meta, "global_hazard_rate": recalibrator["global_hazard_rate"], "bucket_count": len(recalibrator["buckets"])},
        "validation_top10": [{"runtime_config": asdict(r["config"]), "score": r["score"], "validation_1x": r["validation_1x"], "validation_2x": r["validation_2x"], "validation_3x": r["validation_3x"]} for r in sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:10]],
        "artifacts": {"model": str(model_out), "report": str(args.report_out), "grid": str(args.grid_out), "ledger": str(args.ledger_out), "audit": str(args.audit_out), "contract": str(args.contract_out)},
        "frozen_artifacts": {"base_policy": str(args.policy), "base_policy_sha256": _sha256(args.policy), "base_exit_governor": str(args.exit_model), "base_exit_governor_sha256": _sha256(args.exit_model)},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_doc = {
        "model_id": MODEL_ID,
        "status": "pass" if invariant.get("passed") and float(oos1["trades"]) == len(oos_full[1.0].get("lifecycle_plan", [])) else "fail",
        "checks": {
            "entry_timing_side_cooldown_preserved": bool(invariant.get("passed")),
            "cost_parity_available": all(k in report for k in ("cost_1x", "cost_2x", "cost_3x")),
            "ledger_rows_match_trades": int(oos1["trades"]) == len(oos_full[1.0].get("lifecycle_plan", [])),
            "underlying_policy_forbidden_features_absent": not feature_hits,
            "live_injection_status": "not injected into trading_bot.py",
        },
        "issues": ([{"severity": "WARN", "issue": "underlying Lifecycle V1 policy still contains quarantined/forbidden feature names", "columns": feature_hits}] if feature_hits else []),
        "metrics": {"cost_1x": oos1, "cost_2x": oos2, "cost_3x": oos3, "promotion_gate": gates},
        "red_team_conclusion": "PASS for lifecycle preservation/accounting artifact integrity. Clean no-regime audit remains WARN if the frozen Lifecycle V1 base policy contains forbidden feature names.",
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text("# Lifecycle V1 Core Cost-Aware V2 Contract\n\nFrozen Lifecycle V1 base entry timing, side, and cooldown are preserved. Runtime lifecycle edits are reselected on validation with 1x/2x/3x fee/slippage cost stress. This artifact is not clean no-regime if the frozen base policy feature list contains quarantined regime-derived columns.\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "verdict": report["verdict"], "selected_config": report["selected_config"], "cost_1x": oos1, "cost_2x": oos2, "cost_3x": oos3, "promotion_gate": gates}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lifecycle V1 core cost-aware runtime config upgrade.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--v1-reference", type=Path, default=V1_REFERENCE)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--entry-stride", type=int, default=36)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-age", type=int, default=144)
    p.add_argument("--age-stride", type=int, default=24)
    p.add_argument("--future-horizon", type=int, default=72)
    p.add_argument("--exit-edge", type=float, default=0.0015)
    p.add_argument("--adverse-gap", type=float, default=0.012)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT_OUT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
