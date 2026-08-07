#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_muzero_az_vs_dt_lifecycle_2026 import (  # noqa: E402
    _build_zero_style_current,
    _date_range,
    _run,
    _slice_precomputed,
)
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_AZ_EXIT_MODEL  # noqa: E402
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_muzero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL, _load_az_exit  # noqa: E402
from scripts.train_eval_zero_style_remaining_layers_2026 import _load_mz_risk, _load_pv  # noqa: E402
from scripts.train_eval_zero_style_risk_overlay_2026 import DEFAULT_AZ_RISK_OUT, RISK_ACTIONS, _load_mz_entry  # noqa: E402


MODEL_ID = "current_top_muzero_az_stage2_azexit_2026"
DEFAULT_RANK1_STAGE2_MZ = ROOT / "data/ensemble/supervised/zero_style/remaining_layers_walkforward/wf_stage2_sleeve_mz.pt"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/current_rank1_baseline_red_team_audit_2026.json"
BASELINE_TARGET = {
    "pnl": 752.648580357841,
    "mdd": -18.755787211251405,
    "trades": 353,
    "trades_per_day": 6.017045454545455,
    "avg_leverage": 1.5960290252000644,
}


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compare_metrics(observed: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    tolerance = {"pnl": 1e-6, "mdd": 1e-6, "trades": 0.0, "trades_per_day": 1e-9, "avg_leverage": 1e-9}
    diffs = {k: float(observed.get(k, np.nan)) - float(v) for k, v in target.items()}
    passed = all(abs(diffs[k]) <= tolerance[k] for k in target)
    return {"passed": bool(passed), "target": target, "observed": {k: observed.get(k) for k in target}, "diff": diffs, "tolerance": tolerance}


def _decision_audit(dec: pd.DataFrame, *, max_notional: float, leverage_cap: float) -> dict[str, Any]:
    action = pd.to_numeric(dec.get("action", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec.get("side", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    notional = pd.to_numeric(dec.get("notional_exposure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(dec.get("leverage", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    pf = pd.to_numeric(dec.get("position_fraction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    cooldown = pd.to_numeric(dec.get("cooldown_bars", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    active = (action != 0) & (side != 0) & (notional > 0.0)
    violations = {
        "nonfinite_values": int((~np.isfinite(notional) | ~np.isfinite(leverage) | ~np.isfinite(pf)).sum()),
        "negative_notional": int((notional < -1e-12).sum()),
        "leverage_below_one_active": int((active & (leverage < 1.0 - 1e-12)).sum()),
        "leverage_above_cap": int((active & (leverage > float(leverage_cap) + 1e-12)).sum()),
        "notional_above_max": int((active & (notional > float(max_notional) + 1e-12)).sum()),
        "active_action_side_mismatch": int((((action != 0) ^ (side != 0)) & (notional > 1e-12)).sum()),
        "cash_has_exposure": int(((action == 0) & ((side != 0) | (notional > 1e-12) | (pf > 1e-12))).sum()),
        "position_fraction_mismatch": int((active & (np.abs(pf - notional / np.maximum(leverage, 1e-12)) > 1e-9)).sum()),
        "negative_cooldown": int((cooldown < -1e-12).sum()),
    }
    active_notional = notional[active]
    active_lev = leverage[active]
    return {
        "passed": bool(sum(violations.values()) == 0),
        "rows": int(len(dec)),
        "active_rows": int(active.sum()),
        "cash_rows": int((~active).sum()),
        "long_rows": int((active & (side > 0)).sum()),
        "short_rows": int((active & (side < 0)).sum()),
        "violations": violations,
        "notional": {
            "max": float(active_notional.max()) if active_notional.size else 0.0,
            "mean": float(active_notional.mean()) if active_notional.size else 0.0,
            "p95": float(np.quantile(active_notional, 0.95)) if active_notional.size else 0.0,
        },
        "leverage": {
            "max": float(active_lev.max()) if active_lev.size else 0.0,
            "mean": float(active_lev.mean()) if active_lev.size else 0.0,
            "p95": float(np.quantile(active_lev, 0.95)) if active_lev.size else 0.0,
        },
    }


def _period_breakdown(
    label: str,
    freq: str,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    if "timestamp" not in df.columns:
        return {}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    periods = sorted(ts.dropna().dt.to_period(freq).unique())
    rows: dict[str, Any] = {}
    for period in periods:
        mask = (ts.dt.to_period(freq) == period).to_numpy(dtype=bool)
        if not mask.any():
            continue
        sub = df.loc[mask].reset_index(drop=True)
        pre = _slice_precomputed(precomputed, mask)
        rows[str(period)] = _run(label, sub, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, pre, fee=fee, slip=slip)["eval"]
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red Team audit for current rank-1 MuZero/AZ Stage2 baseline.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-risk-model", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--stage2-mz-model", type=Path, default=DEFAULT_RANK1_STAGE2_MZ)
    p.add_argument("--az-exit-model", type=Path, default=DEFAULT_AZ_EXIT_MODEL)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    entry_cfg, risk_cfg, exit_cfg_raw = _load_selected(args.selection_report)
    max_notional = float(risk_cfg.get("max_notional", entry_cfg.get("max_notional", 3.6)))
    entry_cfg = dict(entry_cfg)
    risk_cfg = dict(risk_cfg)
    entry_cfg["max_notional"] = max_notional
    risk_cfg["max_notional"] = max_notional
    exit_cfg = {"exit_threshold": 0.45, "min_exit_age": int(exit_cfg_raw["min_exit_age"])}

    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    mz_entry = _load_mz_entry(args.mz_entry_model, device)
    az_risk = _load_pv(args.az_risk_model, len(RISK_ACTIONS), RISK_ACTIONS, device)
    stage2_mz = _load_mz_risk(args.stage2_mz_model, device)
    az_exit = _load_az_exit(args.az_exit_model, device)
    if az_exit is None:
        raise FileNotFoundError(f"AZ exit model not found: {args.az_exit_model}")

    precomputed = _build_zero_style_current(
        eval_df,
        policy,
        entry_cfg,
        mz_entry=mz_entry,
        az_risk=az_risk,
        mz_risk=stage2_mz,
        device=device,
        max_notional=max_notional,
        leverage_cap=5.0,
        stage2_gamma=0.55,
        stage2_prior=0.0,
        stage2_depth=1,
        stage2_score_floor=0.12,
    )
    _, decisions, _, _ = precomputed
    baseline = _run(MODEL_ID, eval_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=args.fee, slip=args.slip, monthly=True)
    cost_stress = {}
    for mult in (1.0, 2.0, 3.0, 4.0, 5.0):
        cost_stress[f"cost_{mult:g}x"] = _run(
            f"{MODEL_ID}_cost_{mult:g}x",
            eval_df,
            policy,
            az_exit,
            entry_cfg,
            risk_cfg,
            exit_cfg,
            precomputed,
            fee=args.fee * mult,
            slip=args.slip * mult,
        )
    fee_slip_decomp = {
        "fee_2x_slip_1x": _run("fee_2x_slip_1x", eval_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=args.fee * 2.0, slip=args.slip)["eval"],
        "fee_1x_slip_2x": _run("fee_1x_slip_2x", eval_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=args.fee, slip=args.slip * 2.0)["eval"],
        "fee_3x_slip_1x": _run("fee_3x_slip_1x", eval_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=args.fee * 3.0, slip=args.slip)["eval"],
        "fee_1x_slip_3x": _run("fee_1x_slip_3x", eval_df, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=args.fee, slip=args.slip * 3.0)["eval"],
    }
    weekly = _period_breakdown("weekly", "W-SUN", eval_df, precomputed, policy, az_exit, entry_cfg, risk_cfg, exit_cfg, fee=args.fee, slip=args.slip)
    monthly = baseline.get("monthly", {})

    artifact_paths = {
        "policy": args.policy,
        "selection_report": args.selection_report,
        "train_csv": args.train_csv,
        "eval_csv": args.eval_csv,
        "mz_entry_model": args.mz_entry_model,
        "az_risk_model": args.az_risk_model,
        "rank1_stage2_mz_model": args.stage2_mz_model,
        "az_exit_model": args.az_exit_model,
    }
    artifact_audit = {k: {"path": str(v), "exists": bool(v.exists()), "sha256": _sha256(v)} for k, v in artifact_paths.items()}
    active_paths = [str(args.mz_entry_model), str(args.az_risk_model), str(args.stage2_mz_model), str(args.az_exit_model)]
    stage3_stage4_excluded = all("stage3" not in p.lower() and "stage4" not in p.lower() for p in active_paths)

    reproduction = _compare_metrics(baseline["eval"], BASELINE_TARGET)
    decision_audit = _decision_audit(decisions, max_notional=max_notional, leverage_cap=5.0)
    source_audit = _audit(args.train_csv, args.eval_csv, policy)

    cost_eval = {k: v["eval"] for k, v in cost_stress.items()}
    weekly_pnls = [float(v["pnl"]) for v in weekly.values()]
    monthly_pnls = [float(v["pnl"]) for v in monthly.values()]
    blocking_findings = []
    if not reproduction["passed"]:
        blocking_findings.append("baseline_reproduction_failed")
    if not decision_audit["passed"]:
        blocking_findings.append("decision_invariant_failed")
    if not stage3_stage4_excluded:
        blocking_findings.append("stage3_stage4_artifact_in_active_path")
    if float(cost_eval["cost_3x"]["pnl"]) <= 0.0:
        blocking_findings.append("cost3x_pnl_not_positive")
    if float(cost_eval["cost_5x"]["pnl"]) <= 0.0:
        blocking_findings.append("cost5x_pnl_not_positive")

    safety_limitations = [
        "funding_cost_not_applied_in_accounting",
        "liquidation_and_maintenance_margin_not_modeled",
        "intra_position_resize_or_reverse_not_modeled_by_backtest_no_limit_exit",
        "single_2026_oos_window_only_no_walk_forward_confidence_interval",
        "orderbook_depth_market_impact_not_modeled_beyond_constant_slippage",
        "trade_level_ledger_and_equity_curve_not_returned_by_backtest_function",
    ]
    report = {
        "type": "current_rank1_baseline_red_team_audit_2026",
        "model_id": MODEL_ID,
        "verdict": "shadow_only_not_live_safe" if safety_limitations else "safe",
        "blocking_findings": blocking_findings,
        "safety_limitations": safety_limitations,
        "audit_config": {
            "fee": float(args.fee),
            "slip": float(args.slip),
            "stage2": {"gamma": 0.55, "prior": 0.0, "depth": 1, "score_floor": 0.12},
            "exit": exit_cfg,
            "risk": risk_cfg,
            "entry": entry_cfg,
            "device": device,
        },
        "artifact_audit": artifact_audit,
        "stage3_stage4_exclusion": {"passed": bool(stage3_stage4_excluded), "active_paths": active_paths},
        "data_audit": {
            "source_audit": source_audit,
            "train_range": _date_range(train_df),
            "eval_range": _date_range(eval_df),
            "train_rows": int(len(train_df)),
            "eval_rows": int(len(eval_df)),
        },
        "baseline_reproduction": reproduction,
        "baseline_eval": baseline,
        "decision_audit": decision_audit,
        "cost_stress": cost_eval,
        "fee_slip_decomposition": fee_slip_decomp,
        "period_breakdown": {
            "monthly": monthly,
            "weekly": weekly,
            "monthly_min_pnl": float(min(monthly_pnls)) if monthly_pnls else None,
            "weekly_min_pnl": float(min(weekly_pnls)) if weekly_pnls else None,
            "weekly_negative_count": int(sum(p < 0.0 for p in weekly_pnls)),
        },
        "red_team_checks": {
            "baseline_reproduction_required": True,
            "decision_invariants_required": True,
            "cost_1x_2x_3x_required": True,
            "stage3_stage4_excluded_required": True,
            "funding_liquidation_margin_required_for_live": True,
            "trade_ledger_required_for_live": True,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "verdict": report["verdict"],
                "blocking_findings": blocking_findings,
                "baseline": baseline["eval"],
                "cost_stress": {k: v["pnl"] for k, v in cost_eval.items()},
                "decision_invariants_passed": decision_audit["passed"],
                "stage3_stage4_excluded": stage3_stage4_excluded,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
