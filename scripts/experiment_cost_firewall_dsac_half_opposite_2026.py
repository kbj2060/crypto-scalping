#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_dsac_priority1_entry_arbiter_2026 import (  # noqa: E402
    Candidate,
    DSAC_STATE_DIM,
    _attach_context,
    _audit_selected_ledger,
    _flat_dsac_signals,
    _price_arrays,
    _read_features,
    _read_ledger,
    _safe_float,
    _score,
    replay,
)


MODEL_ID = "cost_firewall_dsac_half_opposite_2026"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_ledger.csv"
DEFAULT_FEATURE_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_DSAC_CKPT = ROOT / "data/ensemble/ckpt/dsac_priority1_full_retrain_20260507/best_dsac_agents.pth"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/cost_firewall_dsac_half_opposite_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/cost_firewall_dsac_half_opposite_2026_grid.csv"
DEFAULT_LEDGER_OUT = ROOT / "data/ensemble/reports/cost_firewall_dsac_half_opposite_2026_selected_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/cost_firewall_dsac_half_opposite_2026_audit.json"


def build_candidates() -> list[Candidate]:
    out: list[Candidate] = [
        Candidate("noop_dgg_v2_replay"),
        Candidate("cost_firewall_buf_0p0010", cost_buffer=0.0010),
        Candidate("cost_firewall_buf_0p0020", cost_buffer=0.0020),
        Candidate("cost_firewall_buf_0p0035", cost_buffer=0.0035),
        Candidate(
            "dsac_half_opposite_0p30",
            opposite_threshold=0.30,
            opposite_action="half",
        ),
        Candidate(
            "dsac_half_opposite_0p50",
            opposite_threshold=0.50,
            opposite_action="half",
        ),
        Candidate(
            "dsac_veto_opposite_0p50",
            opposite_threshold=0.50,
            opposite_action="veto",
        ),
    ]
    for buf in (0.0010, 0.0020, 0.0035):
        for th in (0.30, 0.50, 0.70):
            out.append(
                Candidate(
                    name=f"cost_firewall_{buf:.4f}_dsac_half_opp_{th:.2f}".replace(".", "p"),
                    cost_buffer=buf,
                    opposite_threshold=th,
                    opposite_action="half",
                )
            )
        for th in (0.50, 0.70):
            out.append(
                Candidate(
                    name=f"cost_firewall_{buf:.4f}_dsac_veto_opp_{th:.2f}".replace(".", "p"),
                    cost_buffer=buf,
                    opposite_threshold=th,
                    opposite_action="veto",
                )
            )
    for buf in (0.0020, 0.0035):
        for same_th, boost, opp_th in ((0.70, 1.10, 0.50), (0.80, 1.15, 0.60)):
            out.append(
                Candidate(
                    name=(
                        f"cost_firewall_{buf:.4f}_same_{same_th:.2f}_b{boost:.2f}_half_{opp_th:.2f}"
                    ).replace(".", "p"),
                    cost_buffer=buf,
                    same_threshold=same_th,
                    same_boost=boost,
                    opposite_threshold=opp_th,
                    opposite_action="half",
                )
            )
    return out


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "blocked",
        "scaled_down",
        "boosted",
        "wr",
        "trades_per_day",
        "avg_notional",
        "max_notional",
        "final_cash",
        "reason_counts",
        "cost_mult",
    )
    return {k: metrics.get(k) for k in keys}


def _pre_audit(
    df: pd.DataFrame,
    candidates: list[Candidate],
    context: dict[str, Any],
    dsac_meta: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    if context.get("missing_feature_rows", 0):
        blocking.append("feature rows missing for some ledger entry_idx")
    if context.get("timestamp_alignment_mismatches_gt_300s", 0):
        blocking.append("feature timestamps differ from ledger entry timestamps by more than 300 seconds")
    if not Path(str(dsac_meta.get("path", ""))).exists():
        blocking.append("DSAC checkpoint path missing")
    if int(dsac_meta.get("state_dim", 0) or 0) != DSAC_STATE_DIM:
        blocking.append("DSAC state_dim mismatch")
    if any(c.max_notional > 3.6 for c in candidates):
        blocking.append("candidate max_notional exceeds 3.6")
    if any((c.cost_buffer is not None and c.cost_buffer < 0.0) for c in candidates):
        blocking.append("candidate cost_buffer below zero")
    for col in ("core_side", "effective_core_notional", "core_pnl_pct", "dsac_side", "dsac_score"):
        if col not in df.columns:
            blocking.append(f"missing column {col}")
            continue
        if not np.isfinite(pd.to_numeric(df[col], errors="coerce")).all():
            blocking.append(f"non-finite column {col}")
    cost_candidates = [c.name for c in candidates if c.cost_buffer is not None]
    if not cost_candidates:
        warnings.append("no formal cost firewall candidates in experiment")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "context": context,
        "dsac": dsac_meta,
        "pre_run_checks": {
            "experiment_family": "formal_cost_firewall_plus_dsac_soft_scaler",
            "cost_firewall_is_declared_layer_not_dsac_only_candidate": True,
            "candidate_max_notional_lte_3p6": all(c.max_notional <= 3.6 for c in candidates),
            "feature_entry_idx_timestamp_aligned": context.get("timestamp_alignment_mismatches_gt_300s", 0) == 0,
            "declared_entry_time_inputs": [
                "DGG signal fields",
                "deep prediction fields",
                "M7 prediction fields",
                "regime fields",
                "DSAC actor signal at entry_idx",
            ],
            "forbidden_selection_inputs": [
                "holdout metrics",
                "future realized pnl",
                "future realized drawdown",
            ],
        },
    }


def _selection_blockers(row: pd.Series, baseline: pd.Series) -> list[str]:
    blockers: list[str] = []
    if not bool(row.get("selectable", True)):
        blockers.append("not_selectable")
    if int(row.get("val_trades", 0) or 0) <= 0:
        blockers.append("validation_no_trades")
    if float(row.get("val_cost2_pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost2_not_survived")
    if float(row.get("val_cost3_pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost3_not_survived")
    if float(row.get("val_mdd", -1e9)) < float(baseline.get("val_mdd", -1e9)) - 1e-9:
        blockers.append("validation_mdd_worse_than_noop")
    return blockers


def _post_audit(
    source: pd.DataFrame,
    selected_ledger: pd.DataFrame,
    baseline: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    base = _audit_selected_ledger(source, selected_ledger, baseline["full_cost1"], selected)
    blocking = list(base.get("blocking", []))
    warnings = list(base.get("warnings", []))
    full = selected["full_cost1"]
    cost2 = selected["full_cost2"]
    cost3 = selected["full_cost3"]
    noop_full = baseline["full_cost1"]
    if float(full.get("mdd", 0.0)) < float(noop_full.get("mdd", 0.0)) - 1e-9:
        blocking.append("selected full MDD is worse than noop")
    if float(cost2.get("pnl", 0.0)) <= 0.0:
        blocking.append("selected does not survive full 2x cost stress")
    if float(cost3.get("pnl", 0.0)) <= 0.0:
        blocking.append("selected does not survive full 3x cost stress")
    if int(full.get("trades", 0) or 0) <= 0:
        blocking.append("selected has no full-period trades")
    invariants = dict(base.get("invariants", {}))
    invariants.update(
        {
            "formal_cost_firewall_declared": True,
            "cost_gate_uses_entry_time_prediction_proxy_only": True,
            "cost_gate_does_not_use_realized_pnl_columns": True,
            "full_mdd_lte_noop": float(full.get("mdd", 0.0)) >= float(noop_full.get("mdd", 0.0)) - 1e-9,
            "full_cost2_survives": float(cost2.get("pnl", 0.0)) > 0.0,
            "full_cost3_survives": float(cost3.get("pnl", 0.0)) > 0.0,
        }
    )
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": invariants,
        "baseline_replay": base.get("baseline_replay", {}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formal DGG V2 cost firewall plus DSAC half-opposite experiment.")
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURE_CSV)
    p.add_argument("--dsac-ckpt", type=Path, default=DEFAULT_DSAC_CKPT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER_OUT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split-date", default="2026-02-01")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ledger = _read_ledger(args.ledger)
    features = _read_features(args.features)
    prices = _price_arrays(features)
    dsac, dsac_meta = _flat_dsac_signals(features, args.dsac_ckpt, args.device)
    df, context = _attach_context(ledger, features, dsac)
    candidates = build_candidates()
    pre = _pre_audit(df, candidates, context, dsac_meta)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    if pre["status"] != "pass":
        args.audit_out.write_text(json.dumps({"model_id": MODEL_ID, "pre_run": pre}, indent=2), encoding="utf-8")
        raise SystemExit("blocking pre-run audit failed")

    split = pd.Timestamp(args.split_date)
    validation = df[df["timestamp"] < split].reset_index(drop=True)
    holdout = df[df["timestamp"] >= split].reset_index(drop=True)
    if validation.empty or holdout.empty:
        validation = df.iloc[: max(1, len(df) // 2)].reset_index(drop=True)
        holdout = df.iloc[max(1, len(df) // 2) :].reset_index(drop=True)

    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        metrics = {
            "full_cost1": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "full_cost2": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "full_cost3": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "validation_cost1": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "validation_cost2": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "validation_cost3": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "holdout_cost1": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "holdout_cost2": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "holdout_cost3": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0),
        }
        score = _score(metrics)
        detailed.append({"candidate": asdict(cand), "score": score, **metrics})
        row: dict[str, Any] = {
            "name": cand.name,
            "selectable": cand.selectable,
            "cost_buffer": cand.cost_buffer,
            "same_threshold": cand.same_threshold,
            "same_boost": cand.same_boost,
            "opposite_threshold": cand.opposite_threshold,
            "opposite_action": cand.opposite_action,
            "score": score,
        }
        for prefix, data in (
            ("full", metrics["full_cost1"]),
            ("cost2", metrics["full_cost2"]),
            ("cost3", metrics["full_cost3"]),
            ("val", metrics["validation_cost1"]),
            ("val_cost2", metrics["validation_cost2"]),
            ("val_cost3", metrics["validation_cost3"]),
            ("holdout", metrics["holdout_cost1"]),
            ("holdout_cost2", metrics["holdout_cost2"]),
            ("holdout_cost3", metrics["holdout_cost3"]),
        ):
            for key in (
                "pnl",
                "mdd",
                "trades",
                "blocked",
                "scaled_down",
                "boosted",
                "trades_per_day",
                "avg_notional",
                "max_notional",
            ):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)

    grid = pd.DataFrame(rows)
    baseline_row = grid[grid["name"] == "noop_dgg_v2_replay"].iloc[0]
    blockers = grid.apply(lambda r: _selection_blockers(r, baseline_row), axis=1)
    grid["selection_eligible"] = blockers.apply(lambda xs: len(xs) == 0)
    grid["selection_blockers"] = blockers.apply(lambda xs: "|".join(xs))
    grid = grid.sort_values("score", ascending=False).reset_index(drop=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)

    eligible = grid[grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    if eligible.empty:
        selected_name = "noop_dgg_v2_replay"
    else:
        selected_name = str(eligible.iloc[0]["name"])
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    baseline = next(d for d in detailed if d["candidate"]["name"] == "noop_dgg_v2_replay")
    replay(
        df,
        Candidate(**selected["candidate"]),
        prices=prices,
        fee=args.fee,
        slip=args.slip,
        cost_mult=1.0,
        ledger_out=args.ledger_out,
    )
    selected_ledger = pd.read_csv(args.ledger_out)
    post = _post_audit(df, selected_ledger, baseline, selected)
    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if pre["status"] == "pass" and post["status"] == "pass" else "fail",
        "pre_run": pre,
        "post_selection": post,
    }
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["status"] != "pass":
        raise SystemExit("post-selection audit failed")

    report = {
        "model_id": MODEL_ID,
        "created_from": {
            "ledger": str(args.ledger),
            "features": str(args.features),
            "dsac_ckpt": str(args.dsac_ckpt),
            "split_date": args.split_date,
            "fee": args.fee,
            "slip": args.slip,
        },
        "architecture": "Deep Gated Gross V2 + formal entry-time cost firewall + DSAC half/veto opposite soft scaler",
        "selection_policy": (
            "score uses validation_cost1/2/3 only; candidate eligibility uses validation-only cost survival "
            "and validation MDD not worse than noop; holdout is report-only"
        ),
        "audit_path": str(args.audit_out),
        "audit": audit,
        "selected": {
            **selected,
            "compact": {
                "full_cost1": _compact(selected["full_cost1"]),
                "full_cost2": _compact(selected["full_cost2"]),
                "full_cost3": _compact(selected["full_cost3"]),
                "validation_cost1": _compact(selected["validation_cost1"]),
                "holdout_cost1": _compact(selected["holdout_cost1"]),
            },
        },
        "baseline": {
            **baseline,
            "compact": {
                "full_cost1": _compact(baseline["full_cost1"]),
                "full_cost2": _compact(baseline["full_cost2"]),
                "full_cost3": _compact(baseline["full_cost3"]),
                "validation_cost1": _compact(baseline["validation_cost1"]),
                "holdout_cost1": _compact(baseline["holdout_cost1"]),
            },
        },
        "top": grid.head(12).to_dict(orient="records"),
        "top_eligible": eligible.head(12).to_dict(orient="records"),
        "grid_path": str(args.grid_out),
        "selected_ledger_path": str(args.ledger_out),
        "red_team_notes": [
            "This experiment intentionally formalizes the previously reference-only cost-gated row.",
            "The cost firewall uses only entry-time prediction proxies and configured fee/slip hurdle.",
            "DSAC can halve or veto exposure when it opposes DGG; it cannot flip side or create entries.",
            "Selection is validation-only; holdout metrics are reported after selection.",
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected_name,
                "audit": audit["status"],
                "report": str(args.report_out),
                "grid": str(args.grid_out),
                "ledger": str(args.ledger_out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
