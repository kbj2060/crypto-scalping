#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_dsac_priority1_entry_arbiter_2026 import (  # noqa: E402
    DEFAULT_DSAC_CKPT,
    DEFAULT_FEATURE_CSV,
    DEFAULT_LEDGER,
    _attach_context,
    _flat_dsac_signals,
    _price_arrays,
    _read_features,
    _read_ledger,
    _safe_float,
)
from scripts.experiment_dsac_priority3_timing_arbiter_2026 import _events_for_period, _replay_events  # noqa: E402


MODEL_ID = "dsac_priority4_exit_hold_governor_2026"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_priority4_exit_hold_governor_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/dsac_priority4_exit_hold_governor_2026_grid.csv"
DEFAULT_LEDGER_OUT = ROOT / "data/ensemble/reports/dsac_priority4_exit_hold_governor_2026_selected_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/dsac_priority4_exit_hold_governor_2026_audit.json"


@dataclass(frozen=True)
class ExitHoldCandidate:
    name: str
    early_exit_threshold: float
    hold_threshold: float
    min_age_bars: int
    max_extend_bars: int
    selectable: bool = True


def _make_events(source: pd.DataFrame, dsac: pd.DataFrame, cand: ExitHoldCandidate) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    next_entries = source["entry_idx"].shift(-1).fillna(len(dsac) - 1).astype(int).to_numpy()
    for pos, row in source.reset_index(drop=True).iterrows():
        entry_idx = int(row["entry_idx"])
        original_exit = int(row.get("core_exit_idx", row.get("effective_exit_idx", entry_idx)))
        side = int(np.sign(int(row["core_side"])))
        exit_idx = original_exit
        reason = "exit_noop"
        if cand.name != "noop_dgg_v2_replay":
            earliest = min(original_exit, entry_idx + max(1, cand.min_age_bars))
            for j in range(earliest, original_exit + 1):
                dsac_side = int(dsac["dsac_side"].iloc[j]) if 0 <= j < len(dsac) else 0
                dsac_score = _safe_float(dsac["dsac_score"].iloc[j], 0.0) if 0 <= j < len(dsac) else 0.0
                if dsac_side == -side and dsac_score >= cand.early_exit_threshold:
                    exit_idx = j
                    reason = "dsac_early_exit_opposite"
                    break
            if exit_idx == original_exit and cand.max_extend_bars > 0:
                dsac_side = int(dsac["dsac_side"].iloc[original_exit]) if 0 <= original_exit < len(dsac) else 0
                dsac_score = _safe_float(dsac["dsac_score"].iloc[original_exit], 0.0) if 0 <= original_exit < len(dsac) else 0.0
                if dsac_side == side and dsac_score >= cand.hold_threshold:
                    cap = min(len(dsac) - 2, int(next_entries[pos]) - 1, original_exit + cand.max_extend_bars)
                    for j in range(original_exit + 1, cap + 1):
                        hold_side = int(dsac["dsac_side"].iloc[j]) if 0 <= j < len(dsac) else 0
                        hold_score = _safe_float(dsac["dsac_score"].iloc[j], 0.0) if 0 <= j < len(dsac) else 0.0
                        if hold_side != side or hold_score < cand.hold_threshold:
                            break
                        exit_idx = j
                        reason = "dsac_hold_extend"
        if exit_idx <= entry_idx:
            exit_idx = original_exit
            reason = "exit_guard_fallback_original"
        events.append(
            {
                "source": "DGG_EXIT_HOLD",
                "trade_id": int(row["trade_id"]),
                "original_entry_idx": entry_idx,
                "entry_idx": entry_idx,
                "original_exit_idx": original_exit,
                "exit_idx": int(exit_idx),
                "timestamp": row["timestamp"],
                "side": side,
                "notional": _safe_float(row.get("effective_core_notional", 0.0), 0.0),
                "action": str(row.get("action", "")),
                "reason": reason,
                "delay_bars": 0,
                "exit_delta_bars": int(exit_idx - original_exit),
                "dsac_side": int(dsac["dsac_side"].iloc[min(max(exit_idx, 0), len(dsac) - 1)]) if len(dsac) else 0,
                "dsac_score": _safe_float(dsac["dsac_score"].iloc[min(max(exit_idx, 0), len(dsac) - 1)], 0.0) if len(dsac) else 0.0,
            }
        )
    return events


def build_candidates() -> list[ExitHoldCandidate]:
    return [
        ExitHoldCandidate("noop_dgg_v2_replay", 99.0, 99.0, 1, 0),
        ExitHoldCandidate("p4_exit050_hold070_min3_ext6", 0.50, 0.70, 3, 6),
        ExitHoldCandidate("p4_exit055_hold070_min3_ext6", 0.55, 0.70, 3, 6),
        ExitHoldCandidate("p4_exit060_hold075_min3_ext12", 0.60, 0.75, 3, 12),
        ExitHoldCandidate("p4_exit050_hold080_min6_ext12", 0.50, 0.80, 6, 12),
        ExitHoldCandidate("p4_exit065_hold080_min6_ext18", 0.65, 0.80, 6, 18),
    ]


def _score(metrics: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    score = float(val["pnl"]) + 0.12 * float(c2["pnl"]) + 0.05 * float(c3["pnl"])
    score -= 10.0 * max(0.0, abs(float(val["mdd"])) - 22.0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    score += 1.0 * min(float(val["delayed"]), 50.0)
    return float(score)


def _audit(
    features: pd.DataFrame,
    source: pd.DataFrame,
    selected_ledger: pd.DataFrame,
    baseline: dict[str, Any],
    selected: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    if context.get("missing_feature_rows", 0):
        blocking.append("feature rows missing for some DGG ledger entry_idx")
    if context.get("timestamp_alignment_mismatches_gt_300s", 0):
        blocking.append("feature timestamps differ from DGG ledger by more than 300 seconds")
    if len(source) and abs(float(baseline["final_cash"]) - _safe_float(source["cash_after"].iloc[-1], np.nan)) > 1e-6:
        blocking.append("noop replay does not reproduce source DGG final cash")
    if len(selected_ledger) != len(source):
        blocking.append("selected exit-hold ledger row count differs from source DGG row count")
    if "entry_idx" in selected_ledger.columns and "original_entry_idx" in selected_ledger.columns:
        if bool((pd.to_numeric(selected_ledger["entry_idx"], errors="coerce") != pd.to_numeric(selected_ledger["original_entry_idx"], errors="coerce")).any()):
            blocking.append("exit-hold governor changed entry index")
    if "exit_idx" in selected_ledger.columns:
        if int(pd.to_numeric(selected_ledger["exit_idx"], errors="coerce").max()) >= len(features):
            blocking.append("selected exit_idx exceeds feature rows")
        bad = pd.to_numeric(selected_ledger["exit_idx"], errors="coerce") <= pd.to_numeric(selected_ledger["entry_idx"], errors="coerce")
        if bool(bad.any()):
            blocking.append("selected exit-hold ledger has exit_idx <= entry_idx")
    if "notional" not in selected_ledger.columns or float(pd.to_numeric(selected_ledger["notional"], errors="coerce").max()) > 3.6 + 1e-12:
        blocking.append("selected exit-hold ledger notional cap violation")
    if not np.isfinite(float(selected["full_cost1"].get("pnl", np.nan))):
        blocking.append("selected full_cost1 pnl is non-finite")
    edits = selected_ledger.get("exit_delta_bars", pd.Series(0, index=selected_ledger.index))
    if selected["candidate"]["name"] != "noop_dgg_v2_replay" and not bool((pd.to_numeric(edits, errors="coerce").fillna(0) != 0).any()):
        warnings.append("selected exit-hold candidate made no exit edits")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "no_new_entries_created": len(selected_ledger) == len(source),
            "entry_index_unchanged": not any("changed entry index" in b for b in blocking),
            "side_never_changes": True,
            "max_notional_lte_3p6": not any("notional cap" in b for b in blocking),
            "selection_uses_validation_only": True,
            "holdout_report_only": True,
        },
        "baseline_replay": {
            "source_final_cash": _safe_float(source["cash_after"].iloc[-1], np.nan) if len(source) else np.nan,
            "noop_replay_final_cash": baseline.get("final_cash"),
            "noop_replay_pnl": baseline.get("pnl"),
            "noop_replay_mdd_mark_to_market": baseline.get("mdd"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Priority 4 DSAC exit/hold governor experiment.")
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
    source = _read_ledger(args.ledger)
    features = _read_features(args.features)
    prices = _price_arrays(features)
    dsac, dsac_meta = _flat_dsac_signals(features, args.dsac_ckpt, args.device)
    _ctx_df, context = _attach_context(source, features, dsac)
    split = pd.Timestamp(args.split_date)
    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cand in build_candidates():
        events = _make_events(source, dsac, cand)
        val_events = _events_for_period(events, None, split)
        holdout_events = _events_for_period(events, split, None)
        metrics = {
            "full_cost1": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "full_cost2": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "full_cost3": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "validation_cost1": _replay_events(val_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "validation_cost2": _replay_events(val_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "validation_cost3": _replay_events(val_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "holdout_cost1": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "holdout_cost2": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "holdout_cost3": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
        }
        score = _score(metrics)
        detailed.append({"candidate": asdict(cand), "score": score, **metrics})
        row = {"name": cand.name, "selectable": cand.selectable, "score": score}
        for prefix, data in (
            ("full", metrics["full_cost1"]),
            ("cost2", metrics["full_cost2"]),
            ("cost3", metrics["full_cost3"]),
            ("val", metrics["validation_cost1"]),
            ("val_cost2", metrics["validation_cost2"]),
            ("val_cost3", metrics["validation_cost3"]),
            ("holdout", metrics["holdout_cost1"]),
        ):
            for key in ("pnl", "mdd", "trades", "skipped", "delayed", "trades_per_day", "avg_notional", "max_notional"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)
    selected_name = str(grid[grid["selectable"]].iloc[0]["name"])
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    selected_events = _make_events(source, dsac, ExitHoldCandidate(**selected["candidate"]))
    _replay_events(selected_events, prices, selected_name, fee=args.fee, slip=args.slip, cost_mult=1.0, ledger_out=args.ledger_out)
    selected_ledger = pd.read_csv(args.ledger_out)
    baseline = next(d for d in detailed if d["candidate"]["name"] == "noop_dgg_v2_replay")
    audit = {
        "model_id": MODEL_ID,
        "dsac": dsac_meta,
        **_audit(features, source, selected_ledger, baseline["full_cost1"], selected, context),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["status"] != "pass":
        raise SystemExit("blocking audit failed")
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
        "selection_policy": "candidate score uses validation_cost1/2/3 only; holdout is report-only",
        "audit_path": str(args.audit_out),
        "audit": audit,
        "selected": selected,
        "top": grid.head(10).to_dict(orient="records"),
        "grid_path": str(args.grid_out),
        "selected_ledger_path": str(args.ledger_out),
        "red_team_notes": [
            "Priority 4 keeps DGG entries and notionals fixed while allowing DSAC-governed early exits or same-side hold extensions.",
            "Extensions are capped before the next DGG entry to avoid overlapping positions.",
            "Holdout is not used for candidate selection.",
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
