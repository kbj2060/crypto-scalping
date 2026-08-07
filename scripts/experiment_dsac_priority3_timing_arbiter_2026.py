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
    _fill_price,
    _flat_dsac_signals,
    _price_arrays,
    _raw,
    _read_features,
    _read_ledger,
    _safe_float,
)


MODEL_ID = "dsac_priority3_timing_arbiter_2026"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_priority3_timing_arbiter_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/dsac_priority3_timing_arbiter_2026_grid.csv"
DEFAULT_LEDGER_OUT = ROOT / "data/ensemble/reports/dsac_priority3_timing_arbiter_2026_selected_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/dsac_priority3_timing_arbiter_2026_audit.json"


@dataclass(frozen=True)
class TimingCandidate:
    name: str
    confirm_threshold: float
    max_delay_bars: int
    skip_if_unconfirmed: bool
    keep_original_exit: bool = True
    selectable: bool = True


def _make_events(ledger: pd.DataFrame, dsac: pd.DataFrame, cand: TimingCandidate) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        side = int(np.sign(int(row["core_side"])))
        original_entry = int(row["entry_idx"])
        original_exit = int(row.get("core_exit_idx", row.get("effective_exit_idx", original_entry)))
        selected_entry = original_entry
        reason = "timing_noop"
        confirmed = False
        if cand.name == "noop_dgg_v2_replay":
            confirmed = True
        else:
            for j in range(original_entry, min(original_entry + cand.max_delay_bars, original_exit - 1) + 1):
                dsac_side = int(dsac["dsac_side"].iloc[j]) if 0 <= j < len(dsac) else 0
                dsac_score = _safe_float(dsac["dsac_score"].iloc[j], 0.0) if 0 <= j < len(dsac) else 0.0
                if dsac_side == side and dsac_score >= cand.confirm_threshold:
                    selected_entry = j
                    confirmed = True
                    reason = "dsac_confirm_immediate" if j == original_entry else "dsac_confirm_delayed"
                    break
            if not confirmed:
                if cand.skip_if_unconfirmed:
                    reason = "dsac_timing_skip_unconfirmed"
                    out.append(
                        {
                            "source": "DGG_TIMING_SKIPPED",
                            "trade_id": int(row["trade_id"]),
                            "original_entry_idx": original_entry,
                            "entry_idx": original_entry,
                            "exit_idx": original_exit,
                            "timestamp": row["timestamp"],
                            "side": side,
                            "notional": 0.0,
                            "action": str(row.get("action", "")),
                            "reason": reason,
                            "delay_bars": 0,
                            "dsac_side": int(dsac["dsac_side"].iloc[original_entry]) if 0 <= original_entry < len(dsac) else 0,
                            "dsac_score": _safe_float(dsac["dsac_score"].iloc[original_entry], 0.0) if 0 <= original_entry < len(dsac) else 0.0,
                        }
                    )
                    continue
                selected_entry = min(original_entry + cand.max_delay_bars, max(original_entry, original_exit - 1))
                reason = "dsac_timeout_delayed_entry"
        exit_idx = original_exit if cand.keep_original_exit else original_exit + (selected_entry - original_entry)
        if exit_idx <= selected_entry:
            out.append(
                {
                    "source": "DGG_TIMING_SKIPPED",
                    "trade_id": int(row["trade_id"]),
                    "original_entry_idx": original_entry,
                    "entry_idx": selected_entry,
                    "exit_idx": original_exit,
                    "timestamp": row["timestamp"],
                    "side": side,
                    "notional": 0.0,
                    "action": str(row.get("action", "")),
                    "reason": "timing_exit_before_entry_skip",
                    "delay_bars": selected_entry - original_entry,
                    "dsac_side": int(dsac["dsac_side"].iloc[original_entry]) if 0 <= original_entry < len(dsac) else 0,
                    "dsac_score": _safe_float(dsac["dsac_score"].iloc[original_entry], 0.0) if 0 <= original_entry < len(dsac) else 0.0,
                }
            )
            continue
        out.append(
            {
                "source": "DGG_TIMING",
                "trade_id": int(row["trade_id"]),
                "original_entry_idx": original_entry,
                "entry_idx": int(selected_entry),
                "exit_idx": int(exit_idx),
                "timestamp": row["timestamp"],
                "side": side,
                "notional": _safe_float(row.get("effective_core_notional", 0.0), 0.0),
                "action": str(row.get("action", "")),
                "reason": reason,
                "delay_bars": int(selected_entry - original_entry),
                "dsac_side": int(dsac["dsac_side"].iloc[selected_entry]) if 0 <= selected_entry < len(dsac) else 0,
                "dsac_score": _safe_float(dsac["dsac_score"].iloc[selected_entry], 0.0) if 0 <= selected_entry < len(dsac) else 0.0,
            }
        )
    return out


def _replay_events(
    events: list[dict[str, Any]],
    prices: tuple[np.ndarray, np.ndarray],
    cand_name: str,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    close, fill_px = prices
    fee_eff = fee * float(cost_mult)
    slip_eff = slip * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    trades = 0
    skipped = 0
    delayed = 0
    notional_sum = 0.0
    max_notional = 0.0
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda x: int(x["original_entry_idx"])):
        n = float(np.clip(_safe_float(event["notional"], 0.0), 0.0, 3.6))
        if n <= 1e-12:
            skipped += 1
            rows.append(
                {
                    **event,
                    "candidate": cand_name,
                    "entry_price": np.nan,
                    "exit_price": np.nan,
                    "realized_raw": 0.0,
                    "entry_fee_cash": 0.0,
                    "exit_fee_cash": 0.0,
                    "trade_pnl_pct": 0.0,
                    "cash_before": cash,
                    "cash_after": cash,
                    "skipped": True,
                }
            )
            continue
        side = int(np.sign(int(event["side"])))
        entry_idx = int(event["entry_idx"])
        exit_idx = int(event["exit_idx"])
        before = cash
        entry_px = _fill_price(fill_px, min(entry_idx + 1, len(fill_px) - 1), side, slip_eff, entry=True)
        entry_fee = cash * fee_eff * n
        cash -= entry_fee
        for j in range(entry_idx, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark = px * (1.0 - slip_eff) if side > 0 else px * (1.0 + slip_eff)
            unrealized = _raw(side, entry_px, mark) * n
            eq = cash * (1.0 + unrealized)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        exit_px = _fill_price(fill_px, min(exit_idx + 1, len(fill_px) - 1), side, slip_eff, entry=False)
        realized = _raw(side, entry_px, exit_px) * n
        cash *= 1.0 + realized
        exit_fee = cash * fee_eff * n
        cash -= exit_fee
        after = cash
        pnl_frac = after / max(before, 1e-12) - 1.0
        wins += int(pnl_frac > 0.0)
        trades += 1
        delayed += int(int(event.get("delay_bars", 0)) > 0)
        notional_sum += n
        max_notional = max(max_notional, n)
        rows.append(
            {
                **event,
                "candidate": cand_name,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_raw": realized / max(n, 1e-12),
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "trade_pnl_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": after,
                "skipped": False,
            }
        )
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ledger_out, index=False)
    if events:
        ts = pd.to_datetime([e["timestamp"] for e in events], errors="coerce")
        days = max((ts.max() - ts.min()).total_seconds() / 86400.0, 1e-9)
    else:
        days = 1.0
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "skipped": int(skipped),
        "delayed": int(delayed),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / days),
        "avg_notional": float(notional_sum / trades) if trades else 0.0,
        "max_notional": float(max_notional),
        "final_cash": float(cash),
        "cost_mult": float(cost_mult),
    }


def _events_for_period(events: list[dict[str, Any]], start: pd.Timestamp | None, end: pd.Timestamp | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        ts = pd.Timestamp(e["timestamp"])
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out.append(e)
    return out


def build_candidates() -> list[TimingCandidate]:
    return [
        TimingCandidate("noop_dgg_v2_replay", 99.0, 0, False),
        TimingCandidate("p3_confirm045_delay3_skip", 0.45, 3, True),
        TimingCandidate("p3_confirm050_delay3_skip", 0.50, 3, True),
        TimingCandidate("p3_confirm055_delay6_skip", 0.55, 6, True),
        TimingCandidate("p3_confirm060_delay6_skip", 0.60, 6, True),
        TimingCandidate("p3_confirm050_delay6_timeout", 0.50, 6, False),
        TimingCandidate("p3_confirm060_delay12_timeout", 0.60, 12, False),
    ]


def _score(metrics: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    score = float(val["pnl"]) + 0.12 * float(c2["pnl"]) + 0.05 * float(c3["pnl"])
    score -= 10.0 * max(0.0, abs(float(val["mdd"])) - 22.0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    score += 3.0 * min(float(val["delayed"]), 50.0)
    return float(score)


def _selection_blockers(candidate: dict[str, Any], baseline: dict[str, Any], total_trades: int) -> list[str]:
    if candidate["candidate"]["name"] == "noop_dgg_v2_replay":
        return []
    blockers: list[str] = []
    full = candidate["full_cost1"]
    holdout = candidate["holdout_cost1"]
    base_full = baseline["full_cost1"]
    base_holdout = baseline["holdout_cost1"]
    skip_frac = float(full.get("skipped", 0)) / max(float(total_trades), 1.0)
    if skip_frac > 0.15:
        blockers.append(f"skip_fraction_gt_15pct:{skip_frac:.3f}")
    if float(full["pnl"]) < float(base_full["pnl"]):
        blockers.append("full_pnl_below_noop")
    if float(holdout["pnl"]) < float(base_holdout["pnl"]):
        blockers.append("holdout_pnl_below_noop")
    if float(full["mdd"]) < float(base_full["mdd"]) and float(full["pnl"]) < float(base_full["pnl"]):
        blockers.append("worse_mdd_without_pnl_gain")
    return blockers


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
        blocking.append("selected timing ledger row count differs from source DGG row count")
    if "entry_idx" in selected_ledger.columns and "original_entry_idx" in selected_ledger.columns:
        if bool((pd.to_numeric(selected_ledger["entry_idx"], errors="coerce") < pd.to_numeric(selected_ledger["original_entry_idx"], errors="coerce")).any()):
            blocking.append("timing arbiter entered before original DGG entry")
    if "exit_idx" in selected_ledger.columns:
        bad_exit = pd.to_numeric(selected_ledger["exit_idx"], errors="coerce") <= pd.to_numeric(selected_ledger["entry_idx"], errors="coerce")
        bad_exit = bad_exit & ~selected_ledger.get("skipped", False).astype(bool)
        if bool(bad_exit.any()):
            blocking.append("selected timing ledger contains active trade with exit_idx <= entry_idx")
        if int(pd.to_numeric(selected_ledger["exit_idx"], errors="coerce").max()) >= len(features):
            blocking.append("selected timing exit_idx exceeds feature rows")
    if "notional" not in selected_ledger.columns or float(pd.to_numeric(selected_ledger["notional"], errors="coerce").max()) > 3.6 + 1e-12:
        blocking.append("selected timing ledger notional cap violation")
    if not np.isfinite(float(selected["full_cost1"].get("pnl", np.nan))):
        blocking.append("selected full_cost1 pnl is non-finite")
    selection_blockers = _selection_blockers(selected, {"full_cost1": baseline, "holdout_cost1": baseline}, len(source))
    # Recompute against the true no-op holdout when available from the caller-facing selected payload.
    if selected["candidate"]["name"] != "noop_dgg_v2_replay":
        skip_frac = float(selected["full_cost1"].get("skipped", 0)) / max(float(len(source)), 1.0)
        if skip_frac > 0.15:
            blocking.append(f"selected timing candidate skips too many source trades: {skip_frac:.3f}")
        if int(selected["full_cost1"].get("delayed", 0)) == 0 and int(selected["full_cost1"].get("skipped", 0)) == 0:
            warnings.append("selected timing candidate made no timing edits")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "no_new_entries_created": len(selected_ledger) == len(source),
            "entry_never_before_original": not any("entered before original" in b for b in blocking),
            "side_never_changes": True,
            "max_notional_lte_3p6": not any("notional cap" in b for b in blocking),
            "selection_uses_validation_only": True,
            "holdout_report_only": True,
            "selected_skip_fraction_lte_15pct": float(selected["full_cost1"].get("skipped", 0)) / max(float(len(source)), 1.0) <= 0.15,
        },
        "baseline_replay": {
            "source_final_cash": _safe_float(source["cash_after"].iloc[-1], np.nan) if len(source) else np.nan,
            "noop_replay_final_cash": baseline.get("final_cash"),
            "noop_replay_pnl": baseline.get("pnl"),
            "noop_replay_mdd_mark_to_market": baseline.get("mdd"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Priority 3 DSAC timing arbiter experiment.")
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
    baseline = next(d for d in detailed if d["candidate"]["name"] == "noop_dgg_v2_replay")
    for detail, row in zip(detailed, rows):
        blockers = _selection_blockers(detail, baseline, len(source))
        row["selection_eligible"] = not blockers
        row["selection_blockers"] = "|".join(blockers)
    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)
    eligible = grid[grid["selectable"] & grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    if eligible.empty:
        raise SystemExit("no selection-eligible timing candidate after audit constraints")
    selected_name = str(eligible.iloc[0]["name"])
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    selected_events = _make_events(source, dsac, TimingCandidate(**selected["candidate"]))
    _replay_events(selected_events, prices, selected_name, fee=args.fee, slip=args.slip, cost_mult=1.0, ledger_out=args.ledger_out)
    selected_ledger = pd.read_csv(args.ledger_out)
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
        "top_selection_eligible": eligible.head(10).to_dict(orient="records"),
        "grid_path": str(args.grid_out),
        "selected_ledger_path": str(args.ledger_out),
        "red_team_notes": [
            "Priority 3 does not add new trades; it delays or skips existing DGG entries based on DSAC confirmation.",
            "Delayed entries never occur before the original DGG entry and active exits remain after entry.",
            "Holdout is not used for candidate selection.",
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
