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
    DSAC_STATE_DIM,
    _attach_context,
    _edge_proxy,
    _fill_price,
    _flat_dsac_signals,
    _price_arrays,
    _raw,
    _read_features,
    _read_ledger,
    _safe_float,
)


MODEL_ID = "cost_firewall_notional_cap_sweep_2026"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_ledger.csv"
DEFAULT_FEATURE_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_DSAC_CKPT = ROOT / "data/ensemble/ckpt/dsac_priority1_full_retrain_20260507/best_dsac_agents.pth"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/cost_firewall_notional_cap_sweep_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/cost_firewall_notional_cap_sweep_2026_grid.csv"
DEFAULT_LEDGER_OUT = ROOT / "data/ensemble/reports/cost_firewall_notional_cap_sweep_2026_selected_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/cost_firewall_notional_cap_sweep_2026_audit.json"


@dataclass(frozen=True)
class CapCandidate:
    name: str
    cost_buffer: float | None
    notional_mult: float
    max_notional: float
    gate_notional_mode: str = "base"
    min_notional: float = 0.0
    selectable: bool = True


def _candidate_name(buf: float | None, mult: float, cap: float, gate: str) -> str:
    b = "none" if buf is None else f"{buf:.4f}".replace(".", "p")
    return f"cf_{b}_nm{mult:.2f}_cap{cap:.1f}_gate{gate}".replace(".", "p")


def build_candidates() -> list[CapCandidate]:
    out = [
        CapCandidate("noop_dgg_v2_replay", None, 1.0, 3.6, "base"),
        CapCandidate("cost_firewall_0p0035_nm1p00_cap3p6_gatebase", 0.0035, 1.0, 3.6, "base"),
    ]
    for buf in (0.0035, 0.0050, 0.0075):
        for mult in (1.10, 1.25, 1.40, 1.60, 2.00):
            for cap in (4.0, 4.5, 5.0, 6.0, 8.0):
                for gate in ("base", "final"):
                    out.append(CapCandidate(_candidate_name(buf, mult, cap, gate), buf, mult, cap, gate))
    return out


def _candidate_notional(row: pd.Series, cand: CapCandidate, fee: float, slip: float) -> tuple[float, dict[str, Any]]:
    old_n = _safe_float(row.get("effective_core_notional", 0.0), 0.0)
    base_n = float(np.clip(old_n, cand.min_notional, cand.max_notional))
    planned_n = float(np.clip(old_n * cand.notional_mult, cand.min_notional, cand.max_notional))
    edge = _edge_proxy(row)
    reasons: list[str] = []

    if cand.cost_buffer is not None:
        gate_n = base_n if cand.gate_notional_mode == "base" else planned_n
        expected_equity_edge = edge * max(gate_n, 0.0)
        hurdle = 2.0 * (fee + slip) * max(gate_n, 0.0) + float(cand.cost_buffer)
        if expected_equity_edge <= hurdle:
            reasons.append("cost_gate_block")
            return 0.0, {
                "edge": edge,
                "gate_notional": gate_n,
                "cost_hurdle": hurdle,
                "expected_equity_edge": expected_equity_edge,
                "reasons": reasons,
            }

    if planned_n > old_n + 1e-12:
        reasons.append("notional_boost")
    if planned_n < old_n - 1e-12:
        reasons.append("notional_cap_downscale")
    return planned_n, {
        "edge": edge,
        "gate_notional": base_n if cand.gate_notional_mode == "base" else planned_n,
        "cost_hurdle": 0.0,
        "expected_equity_edge": 0.0,
        "reasons": reasons,
    }


def replay(
    df: pd.DataFrame,
    cand: CapCandidate,
    *,
    prices: tuple[np.ndarray, np.ndarray],
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    exchange_leverage_cap: float = 5.0,
    maintenance_margin: float = 0.006,
    liquidation_fee: float = 0.002,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    trades = 0
    blocked = 0
    boosted = 0
    capped_down = 0
    liquidations = 0
    ruin_events = 0
    notional_sum = 0.0
    max_notional = 0.0
    max_margin_fraction = 0.0
    min_liq_buffer_pct = float("inf")
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    fee_eff = fee * float(cost_mult)
    slip_eff = slip * float(cost_mult)
    close, fill_px = prices

    for _, row in df.iterrows():
        old_n = _safe_float(row.get("effective_core_notional", 0.0), 0.0)
        before = cash
        side = int(np.sign(int(row["core_side"])))
        base_record = {
            "trade_id": int(row["trade_id"]),
            "entry_idx": int(row["entry_idx"]),
            "core_exit_idx": int(row.get("core_exit_idx", row["entry_idx"])),
            "timestamp": row["timestamp"],
            "core_side": int(row["core_side"]),
            "action": str(row.get("action", "")),
            "regime": str(row.get("regime", "UNKNOWN")),
            "original_notional": old_n,
            "candidate": cand.name,
            "notional_mult": cand.notional_mult,
            "max_notional_cap": cand.max_notional,
            "exchange_leverage_cap": exchange_leverage_cap,
        }
        if old_n <= 1e-12 or cash <= 0.0:
            blocked += 1
            rows.append(
                {
                    **base_record,
                    "experiment_notional": 0.0,
                    "candidate_reasons": "source_zero_or_account_ruin",
                    "edge_proxy": 0.0,
                    "entry_fee_cash": 0.0,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": 0.0,
                    "trade_pnl_pct": 0.0,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                    "min_liq_buffer_pct": np.nan,
                }
            )
            continue
        n, meta = _candidate_notional(row, cand, fee_eff, slip_eff)
        for reason in meta["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if n <= 1e-12:
            blocked += 1
            rows.append(
                {
                    **base_record,
                    "experiment_notional": 0.0,
                    "candidate_reasons": "|".join(meta["reasons"]) or "candidate_block",
                    "edge_proxy": meta["edge"],
                    "gate_notional": meta["gate_notional"],
                    "cost_hurdle": meta["cost_hurdle"],
                    "expected_equity_edge": meta["expected_equity_edge"],
                    "entry_fee_cash": 0.0,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": 0.0,
                    "trade_pnl_pct": 0.0,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                    "min_liq_buffer_pct": np.nan,
                }
            )
            continue
        if n > old_n + 1e-12:
            boosted += 1
        if n < old_n - 1e-12:
            capped_down += 1
        margin_fraction = float(n / max(exchange_leverage_cap, 1e-12))
        max_margin_fraction = max(max_margin_fraction, margin_fraction)
        entry_idx = int(row["entry_idx"])
        exit_idx = int(row.get("core_exit_idx", row.get("effective_exit_idx", entry_idx)))
        entry_px = _fill_price(fill_px, min(entry_idx + 1, len(fill_px) - 1), side, slip_eff, entry=True)
        entry_fee = cash * fee_eff * n
        cash -= entry_fee
        trade_min_liq_buffer = float("inf")
        liquidated = False
        liquidation_fee_cash = 0.0
        for j in range(entry_idx, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark = px * (1.0 - slip_eff) if side > 0 else px * (1.0 + slip_eff)
            unrealized = _raw(side, entry_px, mark) * n
            eq = cash * (1.0 + unrealized)
            liq_floor = before * n * maintenance_margin
            if liq_floor > 0.0:
                trade_min_liq_buffer = min(trade_min_liq_buffer, (eq - liq_floor) / liq_floor * 100.0)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            if eq <= liq_floor or eq <= 0.0:
                liquidated = True
                liquidations += 1
                liquidation_fee_cash = before * n * liquidation_fee
                cash = max(0.0, eq - liquidation_fee_cash)
                if cash <= 0.0:
                    ruin_events += 1
                break
        min_liq_buffer_pct = min(min_liq_buffer_pct, trade_min_liq_buffer)
        if liquidated:
            after = cash
            pnl_frac = after / max(before, 1e-12) - 1.0
            wins += int(pnl_frac > 0.0)
            trades += 1
            notional_sum += n
            max_notional = max(max_notional, n)
            rows.append(
                {
                    **base_record,
                    "experiment_notional": n,
                    "candidate_reasons": "|".join(meta["reasons"]),
                    "edge_proxy": meta["edge"],
                    "gate_notional": meta["gate_notional"],
                    "cost_hurdle": meta["cost_hurdle"],
                    "expected_equity_edge": meta["expected_equity_edge"],
                    "entry_price": entry_px,
                    "exit_price": np.nan,
                    "realized_raw": np.nan,
                    "entry_fee_cash": entry_fee,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": liquidation_fee_cash,
                    "trade_pnl_pct": pnl_frac * 100.0,
                    "cash_before": before,
                    "cash_after": after,
                    "blocked": False,
                    "liquidated": True,
                    "margin_fraction": margin_fraction,
                    "min_liq_buffer_pct": trade_min_liq_buffer,
                }
            )
            continue
        exit_px = _fill_price(fill_px, min(exit_idx + 1, len(fill_px) - 1), side, slip_eff, entry=False)
        realized = _raw(side, entry_px, exit_px) * n
        cash *= 1.0 + realized
        exit_fee = cash * fee_eff * n
        cash -= exit_fee
        after = cash
        if cash <= 0.0:
            ruin_events += 1
        pnl_frac = after / max(before, 1e-12) - 1.0
        wins += int(pnl_frac > 0.0)
        trades += 1
        notional_sum += n
        max_notional = max(max_notional, n)
        rows.append(
            {
                **base_record,
                "experiment_notional": n,
                "candidate_reasons": "|".join(meta["reasons"]),
                "edge_proxy": meta["edge"],
                "gate_notional": meta["gate_notional"],
                "cost_hurdle": meta["cost_hurdle"],
                "expected_equity_edge": meta["expected_equity_edge"],
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_raw": realized / max(n, 1e-12),
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "liquidation_fee_cash": liquidation_fee_cash,
                "trade_pnl_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": after,
                "blocked": False,
                "liquidated": False,
                "margin_fraction": margin_fraction,
                "min_liq_buffer_pct": trade_min_liq_buffer,
            }
        )

    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ledger_out, index=False)

    days = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0, 1e-9) if len(df) else 1.0
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "blocked": int(blocked),
        "boosted": int(boosted),
        "capped_down": int(capped_down),
        "liquidations": int(liquidations),
        "ruin_events": int(ruin_events),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / days),
        "avg_notional": float(notional_sum / trades) if trades else 0.0,
        "max_notional": float(max_notional),
        "max_margin_fraction": float(max_margin_fraction),
        "min_liq_buffer_pct": float(min_liq_buffer_pct if np.isfinite(min_liq_buffer_pct) else np.nan),
        "final_cash": float(cash),
        "reason_counts": reason_counts,
        "cost_mult": float(cost_mult),
    }


def _score(metrics: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    score = float(val["pnl"]) + 0.10 * float(c2["pnl"]) + 0.06 * float(c3["pnl"])
    score -= 6.0 * max(0.0, abs(float(val["mdd"])) - 25.0)
    score -= 120.0 * int(val.get("liquidations", 0) or 0)
    score -= 250.0 * int(val.get("ruin_events", 0) or 0)
    score -= 40.0 * max(0.0, float(val.get("max_margin_fraction", 0.0)) - 1.0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    return float(score)


def _selection_blockers(row: pd.Series, *, live_cap: float, baseline: pd.Series) -> list[str]:
    blockers: list[str] = []
    if not bool(row.get("selectable", True)):
        blockers.append("not_selectable")
    if float(row.get("max_notional_cap", 0.0)) > live_cap + 1e-12:
        blockers.append("cap_exceeds_current_live_limit")
    if float(row.get("val_max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blockers.append("validation_margin_fraction_gt_1")
    if int(row.get("val_liquidations", 0) or 0) > 0:
        blockers.append("validation_liquidation")
    if int(row.get("val_ruin_events", 0) or 0) > 0:
        blockers.append("validation_ruin")
    if float(row.get("val_cost2_pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost2_not_survived")
    if float(row.get("val_cost3_pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost3_not_survived")
    if float(row.get("val_pnl", -1e9)) <= float(baseline.get("val_pnl", -1e9)):
        blockers.append("validation_pnl_not_above_cost_firewall_base")
    return blockers


def _pre_audit(
    df: pd.DataFrame,
    candidates: list[CapCandidate],
    context: dict[str, Any],
    dsac_meta: dict[str, Any],
    *,
    max_diagnostic_cap: float,
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
    if any(c.max_notional > max_diagnostic_cap for c in candidates):
        blocking.append("candidate max_notional exceeds diagnostic cap")
    for col in ("core_side", "effective_core_notional", "core_pnl_pct"):
        if col not in df.columns:
            blocking.append(f"missing column {col}")
        elif not np.isfinite(pd.to_numeric(df[col], errors="coerce")).all():
            blocking.append(f"non-finite column {col}")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "context": context,
        "dsac": dsac_meta,
        "pre_run_checks": {
            "experiment_family": "cost_firewall_notional_multiplier_cap_sweep",
            "selection_uses_validation_only": True,
            "holdout_is_report_only": True,
            "diagnostic_caps_may_exceed_current_live_limit": True,
            "declared_entry_time_inputs": [
                "DGG signal fields",
                "deep prediction fields",
                "M7 prediction fields",
                "regime fields",
            ],
        },
    }


def _post_audit(
    source: pd.DataFrame,
    selected_ledger: pd.DataFrame,
    selected: dict[str, Any],
    baseline: dict[str, Any],
    *,
    live_cap: float,
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    if len(selected_ledger) != len(source):
        blocking.append("selected ledger row count does not match source ledger row count")
    n = pd.to_numeric(selected_ledger.get("experiment_notional", pd.Series(dtype=float)), errors="coerce")
    if n.empty:
        blocking.append("selected ledger missing experiment_notional")
    else:
        if not np.isfinite(n).all():
            blocking.append("selected ledger contains non-finite experiment_notional")
        if float(n.max()) > float(selected["candidate"]["max_notional"]) + 1e-12:
            blocking.append("selected ledger exceeds selected max_notional")
        if float(n.max()) > live_cap + 1e-12:
            blocking.append("selected ledger exceeds current live cap")
        if float(n.min()) < -1e-12:
            blocking.append("selected ledger contains negative notional")
    if "core_side" in selected_ledger.columns:
        src = pd.to_numeric(source["core_side"], errors="coerce").fillna(0).astype(int).to_numpy()
        got = pd.to_numeric(selected_ledger["core_side"], errors="coerce").fillna(0).astype(int).to_numpy()
        if len(src) == len(got) and not np.array_equal(src, got):
            blocking.append("selected ledger changed core side")
    full = selected["full_cost1"]
    cost2 = selected["full_cost2"]
    cost3 = selected["full_cost3"]
    base_full = baseline["full_cost1"]
    if int(full.get("liquidations", 0) or 0) > 0:
        blocking.append("selected full replay has liquidation events")
    if int(full.get("ruin_events", 0) or 0) > 0:
        blocking.append("selected full replay has account ruin events")
    if float(full.get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blocking.append("selected max margin fraction exceeds 1.0 under current exchange leverage cap")
    if float(cost2.get("pnl", 0.0)) <= 0.0:
        blocking.append("selected full 2x cost stress not survived")
    if float(cost3.get("pnl", 0.0)) <= 0.0:
        blocking.append("selected full 3x cost stress not survived")
    if float(full.get("pnl", -1e9)) <= float(base_full.get("pnl", -1e9)):
        warnings.append("selected full PnL is not above cost firewall base")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "no_new_entries_created": len(selected_ledger) == len(source),
            "uses_existing_dgg_v2_entries_and_exits": True,
            "no_side_flip": not any("changed core side" in b for b in blocking),
            "selected_cap_lte_current_live_limit": float(selected["candidate"]["max_notional"]) <= live_cap + 1e-12,
            "selected_max_margin_fraction_lte_1": float(full.get("max_margin_fraction", 0.0)) <= 1.0 + 1e-12,
            "selected_no_liquidations": int(full.get("liquidations", 0) or 0) == 0,
            "selected_no_ruin": int(full.get("ruin_events", 0) or 0) == 0,
            "selected_cost2_survives": float(cost2.get("pnl", 0.0)) > 0.0,
            "selected_cost3_survives": float(cost3.get("pnl", 0.0)) > 0.0,
            "selection_uses_validation_only": True,
        },
    }


def _compact(m: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "pnl",
        "mdd",
        "trades",
        "blocked",
        "boosted",
        "capped_down",
        "liquidations",
        "ruin_events",
        "trades_per_day",
        "avg_notional",
        "max_notional",
        "max_margin_fraction",
        "min_liq_buffer_pct",
        "final_cash",
        "reason_counts",
        "cost_mult",
    )
    return {k: m.get(k) for k in keys}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep higher notional caps on the audited cost firewall model.")
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
    p.add_argument("--current-live-cap", type=float, default=5.0)
    p.add_argument("--exchange-leverage-cap", type=float, default=5.0)
    p.add_argument("--max-diagnostic-cap", type=float, default=8.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ledger = _read_ledger(args.ledger)
    features = _read_features(args.features)
    prices = _price_arrays(features)
    dsac, dsac_meta = _flat_dsac_signals(features, args.dsac_ckpt, args.device)
    df, context = _attach_context(ledger, features, dsac)
    candidates = build_candidates()
    pre = _pre_audit(df, candidates, context, dsac_meta, max_diagnostic_cap=float(args.max_diagnostic_cap))
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
            "full_cost1": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "full_cost2": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "full_cost3": replay(df, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost1": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost2": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost3": replay(validation, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "holdout_cost1": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "holdout_cost2": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "holdout_cost3": replay(holdout, cand, prices=prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
        score = _score(metrics)
        detailed.append({"candidate": asdict(cand), "score": score, **metrics})
        row: dict[str, Any] = {
            "name": cand.name,
            "selectable": cand.selectable,
            "cost_buffer": cand.cost_buffer,
            "notional_mult": cand.notional_mult,
            "max_notional_cap": cand.max_notional,
            "gate_notional_mode": cand.gate_notional_mode,
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
                "boosted",
                "capped_down",
                "liquidations",
                "ruin_events",
                "trades_per_day",
                "avg_notional",
                "max_notional",
                "max_margin_fraction",
                "min_liq_buffer_pct",
            ):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)

    grid = pd.DataFrame(rows)
    base_name = "cost_firewall_0p0035_nm1p00_cap3p6_gatebase"
    baseline_row = grid[grid["name"] == base_name].iloc[0]
    blockers = grid.apply(lambda r: _selection_blockers(r, live_cap=float(args.current_live_cap), baseline=baseline_row), axis=1)
    grid["live_eligible"] = blockers.apply(lambda xs: len(xs) == 0)
    grid["live_blockers"] = blockers.apply(lambda xs: "|".join(xs))
    grid = grid.sort_values("score", ascending=False).reset_index(drop=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)

    live_eligible = grid[grid["live_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    if live_eligible.empty:
        selected_name = base_name
    else:
        selected_name = str(live_eligible.iloc[0]["name"])
    diagnostic_name = str(grid.sort_values("score", ascending=False).iloc[0]["name"])
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    diagnostic = next(d for d in detailed if d["candidate"]["name"] == diagnostic_name)
    baseline = next(d for d in detailed if d["candidate"]["name"] == base_name)
    replay(
        df,
        CapCandidate(**selected["candidate"]),
        prices=prices,
        fee=args.fee,
        slip=args.slip,
        cost_mult=1.0,
        exchange_leverage_cap=args.exchange_leverage_cap,
        ledger_out=args.ledger_out,
    )
    selected_ledger = pd.read_csv(args.ledger_out)
    post = _post_audit(selected_ledger=selected_ledger, source=df, selected=selected, baseline=baseline, live_cap=float(args.current_live_cap))
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
            "current_live_cap": args.current_live_cap,
            "exchange_leverage_cap": args.exchange_leverage_cap,
        },
        "architecture": "Deep Gated Gross V2 + formal cost firewall + notional multiplier/cap sweep",
        "selection_policy": (
            "live selected candidate uses validation-only score, must stay within current live cap and margin fraction, "
            "must survive validation 2x/3x cost stress, and must beat the audited cost firewall baseline on validation PnL"
        ),
        "audit_path": str(args.audit_out),
        "audit": audit,
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
        "selected_live_eligible": {
            **selected,
            "compact": {
                "full_cost1": _compact(selected["full_cost1"]),
                "full_cost2": _compact(selected["full_cost2"]),
                "full_cost3": _compact(selected["full_cost3"]),
                "validation_cost1": _compact(selected["validation_cost1"]),
                "holdout_cost1": _compact(selected["holdout_cost1"]),
            },
        },
        "best_diagnostic_not_live_selected": {
            **diagnostic,
            "compact": {
                "full_cost1": _compact(diagnostic["full_cost1"]),
                "full_cost2": _compact(diagnostic["full_cost2"]),
                "full_cost3": _compact(diagnostic["full_cost3"]),
                "validation_cost1": _compact(diagnostic["validation_cost1"]),
                "holdout_cost1": _compact(diagnostic["holdout_cost1"]),
            },
        },
        "top": grid.head(15).to_dict(orient="records"),
        "top_live_eligible": live_eligible.head(15).to_dict(orient="records"),
        "grid_path": str(args.grid_out),
        "selected_ledger_path": str(args.ledger_out),
        "red_team_notes": [
            "Raising notional increases PnL and MDD mechanically; live selection is capped by current bot/exchange leverage assumptions.",
            "Diagnostic candidates above the current live cap are reported but not selected for live use.",
            "Liquidation and maintenance margin are approximated in replay; any live injection still needs exchange-specific reduceOnly/order sizing checks.",
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_live": selected_name,
                "best_diagnostic": diagnostic_name,
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
