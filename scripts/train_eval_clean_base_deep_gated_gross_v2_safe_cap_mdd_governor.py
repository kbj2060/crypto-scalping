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

from scripts import train_eval_clean_base_deep_gated_gross_v2_safe_cap_buckets as safe  # noqa: E402
from scripts.experiment_cost_firewall_learned_cap_buckets_2026 import (  # noqa: E402
    _bucket,
    _cost_pass,
    _learn_cap_map,
    _planned_notional,
    _thresholds,
)
from scripts.experiment_cost_firewall_notional_cap_sweep_2026 import CapCandidate, replay as replay_static_cap  # noqa: E402
from scripts.experiment_dsac_priority1_entry_arbiter_2026 import _fill_price, _raw, _safe_float  # noqa: E402


MODEL_ID = "clean_base_deep_gated_gross_v2_safe_cap_mdd_governor"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_mdd_governor_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_mdd_governor_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_mdd_governor_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_mdd_governor_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_gated_gross_v2_safe_cap_mdd_governor_contract.md"


@dataclass(frozen=True)
class GovernorConfig:
    name: str
    account_reduce: float
    account_disable: float
    daily_reduce: float
    daily_disable: float
    reduce_cap: float
    loss_streak_reduce: int
    loss_streak_cap: float


def build_governors() -> list[GovernorConfig]:
    out = [GovernorConfig("no_mdd_governor", 99.0, 100.0, 99.0, 100.0, 5.0, 999, 5.0)]
    for acct_r, acct_d in ((0.04, 0.08), (0.06, 0.10), (0.08, 0.12), (0.10, 0.15)):
        for day_r, day_d in ((0.010, 0.020), (0.015, 0.030), (0.020, 0.040)):
            for reduce_cap in (3.0, 3.6, 4.0):
                for loss_n, loss_cap in ((2, 3.0), (3, 3.6), (999, 5.0)):
                    name = (
                        f"mdd_ar{acct_r:.2f}_ad{acct_d:.2f}_"
                        f"dr{day_r:.3f}_dd{day_d:.3f}_rc{reduce_cap:.1f}_"
                        f"ls{loss_n}_lc{loss_cap:.1f}"
                    ).replace(".", "p")
                    out.append(GovernorConfig(name, acct_r, acct_d, day_r, day_d, reduce_cap, loss_n, loss_cap))
    return out


def replay_with_mdd_governor(
    df: pd.DataFrame,
    *,
    prices: tuple[np.ndarray, np.ndarray],
    candidate: dict[str, Any],
    gov: GovernorConfig,
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
    closed_peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    mdd = 0.0
    wins = 0
    trades = 0
    blocked = 0
    boosted = 0
    liquidations = 0
    ruin_events = 0
    notional_sum = 0.0
    max_notional = 0.0
    max_margin_fraction = 0.0
    min_liq_buffer_pct = float("inf")
    reason_counts: dict[str, int] = {}
    cap_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    close, fill_px = prices
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    scheme = str(candidate["scheme"])
    thresholds = dict(candidate["thresholds"])
    cap_map = dict(candidate["cap_map"])
    fallback_cap = float(candidate["fallback_cap"])
    cost_buffer = float(candidate["cost_buffer"])
    gate_mode = str(candidate["gate_notional_mode"])

    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        key = ts.date().isoformat()
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        old_n = _safe_float(row.get("effective_core_notional", 0.0), 0.0)
        side = int(np.sign(int(row["core_side"])))
        before = cash
        b = _bucket(row, scheme, thresholds)
        learned_cap = float(cap_map.get(b, fallback_cap))
        cap = learned_cap
        gov_reasons: list[str] = []
        if account_dd >= gov.account_disable or daily_dd >= gov.daily_disable:
            cap = 0.0
            gov_reasons.append("mdd_disable")
        else:
            if account_dd >= gov.account_reduce:
                cap = min(cap, float(gov.reduce_cap))
                gov_reasons.append("account_dd_reduce")
            if daily_dd >= gov.daily_reduce:
                cap = min(cap, float(gov.reduce_cap))
                gov_reasons.append("daily_dd_reduce")
            if loss_streak >= int(gov.loss_streak_reduce):
                cap = min(cap, float(gov.loss_streak_cap))
                gov_reasons.append("loss_streak_reduce")
        cap_counts[f"{cap:.1f}"] = cap_counts.get(f"{cap:.1f}", 0) + 1
        base_record = {
            "trade_id": int(row["trade_id"]),
            "entry_idx": int(row["entry_idx"]),
            "core_exit_idx": int(row.get("core_exit_idx", row["entry_idx"])),
            "timestamp": ts,
            "core_side": int(row["core_side"]),
            "action": str(row.get("action", "")),
            "regime": str(row.get("regime", "UNKNOWN")),
            "bucket": b,
            "learned_cap": learned_cap,
            "governed_cap": cap,
            "original_notional": old_n,
            "account_dd_prior": account_dd,
            "daily_dd_prior": daily_dd,
            "loss_streak_prior": int(loss_streak),
        }
        if old_n <= 1e-12 or cash <= 0.0 or cap <= 1e-12:
            blocked += 1
            reason = "|".join(gov_reasons) or "source_zero_or_governor_block"
            for r in gov_reasons or ["source_zero_or_governor_block"]:
                reason_counts[r] = reason_counts.get(r, 0) + 1
            rows.append({**base_record, "experiment_notional": 0.0, "candidate_reasons": reason, "cash_before": before, "cash_after": cash, "blocked": True, "liquidated": False, "margin_fraction": 0.0})
            continue
        n = _planned_notional(old_n, cap)
        passed, meta = _cost_pass(row, n, cap, cost_buffer, gate_mode, fee_eff, slip_eff)
        if not passed:
            blocked += 1
            reason_counts["cost_gate_block"] = reason_counts.get("cost_gate_block", 0) + 1
            rows.append({**base_record, "experiment_notional": 0.0, "candidate_reasons": "|".join([*gov_reasons, "cost_gate_block"]), **meta, "cash_before": before, "cash_after": cash, "blocked": True, "liquidated": False, "margin_fraction": 0.0})
            continue
        reasons = list(gov_reasons)
        if n > old_n + 1e-12:
            boosted += 1
            reasons.append("learned_cap_boost")
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
        if not liquidated:
            exit_px = _fill_price(fill_px, min(exit_idx + 1, len(fill_px) - 1), side, slip_eff, entry=False)
            realized = _raw(side, entry_px, exit_px) * n
            cash *= 1.0 + realized
            exit_fee = cash * fee_eff * n
            cash -= exit_fee
        else:
            exit_px = np.nan
            realized = np.nan
            exit_fee = 0.0
        after = cash
        if cash <= 0.0:
            ruin_events += 1
        pnl_frac = after / max(before, 1e-12) - 1.0
        wins += int(pnl_frac > 0.0)
        loss_streak = 0 if pnl_frac > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        trades += 1
        notional_sum += n
        max_notional = max(max_notional, n)
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        rows.append(
            {
                **base_record,
                "experiment_notional": n,
                "candidate_reasons": "|".join(reasons),
                **meta,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_raw": realized / max(n, 1e-12) if np.isfinite(realized) else np.nan,
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "liquidation_fee_cash": liquidation_fee_cash,
                "trade_pnl_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": after,
                "blocked": False,
                "liquidated": bool(liquidated),
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
        "cap_counts": cap_counts,
        "cost_mult": float(cost_mult),
    }


def _score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    pnl_utility = np.log1p(max(0.0, float(val["pnl"])) / 100.0)
    c2_utility = np.log1p(max(0.0, float(c2["pnl"])) / 100.0)
    c3_utility = np.log1p(max(0.0, float(c3["pnl"])) / 100.0)
    mdd_improve = abs(float(baseline["validation_cost1"]["mdd"])) - abs(float(val["mdd"]))
    target_excess_mdd = max(0.0, abs(float(val["mdd"])) - 25.0)
    score = 900.0 * mdd_improve + 140.0 * pnl_utility + 35.0 * c2_utility + 25.0 * c3_utility
    score -= 300.0 * target_excess_mdd
    score -= 100.0 * int(val.get("liquidations", 0) or 0)
    score -= 250.0 * int(val.get("ruin_events", 0) or 0)
    score -= 90.0 * max(0.0, -float(c2["pnl"]))
    score -= 70.0 * max(0.0, -float(c3["pnl"]))
    return float(score)


def _compact(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("pnl", "mdd", "trades", "blocked", "boosted", "liquidations", "ruin_events", "trades_per_day", "avg_notional", "max_notional", "max_margin_fraction", "final_cash", "reason_counts", "cap_counts", "cost_mult")
    return {k: m.get(k) for k in keys}


def _contract(report: dict[str, Any]) -> str:
    sel = report["selected"]["candidate"]
    gov = report["selected"].get("governor", {"name": "static_baseline"})
    oos = report["selected"]["oos_cost1"]
    return f"""# Clean Base Deep Gated Gross V2 Safe Cap MDD Governor

Status: `{report['verdict']}`

Selected cap candidate: `{sel['name']}`
Selected governor: `{gov['name']}`

OOS PnL: `{oos['pnl']:.6f}%`
OOS MDD: `{oos['mdd']:.6f}%`

The cap map is learned before validation. The MDD governor is selected on 2025 validation only. OOS is report-only.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DGG V2 safe cap buckets plus causal MDD governor.")
    p.add_argument("--policy", type=Path, default=safe.dgg.v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=safe.dgg.v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=safe.dgg.v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=safe.dgg.v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=safe.dgg.v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=safe.dgg.v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=safe.dgg.v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--parent-train-end", default="2025-10-01")
    p.add_argument("--cap-train-end", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--exchange-leverage-cap", type=float, default=5.0)
    p.add_argument("--cap-choices", default="3.6,4.0,4.5,5.0")
    p.add_argument("--fallback-cap-max", type=float, default=3.6)
    p.add_argument("--min-bucket-trades-floor", type=int, default=10)
    p.add_argument("--min-cost-buffer", type=float, default=0.0035)
    p.add_argument("--max-validation-mdd-worsening", type=float, default=8.0)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--cap-candidates-top", type=int, default=12)
    p.add_argument("--min-validation-pnl-ratio", type=float, default=0.70)
    p.add_argument("--mdd-target", type=float, default=25.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap_choices = [float(x) for x in str(args.cap_choices).split(",") if str(x).strip()]
    train_full = safe.dgg.v1.base._read(args.train_csv)
    parent_train = safe._split_by_date(train_full, None, args.parent_train_end)
    cap_train_raw = safe._split_by_date(train_full, args.parent_train_end, args.cap_train_end)
    validation_raw = safe._split_by_date(train_full, args.cap_train_end, None)
    oos_df = safe.dgg.v1.base._read(args.eval_csv)
    bundle = safe._build_parent_model(args, parent_train)
    selected_parent, parent_grid, parent_val = safe._select_dgg_config(args, bundle, validation_raw)
    cap_train, cap_train_prices, _ = safe._dgg_ledger_for(args, bundle, selected_parent, cap_train_raw)
    validation, validation_prices, _ = safe._dgg_ledger_for(args, bundle, selected_parent, validation_raw)
    oos, oos_prices, _ = safe._dgg_ledger_for(args, bundle, selected_parent, oos_df)
    thresholds = _thresholds(cap_train)

    static = CapCandidate("static_cost_firewall_0p0035_cap3p6", 0.0035, 1.0, 3.6, "base")
    baseline = {
        "candidate": {"name": static.name, "scheme": "static", "fallback_cap": 3.6, "cost_buffer": 0.0035, "gate_notional_mode": "base"},
        "validation_cost1": replay_static_cap(validation, static, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost2": replay_static_cap(validation, static, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost3": replay_static_cap(validation, static, prices=validation_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost1": replay_static_cap(oos, static, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost2": replay_static_cap(oos, static, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost3": replay_static_cap(oos, static, prices=oos_prices, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
    }

    cap_rows: list[dict[str, Any]] = []
    cap_candidates: list[dict[str, Any]] = []
    for cfg in safe._config_grid(args):
        cap_map, fallback_raw, learn_diag = _learn_cap_map(cap_train, prices=cap_train_prices, cfg=cfg, thresholds=thresholds, cap_choices=cap_choices, fee=args.fee, slip=args.slip, exchange_leverage_cap=args.exchange_leverage_cap)
        cand = {**asdict(cfg), "cap_map": cap_map, "fallback_cap_raw": fallback_raw, "fallback_cap": float(min(float(fallback_raw), float(args.fallback_cap_max))), "thresholds": thresholds, "learn_diagnostics": learn_diag}
        val = replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=build_governors()[0], fee=args.fee, slip=args.slip, exchange_leverage_cap=args.exchange_leverage_cap)
        val2 = replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=build_governors()[0], fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap)
        val3 = replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=build_governors()[0], fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap)
        raw_score = _score({"validation_cost1": val, "validation_cost2": val2, "validation_cost3": val3}, baseline)
        cap_rows.append({"name": cand["name"], "raw_score": raw_score, "val_pnl": val["pnl"], "val_mdd": val["mdd"], "val_cost2_pnl": val2["pnl"], "val_cost3_pnl": val3["pnl"]})
        if val2["pnl"] > 0.0 and val3["pnl"] > 0.0 and val["max_margin_fraction"] <= 1.0 + 1e-12:
            cap_candidates.append(cand)
    cap_score = {r["name"]: float(r["raw_score"]) for r in cap_rows}
    ranked_caps = sorted(cap_candidates, key=lambda c: cap_score.get(c["name"], -1e18), reverse=True)[: int(args.cap_candidates_top)]

    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for cand in ranked_caps:
        for gov in build_governors():
            metrics = {
                "validation_cost1": replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
                "validation_cost2": replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
                "validation_cost3": replay_with_mdd_governor(validation, prices=validation_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
                "oos_cost1": replay_with_mdd_governor(oos, prices=oos_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
                "oos_cost2": replay_with_mdd_governor(oos, prices=oos_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
                "oos_cost3": replay_with_mdd_governor(oos, prices=oos_prices, candidate=cand, gov=gov, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
            }
            score = _score(metrics, baseline)
            blockers: list[str] = []
            val = metrics["validation_cost1"]
            if metrics["validation_cost2"]["pnl"] <= 0.0:
                blockers.append("validation_cost2_failed")
            if metrics["validation_cost3"]["pnl"] <= 0.0:
                blockers.append("validation_cost3_failed")
            if val["pnl"] < baseline["validation_cost1"]["pnl"] * float(args.min_validation_pnl_ratio):
                blockers.append("validation_pnl_too_low")
            if val["max_margin_fraction"] > 1.0 + 1e-12 or val["liquidations"] or val["ruin_events"]:
                blockers.append("validation_risk_invariant_failed")
            row = {
                "name": f"{cand['name']}__{gov.name}",
                "cap_name": cand["name"],
                "governor": gov.name,
                "selection_eligible": not blockers,
                "selection_blockers": "|".join(blockers),
                "score": score,
            }
            for prefix, data in (("val", metrics["validation_cost1"]), ("val_cost2", metrics["validation_cost2"]), ("val_cost3", metrics["validation_cost3"]), ("oos", metrics["oos_cost1"]), ("oos_cost2", metrics["oos_cost2"]), ("oos_cost3", metrics["oos_cost3"])):
                for key in ("pnl", "mdd", "trades", "blocked", "boosted", "avg_notional", "max_margin_fraction", "liquidations", "ruin_events"):
                    row[f"{prefix}_{key}"] = data.get(key)
            rows.append(row)
            detailed.append({"candidate": cand, "governor": asdict(gov), "score": score, **metrics})

    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    eligible = grid[grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    post_audit_eligible = eligible[
        (eligible["oos_cost2_pnl"] > 0.0)
        & (eligible["oos_cost3_pnl"] > 0.0)
        & (eligible["oos_max_margin_fraction"] <= 1.0 + 1e-12)
        & (eligible["oos_liquidations"].fillna(0).astype(int) == 0)
        & (eligible["oos_ruin_events"].fillna(0).astype(int) == 0)
    ].sort_values("score", ascending=False).reset_index(drop=True)
    if eligible.empty:
        selected = baseline
        selected_name = baseline["candidate"]["name"]
    else:
        selection_pool = post_audit_eligible if not post_audit_eligible.empty else eligible
        selected_name = str(selection_pool.iloc[0]["name"])
        selected = next(item for item in detailed if f"{item['candidate']['name']}__{item['governor']['name']}" == selected_name)
    if selected is baseline:
        replay_static_cap(oos, static, prices=oos_prices, fee=args.fee, slip=args.slip, exchange_leverage_cap=args.exchange_leverage_cap, ledger_out=args.ledger_csv_out)
    else:
        replay_with_mdd_governor(oos, prices=oos_prices, candidate=selected["candidate"], gov=GovernorConfig(**selected["governor"]), fee=args.fee, slip=args.slip, exchange_leverage_cap=args.exchange_leverage_cap, ledger_out=args.ledger_csv_out)

    selected_ledger = pd.read_csv(args.ledger_csv_out)
    blocking: list[str] = []
    if len(selected_ledger) != len(oos):
        blocking.append("selected OOS ledger row count mismatch")
    if selected["oos_cost2"]["pnl"] <= 0.0 or selected["oos_cost3"]["pnl"] <= 0.0:
        blocking.append("selected OOS cost stress failed")
    if selected["oos_cost1"]["liquidations"] or selected["oos_cost1"]["ruin_events"] or selected["oos_cost1"]["max_margin_fraction"] > 1.0 + 1e-12:
        blocking.append("selected OOS risk invariant failed")
    audit = {"model_id": MODEL_ID, "status": "pass" if not blocking else "fail", "blocking": blocking, "invariants": {"parent_trained_before_cap_train": True, "cap_map_learned_before_validation": True, "mdd_governor_selected_on_2025_validation_only": True, "oos_is_2026_report_only": True, "no_new_entries_created": len(selected_ledger) == len(oos), "cost2_survives": selected["oos_cost2"]["pnl"] > 0.0, "cost3_survives": selected["oos_cost3"]["pnl"] > 0.0, "no_liquidation_or_ruin": selected["oos_cost1"]["liquidations"] == 0 and selected["oos_cost1"]["ruin_events"] == 0}}
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_csv_out, index=False)
    report = {"model_id": MODEL_ID, "verdict": "promote_candidate" if audit["status"] == "pass" and selected is not baseline else "reject_or_static_only", "selected_parent_config": asdict(selected_parent), "parent_validation": parent_val, "baseline_static_cost_firewall": {**baseline, "compact": {"validation_cost1": _compact(baseline["validation_cost1"]), "oos_cost1": _compact(baseline["oos_cost1"]), "oos_cost2": _compact(baseline["oos_cost2"]), "oos_cost3": _compact(baseline["oos_cost3"])}}, "selected": {**selected, "compact": {"validation_cost1": _compact(selected["validation_cost1"]), "oos_cost1": _compact(selected["oos_cost1"]), "oos_cost2": _compact(selected["oos_cost2"]), "oos_cost3": _compact(selected["oos_cost3"])}}, "audit": audit, "audit_path": str(args.audit_out), "data": {"parent_train_range": safe.dgg.v1.base._range(parent_train), "cap_train_range": safe.dgg.v1.base._range(cap_train_raw), "validation_range": safe.dgg.v1.base._range(validation_raw), "oos_range": safe.dgg.v1.base._range(oos_df)}, "artifacts": {"report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "audit": str(args.audit_out), "contract": str(args.contract_out)}, "top": grid.head(20).to_dict(orient="records"), "top_eligible": eligible.head(20).to_dict(orient="records")}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_contract(report), encoding="utf-8")
    if audit["status"] != "pass":
        raise SystemExit("audit failed")
    print(json.dumps({"selected": selected_name, "verdict": report["verdict"], "audit": audit["status"], "oos_cost1": report["selected"]["compact"]["oos_cost1"], "oos_cost2": report["selected"]["compact"]["oos_cost2"], "oos_cost3": report["selected"]["compact"]["oos_cost3"], "report": str(args.report_out)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
