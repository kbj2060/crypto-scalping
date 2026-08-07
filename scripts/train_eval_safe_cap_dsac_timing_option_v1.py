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

from scripts import train_eval_clean_base_deep_gated_gross_v2 as dgg  # noqa: E402
from scripts import train_eval_clean_base_deep_gated_gross_v2_safe_cap_buckets as safe  # noqa: E402
from scripts.experiment_cost_firewall_learned_cap_buckets_2026 import (  # noqa: E402
    _bucket,
    _compact as _cap_compact,
    _cost_pass,
    _planned_notional,
    _raw,
    _safe_float,
    replay_bucket_map,
)
from scripts.experiment_dsac_priority1_entry_arbiter_2026 import (  # noqa: E402
    DEFAULT_DSAC_CKPT,
    _flat_dsac_signals,
)
from scripts.train_eval_safe_cap_shadow_moe_v1 import _select_safe_cap  # noqa: E402


MODEL_ID = "safe_cap_dsac_timing_option_v1"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/safe_cap_dsac_timing_option_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_ledger.csv"
DEFAULT_BASE_LEDGER = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_base_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_audit.json"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/safe_cap_dsac_timing_option_v1_contract.md"


@dataclass(frozen=True)
class TimingOptionConfig:
    name: str
    confirm_threshold: float
    max_delay_bars: int
    unconfirmed_action: str
    opposite_threshold: float
    opposite_action: str
    reduce_mult: float = 0.65
    selectable: bool = True


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _fill_price(fill_px: np.ndarray, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    if entry:
        return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _attach_dsac_signals(ledger: pd.DataFrame, dsac: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    sides: list[int] = []
    scores: list[float] = []
    raw_actions: list[float] = []
    for _, row in out.iterrows():
        idx = int(row.get("entry_idx", -1))
        if 0 <= idx < len(dsac):
            sig = dsac.iloc[idx]
            sides.append(int(sig.get("dsac_side", 0)))
            scores.append(_safe_float(sig.get("dsac_score", 0.0), 0.0))
            raw_actions.append(_safe_float(sig.get("dsac_raw_action", 0.0), 0.0))
        else:
            sides.append(0)
            scores.append(0.0)
            raw_actions.append(0.0)
    out["dsac_entry_side"] = sides
    out["dsac_entry_score"] = scores
    out["dsac_entry_raw_action"] = raw_actions
    return out


def _base_notional(
    row: pd.Series,
    *,
    safe_candidate: dict[str, Any],
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, str, dict[str, Any]]:
    old_n = _safe_float(row.get("effective_core_notional", 0.0), 0.0)
    if old_n <= 1e-12:
        return 0.0, "source_zero", {"edge": 0.0, "gate_notional": 0.0, "expected_equity_edge": 0.0, "cost_hurdle": 0.0}
    bucket = _bucket(row, str(safe_candidate["scheme"]), safe_candidate["thresholds"])
    learned_cap = float(dict(safe_candidate["cap_map"]).get(bucket, float(safe_candidate["fallback_cap"])))
    base_n = _planned_notional(old_n, learned_cap)
    passed, meta = _cost_pass(
        row,
        base_n,
        learned_cap,
        float(safe_candidate["cost_buffer"]),
        str(safe_candidate["gate_notional_mode"]),
        fee_eff,
        slip_eff,
    )
    if not passed:
        return 0.0, "base_cost_gate_block", {**meta, "bucket": bucket, "learned_cap": learned_cap}
    return float(base_n), "base_safe_cap_pass", {**meta, "bucket": bucket, "learned_cap": learned_cap}


def _timing_decision(
    row: pd.Series,
    dsac: pd.DataFrame,
    cfg: TimingOptionConfig,
    base_n: float,
) -> tuple[int, float, str, dict[str, Any]]:
    original_entry = int(row["entry_idx"])
    original_exit = int(row.get("core_exit_idx", row.get("effective_exit_idx", original_entry)))
    side = int(np.sign(int(row["core_side"])))
    if cfg.name == "noop_safe_cap_replay":
        return original_entry, base_n, "timing_noop", {"confirm_idx": original_entry, "confirm_score": 0.0, "confirm_side": 0}
    if original_exit <= original_entry + 1:
        return original_entry, 0.0, "timing_exit_too_close_skip", {"confirm_idx": original_entry, "confirm_score": 0.0, "confirm_side": 0}

    entry_dsac_side = int(dsac["dsac_side"].iloc[original_entry]) if 0 <= original_entry < len(dsac) else 0
    entry_dsac_score = _safe_float(dsac["dsac_score"].iloc[original_entry], 0.0) if 0 <= original_entry < len(dsac) else 0.0
    if entry_dsac_side == -side and entry_dsac_score >= cfg.opposite_threshold:
        if cfg.opposite_action == "skip":
            return original_entry, 0.0, "dsac_opposite_skip", {"confirm_idx": original_entry, "confirm_score": entry_dsac_score, "confirm_side": entry_dsac_side}
        if cfg.opposite_action == "reduce":
            return original_entry, base_n * cfg.reduce_mult, "dsac_opposite_reduce", {"confirm_idx": original_entry, "confirm_score": entry_dsac_score, "confirm_side": entry_dsac_side}

    last = min(original_entry + int(cfg.max_delay_bars), original_exit - 1, len(dsac) - 1)
    for j in range(original_entry, last + 1):
        dsac_side = int(dsac["dsac_side"].iloc[j]) if 0 <= j < len(dsac) else 0
        dsac_score = _safe_float(dsac["dsac_score"].iloc[j], 0.0) if 0 <= j < len(dsac) else 0.0
        if dsac_side == side and dsac_score >= cfg.confirm_threshold:
            action = "dsac_confirm_immediate" if j == original_entry else "dsac_confirm_delayed"
            return int(j), base_n, action, {"confirm_idx": int(j), "confirm_score": dsac_score, "confirm_side": dsac_side}

    if cfg.unconfirmed_action == "skip":
        return original_entry, 0.0, "dsac_unconfirmed_skip", {"confirm_idx": -1, "confirm_score": 0.0, "confirm_side": 0}
    if cfg.unconfirmed_action == "reduce":
        return original_entry, base_n * cfg.reduce_mult, "dsac_unconfirmed_reduce", {"confirm_idx": -1, "confirm_score": 0.0, "confirm_side": 0}
    if cfg.unconfirmed_action == "timeout":
        timeout_entry = min(last, original_exit - 1)
        return int(timeout_entry), base_n, "dsac_timeout_entry", {"confirm_idx": -1, "confirm_score": 0.0, "confirm_side": 0}
    return original_entry, base_n, "dsac_unconfirmed_keep", {"confirm_idx": -1, "confirm_score": 0.0, "confirm_side": 0}


def replay_timing_option(
    df: pd.DataFrame,
    *,
    dsac: pd.DataFrame,
    prices: tuple[np.ndarray, np.ndarray],
    safe_candidate: dict[str, Any],
    cfg: TimingOptionConfig,
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
    base_blocked = 0
    skipped = 0
    reduced = 0
    delayed = 0
    liquidations = 0
    ruin_events = 0
    notional_sum = 0.0
    max_notional = 0.0
    max_margin_fraction = 0.0
    min_liq_buffer_pct = float("inf")
    reason_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    close, fill_px = prices
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)

    for _, row in df.iterrows():
        before = cash
        original_entry = int(row["entry_idx"])
        original_exit = int(row.get("core_exit_idx", row.get("effective_exit_idx", original_entry)))
        side = int(np.sign(int(row["core_side"])))
        base_n, base_reason, meta = _base_notional(row, safe_candidate=safe_candidate, fee_eff=fee_eff, slip_eff=slip_eff)
        base_record = {
            "trade_id": int(row["trade_id"]),
            "timestamp": row["timestamp"],
            "core_side": side,
            "action": str(row.get("action", "")),
            "original_entry_idx": original_entry,
            "entry_idx": original_entry,
            "original_exit_idx": original_exit,
            "exit_idx": original_exit,
            "base_experiment_notional": base_n,
            "experiment_notional": 0.0,
            "base_reason": base_reason,
            "timing_action": base_reason,
            "delay_bars": 0,
            "dsac_entry_side": int(row.get("dsac_entry_side", 0) or 0),
            "dsac_entry_score": _safe_float(row.get("dsac_entry_score", 0.0), 0.0),
            **meta,
        }
        if base_n <= 1e-12 or cash <= 0.0:
            blocked += 1
            base_blocked += int(base_reason != "base_safe_cap_pass")
            reason_counts[base_reason] = reason_counts.get(base_reason, 0) + 1
            rows.append(
                {
                    **base_record,
                    "candidate_reasons": base_reason,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": base_reason != "base_safe_cap_pass",
                    "liquidated": False,
                    "margin_fraction": 0.0,
                    "entry_fee_cash": 0.0,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": 0.0,
                    "trade_pnl_pct": 0.0,
                }
            )
            continue

        entry_idx, n, action, decision_meta = _timing_decision(row, dsac, cfg, base_n)
        n = float(np.clip(n, 0.0, min(float(exchange_leverage_cap), 5.0)))
        if entry_idx >= original_exit or n <= 1e-12:
            blocked += 1
            skipped += 1
            reason_counts[action] = reason_counts.get(action, 0) + 1
            rows.append(
                {
                    **base_record,
                    **decision_meta,
                    "entry_idx": entry_idx,
                    "experiment_notional": 0.0,
                    "timing_action": action,
                    "candidate_reasons": action,
                    "cash_before": before,
                    "cash_after": cash,
                    "blocked": True,
                    "base_blocked": False,
                    "liquidated": False,
                    "margin_fraction": 0.0,
                    "entry_fee_cash": 0.0,
                    "exit_fee_cash": 0.0,
                    "liquidation_fee_cash": 0.0,
                    "trade_pnl_pct": 0.0,
                }
            )
            continue

        delayed += int(entry_idx > original_entry)
        reduced += int(n < base_n - 1e-12)
        reason_counts[action] = reason_counts.get(action, 0) + 1
        margin_fraction = float(n / max(exchange_leverage_cap, 1e-12))
        max_margin_fraction = max(max_margin_fraction, margin_fraction)
        entry_px = _fill_price(fill_px, min(entry_idx + 1, len(fill_px) - 1), side, slip_eff, entry=True)
        entry_fee = cash * fee_eff * n
        cash -= entry_fee
        trade_min_liq_buffer = float("inf")
        liquidated = False
        liquidation_fee_cash = 0.0
        for j in range(entry_idx, original_exit + 1):
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
                    **decision_meta,
                    "entry_idx": entry_idx,
                    "experiment_notional": n,
                    "timing_action": action,
                    "delay_bars": int(entry_idx - original_entry),
                    "candidate_reasons": action,
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
                    "base_blocked": False,
                    "liquidated": True,
                    "margin_fraction": margin_fraction,
                    "min_liq_buffer_pct": trade_min_liq_buffer,
                }
            )
            continue
        exit_px = _fill_price(fill_px, min(original_exit + 1, len(fill_px) - 1), side, slip_eff, entry=False)
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
                **decision_meta,
                "entry_idx": entry_idx,
                "experiment_notional": n,
                "timing_action": action,
                "delay_bars": int(entry_idx - original_entry),
                "candidate_reasons": action,
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
                "base_blocked": False,
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
        "base_blocked": int(base_blocked),
        "skipped": int(skipped),
        "reduced": int(reduced),
        "delayed": int(delayed),
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


def _build_candidates() -> list[TimingOptionConfig]:
    out = [TimingOptionConfig("noop_safe_cap_replay", 99.0, 0, "keep", 99.0, "none", 1.0)]
    for threshold in (0.06, 0.08, 0.10, 0.12, 0.15, 0.18):
        for delay in (1, 3, 6, 12):
            for unconfirmed in ("keep", "reduce", "skip", "timeout"):
                for opposite_threshold in (0.10, 0.14, 0.18, 99.0):
                    for opposite_action in ("none", "reduce", "skip"):
                        if opposite_action == "none" and opposite_threshold != 99.0:
                            continue
                        if opposite_action != "none" and opposite_threshold == 99.0:
                            continue
                        for reduce_mult in (0.50, 0.70):
                            name = (
                                f"dsac_t{threshold:.2f}_d{delay}_{unconfirmed}_"
                                f"opp{opposite_threshold:.2f}_{opposite_action}_r{reduce_mult:.2f}"
                            ).replace(".", "p")
                            out.append(
                                TimingOptionConfig(
                                    name=name,
                                    confirm_threshold=threshold,
                                    max_delay_bars=delay,
                                    unconfirmed_action=unconfirmed,
                                    opposite_threshold=opposite_threshold,
                                    opposite_action=opposite_action,
                                    reduce_mult=reduce_mult,
                                )
                            )
    return out


def _score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    base = baseline["validation_cost1"]
    score = float(val["pnl"]) + 0.10 * float(c2["pnl"]) + 0.06 * float(c3["pnl"])
    score += 2200.0 * (float(val["mdd"]) - float(base["mdd"]))
    score -= 120.0 * max(0.0, float(base["trades"]) * 0.75 - float(val["trades"]))
    score -= 120.0 * int(val.get("liquidations", 0) or 0)
    score -= 250.0 * int(val.get("ruin_events", 0) or 0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    return float(score)


def _selection_blockers(metrics: dict[str, Any], baseline: dict[str, Any], args: argparse.Namespace, cfg_name: str) -> list[str]:
    if cfg_name == "noop_safe_cap_replay":
        return []
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    base = baseline["validation_cost1"]
    blockers: list[str] = []
    if int(val.get("liquidations", 0) or 0) > 0:
        blockers.append("validation_liquidation")
    if int(val.get("ruin_events", 0) or 0) > 0:
        blockers.append("validation_ruin")
    if float(val.get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blockers.append("validation_margin_fraction_gt_1")
    if float(c2.get("pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost2_not_survived")
    if float(c3.get("pnl", -1e9)) <= 0.0:
        blockers.append("validation_cost3_not_survived")
    if float(val.get("pnl", -1e9)) < float(base.get("pnl", -1e9)) * float(args.min_validation_pnl_ratio):
        blockers.append("validation_pnl_below_safe_cap_floor")
    if float(val.get("trades", 0.0)) < float(base.get("trades", 0.0)) * float(args.min_validation_trade_ratio):
        blockers.append("validation_trade_count_too_low")
    if float(val.get("mdd", 0.0)) < float(base.get("mdd", 0.0)) - float(args.max_validation_mdd_worsening):
        blockers.append("validation_mdd_worse_than_allowed")
    return blockers


def _contract(report: dict[str, Any]) -> str:
    sel = report["selected"]
    c1 = sel["oos_cost1"]
    c3 = sel["oos_cost3"]
    return f"""# Safe Cap DSAC Timing Option V1

Status: `{report['verdict']}`

## Architecture

```mermaid
flowchart TD
    A["5m OHLCV + AI Feature Combo"] --> B["Clean Base DGG V2"]
    B --> C["Safe Learned Cap Buckets"]
    A --> D["Full Retrained DSAC Actor"]
    C --> E["Safe Base Entry Intent"]
    D --> F["Same-side confirmation / opposite-side veto"]
    E --> G["Timing Option Layer"]
    F --> G
    G --> H["Delay / Skip / Reduce / Keep"]
    H --> I["Accounting Replay with fee + slippage"]
```

## Data Splits

- Parent train: `{report['data']['parent_train_range']}`
- Safe-cap train: `{report['data']['cap_train_range']}`
- Timing validation selection: `{report['data']['validation_range']}`
- OOS report-only: `{report['data']['oos_range']}`

## Selected

- Safe cap: `{report['selected_safe_cap']['candidate']['name']}`
- Timing option: `{sel['candidate']['name']}`

## OOS Result

- PnL: `{c1['pnl']:.6f}%`
- MDD: `{c1['mdd']:.6f}%`
- Trades: `{c1['trades']}`
- Delayed: `{c1['delayed']}`
- Skipped: `{c1['skipped']}`
- Reduced: `{c1['reduced']}`
- 3x cost PnL: `{c3['pnl']:.6f}%`

## Invariants

- No new entries are created.
- Side is never changed.
- Exit index is not extended or rewritten.
- Entry can only stay at or move after the original safe-cap entry.
- Notional can stay the same, shrink, or be blocked, never exceed safe-cap base notional.
- Selection uses 2025 validation only; 2026 OOS is report-only.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DSAC timing option layer on safe learned cap.")
    p.add_argument("--policy", type=Path, default=dgg.v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=dgg.v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=dgg.v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=dgg.v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=dgg.v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=dgg.v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--dsac-ckpt", type=Path, default=DEFAULT_DSAC_CKPT)
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
    p.add_argument("--min-validation-pnl-ratio", type=float, default=0.92)
    p.add_argument("--min-validation-trade-ratio", type=float, default=0.75)
    p.add_argument("--device", default="cpu")
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--base-ledger-csv-out", type=Path, default=DEFAULT_BASE_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cap_choices = [float(x) for x in str(args.cap_choices).split(",") if str(x).strip()]
    if max(cap_choices) > float(args.exchange_leverage_cap) + 1e-12:
        raise SystemExit("cap choices exceed exchange leverage cap")

    train_full = dgg.v1.base._read(args.train_csv)
    parent_train = safe._split_by_date(train_full, None, args.parent_train_end)
    cap_train_raw = safe._split_by_date(train_full, args.parent_train_end, args.cap_train_end)
    validation_raw = safe._split_by_date(train_full, args.cap_train_end, None)
    oos_df = dgg.v1.base._read(args.eval_csv)
    if parent_train.empty or cap_train_raw.empty or validation_raw.empty or oos_df.empty:
        raise SystemExit("empty chronological split")

    bundle = safe._build_parent_model(args, parent_train)
    selected_parent, parent_grid, parent_val = safe._select_dgg_config(args, bundle, validation_raw)

    cap_train, cap_train_prices, _ = safe._dgg_ledger_for(args, bundle, selected_parent, cap_train_raw)
    validation, validation_prices, validation_parent = safe._dgg_ledger_for(args, bundle, selected_parent, validation_raw)
    oos, oos_prices, oos_parent = safe._dgg_ledger_for(args, bundle, selected_parent, oos_df)

    dsac_train, dsac_meta_train = _flat_dsac_signals(train_full, args.dsac_ckpt, args.device)
    dsac_oos, dsac_meta_oos = _flat_dsac_signals(oos_df, args.dsac_ckpt, args.device)
    validation_dsac = dsac_train.iloc[validation_raw.index.to_numpy()].reset_index(drop=True)
    validation = _attach_dsac_signals(validation, validation_dsac)
    oos = _attach_dsac_signals(oos, dsac_oos)

    selected_safe, safe_static_baseline, safe_grid = _select_safe_cap(
        args,
        cap_train,
        cap_train_prices,
        validation,
        validation_prices,
        oos,
        oos_prices,
    )
    if str(selected_safe["candidate"].get("scheme")) == "static":
        raise SystemExit("safe cap parent selection fell back to static; refusing timing layer")
    safe_candidate = selected_safe["candidate"]

    baseline_cfg = TimingOptionConfig("noop_safe_cap_replay", 99.0, 0, "keep", 99.0, "none", 1.0)
    baseline = {
        "validation_cost1": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost2": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "validation_cost3": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost1": replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap, ledger_out=args.base_ledger_csv_out),
        "oos_cost2": replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
        "oos_cost3": replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=baseline_cfg, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
    }

    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cfg in _build_candidates():
        metrics = {
            "validation_cost1": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost2": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap),
            "validation_cost3": replay_timing_option(validation, dsac=validation_dsac, prices=validation_prices, safe_candidate=safe_candidate, cfg=cfg, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap),
        }
        score = _score(metrics, baseline)
        blockers = _selection_blockers(metrics, baseline, args, cfg.name)
        candidate = asdict(cfg)
        detailed.append({"candidate": candidate, "score": score, "selection_blockers": blockers, **metrics})
        row: dict[str, Any] = {
            **candidate,
            "score": score,
            "selection_eligible": len(blockers) == 0,
            "selection_blockers": "|".join(blockers),
        }
        for prefix, data in (
            ("val", metrics["validation_cost1"]),
            ("val_cost2", metrics["validation_cost2"]),
            ("val_cost3", metrics["validation_cost3"]),
        ):
            for key in ("pnl", "mdd", "trades", "blocked", "base_blocked", "skipped", "reduced", "delayed", "liquidations", "ruin_events", "avg_notional", "max_notional", "max_margin_fraction"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)

    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    eligible = grid[grid["selection_eligible"]].sort_values("score", ascending=False).reset_index(drop=True)
    noop_score = float(grid.loc[grid["name"] == "noop_safe_cap_replay", "score"].iloc[0])
    promotable = eligible[(eligible["name"] != "noop_safe_cap_replay") & (eligible["score"] > noop_score)].reset_index(drop=True)
    selected_name = str(promotable.iloc[0]["name"]) if not promotable.empty else "noop_safe_cap_replay"
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    selected_cfg = TimingOptionConfig(**selected["candidate"])
    selected["oos_cost1"] = replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=selected_cfg, fee=args.fee, slip=args.slip, cost_mult=1.0, exchange_leverage_cap=args.exchange_leverage_cap, ledger_out=args.ledger_csv_out)
    selected["oos_cost2"] = replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=selected_cfg, fee=args.fee, slip=args.slip, cost_mult=2.0, exchange_leverage_cap=args.exchange_leverage_cap)
    selected["oos_cost3"] = replay_timing_option(oos, dsac=dsac_oos, prices=oos_prices, safe_candidate=safe_candidate, cfg=selected_cfg, fee=args.fee, slip=args.slip, cost_mult=3.0, exchange_leverage_cap=args.exchange_leverage_cap)

    selected_ledger = pd.read_csv(args.ledger_csv_out)
    base_ledger = pd.read_csv(args.base_ledger_csv_out)
    blocking: list[str] = []
    if len(selected_ledger) != len(oos):
        blocking.append("selected OOS ledger row count mismatch")
    if len(base_ledger) != len(oos):
        blocking.append("base OOS ledger row count mismatch")
    if len(selected_ledger) == len(base_ledger):
        base_n = pd.to_numeric(base_ledger["experiment_notional"], errors="coerce").fillna(0.0)
        sel_n = pd.to_numeric(selected_ledger["experiment_notional"], errors="coerce").fillna(0.0)
        if bool(((base_n <= 1e-12) & (sel_n > 1e-12)).any()):
            blocking.append("timing layer created entries blocked by safe cap")
        if bool((sel_n > pd.to_numeric(selected_ledger["base_experiment_notional"], errors="coerce").fillna(0.0) + 1e-12).any()):
            blocking.append("timing layer increased notional above safe base")
        if bool((pd.to_numeric(selected_ledger["core_side"], errors="coerce") != pd.to_numeric(base_ledger["core_side"], errors="coerce")).any()):
            blocking.append("timing layer changed side")
        if bool((pd.to_numeric(selected_ledger["exit_idx"], errors="coerce") != pd.to_numeric(base_ledger["exit_idx"], errors="coerce")).any()):
            blocking.append("timing layer changed exit index")
    if bool((pd.to_numeric(selected_ledger["entry_idx"], errors="coerce") < pd.to_numeric(selected_ledger["original_entry_idx"], errors="coerce")).any()):
        blocking.append("timing layer entered before original entry")
    active = ~selected_ledger["blocked"].astype(bool)
    if bool((pd.to_numeric(selected_ledger.loc[active, "entry_idx"], errors="coerce") >= pd.to_numeric(selected_ledger.loc[active, "exit_idx"], errors="coerce")).any()):
        blocking.append("timing layer has active entry_idx >= exit_idx")
    if int(selected["oos_cost1"].get("liquidations", 0) or 0) > 0:
        blocking.append("selected OOS liquidation")
    if int(selected["oos_cost1"].get("ruin_events", 0) or 0) > 0:
        blocking.append("selected OOS account ruin")
    if float(selected["oos_cost1"].get("max_margin_fraction", 0.0)) > 1.0 + 1e-12:
        blocking.append("selected OOS margin fraction above 1")
    if float(selected["oos_cost2"].get("pnl", 0.0)) <= 0.0:
        blocking.append("selected OOS cost2 failed")
    if float(selected["oos_cost3"].get("pnl", 0.0)) <= 0.0:
        blocking.append("selected OOS cost3 failed")
    audit = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": ["selection fell back to noop; no timing alpha promoted"] if selected_name == "noop_safe_cap_replay" else [],
        "invariants": {
            "parent_trained_before_cap_train": True,
            "cap_map_learned_before_validation": True,
            "timing_selection_uses_2025_validation_only": True,
            "oos_is_2026_report_only": True,
            "no_new_entries": not any("created entries" in b for b in blocking),
            "side_never_changes": not any("changed side" in b for b in blocking),
            "exit_index_unchanged": not any("changed exit" in b for b in blocking),
            "entry_never_before_original": not any("before original" in b for b in blocking),
            "notional_never_above_safe_base": not any("increased notional" in b for b in blocking),
            "max_margin_fraction_lte_1": float(selected["oos_cost1"].get("max_margin_fraction", 0.0)) <= 1.0 + 1e-12,
            "no_liquidations": int(selected["oos_cost1"].get("liquidations", 0) or 0) == 0,
            "no_ruin": int(selected["oos_cost1"].get("ruin_events", 0) or 0) == 0,
            "cost2_survives": float(selected["oos_cost2"].get("pnl", 0.0)) > 0.0,
            "cost3_survives": float(selected["oos_cost3"].get("pnl", 0.0)) > 0.0,
        },
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    policy_out = args.model_dir / "dsac_timing_option_policy.json"
    policy_out.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "selected_safe_cap_candidate": safe_candidate,
                "selected_timing_option": selected["candidate"],
                "dsac_train_meta": dsac_meta_train,
                "dsac_oos_meta": dsac_meta_oos,
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_csv_out, index=False)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    verdict = "promote_candidate" if audit["status"] == "pass" and selected_name != "noop_safe_cap_replay" else "reject_or_noop"
    report = {
        "model_id": MODEL_ID,
        "verdict": verdict,
        "selected_parent_config": asdict(selected_parent),
        "parent_validation": parent_val,
        "selected_safe_cap": {
            **selected_safe,
            "compact": {
                "validation_cost1": _cap_compact(selected_safe["validation_cost1"]),
                "oos_cost1": _cap_compact(selected_safe["oos_cost1"]),
                "oos_cost2": _cap_compact(selected_safe["oos_cost2"]),
                "oos_cost3": _cap_compact(selected_safe["oos_cost3"]),
            },
        },
        "timing_baseline": {
            **baseline,
            "compact": {
                "validation_cost1": _cap_compact(baseline["validation_cost1"]),
                "oos_cost1": _cap_compact(baseline["oos_cost1"]),
                "oos_cost2": _cap_compact(baseline["oos_cost2"]),
                "oos_cost3": _cap_compact(baseline["oos_cost3"]),
            },
        },
        "selected": {
            **selected,
            "compact": {
                "validation_cost1": _cap_compact(selected["validation_cost1"]),
                "oos_cost1": _cap_compact(selected["oos_cost1"]),
                "oos_cost2": _cap_compact(selected["oos_cost2"]),
                "oos_cost3": _cap_compact(selected["oos_cost3"]),
            },
        },
        "audit_path": str(args.audit_out),
        "audit": audit,
        "data": {
            "parent_train_range": dgg.v1.base._range(parent_train),
            "cap_train_range": dgg.v1.base._range(cap_train_raw),
            "validation_range": dgg.v1.base._range(validation_raw),
            "oos_range": dgg.v1.base._range(oos_df),
            "parent_train_rows": int(len(parent_train)),
            "cap_train_parent_trades": int(len(cap_train)),
            "validation_parent_trades": int(len(validation)),
            "oos_parent_trades": int(len(oos)),
        },
        "dsac": {"train_meta": dsac_meta_train, "oos_meta": dsac_meta_oos},
        "artifacts": {
            "policy_json": str(policy_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "base_ledger_csv": str(args.base_ledger_csv_out),
            "audit": str(args.audit_out),
            "contract": str(args.contract_out),
        },
        "parent_grid_top10": sorted(parent_grid, key=lambda r: r["parent_selection_score"], reverse=True)[:10],
        "safe_cap_grid_top10": safe_grid.head(10).to_dict(orient="records"),
        "timing_grid_top15": grid.head(15).to_dict(orient="records"),
        "timing_grid_top_eligible": eligible.head(15).to_dict(orient="records"),
        "timing_grid_top_promotable": promotable.head(15).to_dict(orient="records"),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_contract(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "selected_safe_cap": selected_safe["candidate"]["name"],
                "selected_timing": selected_name,
                "audit": audit["status"],
                "baseline_oos_cost1": report["timing_baseline"]["compact"]["oos_cost1"],
                "selected_oos_cost1": report["selected"]["compact"]["oos_cost1"],
                "selected_oos_cost2": report["selected"]["compact"]["oos_cost2"],
                "selected_oos_cost3": report["selected"]["compact"]["oos_cost3"],
                "report": str(args.report_out),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
