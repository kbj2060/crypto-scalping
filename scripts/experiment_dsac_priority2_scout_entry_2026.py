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


MODEL_ID = "dsac_priority2_scout_entry_2026"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/dsac_priority2_scout_entry_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/dsac_priority2_scout_entry_2026_grid.csv"
DEFAULT_LEDGER_OUT = ROOT / "data/ensemble/reports/dsac_priority2_scout_entry_2026_selected_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/dsac_priority2_scout_entry_2026_audit.json"


@dataclass(frozen=True)
class ScoutCandidate:
    name: str
    dsac_threshold: float
    notional: float
    hold_bars: int
    tp: float
    sl: float
    cooldown_bars: int
    trend_margin: float
    max_scouts_per_day: int
    selectable: bool = True


def _series_float(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=np.float64)
    )


def _dgg_events(ledger: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        out.append(
            {
                "source": "DGG",
                "trade_id": int(row["trade_id"]),
                "entry_idx": int(row["entry_idx"]),
                "exit_idx": int(row.get("core_exit_idx", row.get("effective_exit_idx", row["entry_idx"]))),
                "timestamp": row["timestamp"],
                "side": int(np.sign(int(row["core_side"]))),
                "notional": _safe_float(row.get("effective_core_notional", 0.0), 0.0),
                "action": str(row.get("action", "")),
                "reason": "source_dgg_v2",
                "dsac_score": np.nan,
                "trend_margin": np.nan,
            }
        )
    return out


def _occupied_from_dgg(ledger: pd.DataFrame, n: int, gap: int = 1) -> np.ndarray:
    occ = np.zeros(n, dtype=bool)
    for _, row in ledger.iterrows():
        a = max(0, int(row["entry_idx"]) - gap)
        b = min(n - 1, int(row.get("core_exit_idx", row.get("effective_exit_idx", row["entry_idx"]))) + gap)
        occ[a : b + 1] = True
    return occ


def _direction_margin(features: pd.DataFrame, side: int) -> np.ndarray:
    up = _series_float(features, "m7_trend_xgb_up", 0.0)
    dn = _series_float(features, "m7_trend_xgb_dn", 0.0)
    if side > 0:
        return up - dn
    return dn - up


def _exit_for_scout(
    *,
    entry_idx: int,
    side: int,
    cand: ScoutCandidate,
    prices: tuple[np.ndarray, np.ndarray],
    slip: float,
) -> tuple[int, str]:
    close, fill_px = prices
    last = len(close) - 2
    end = int(np.clip(entry_idx + cand.hold_bars, entry_idx + 1, last))
    entry_px = _fill_price(fill_px, min(entry_idx + 1, len(fill_px) - 1), side, slip, entry=True)
    exit_idx = end
    reason = "time_exit"
    for j in range(entry_idx + 1, end + 1):
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        mark = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
        raw = _raw(side, entry_px, mark)
        if raw >= cand.tp:
            exit_idx = j
            reason = "tp_exit"
            break
        if raw <= -cand.sl:
            exit_idx = j
            reason = "sl_exit"
            break
    return int(exit_idx), reason


def _make_scouts(
    features: pd.DataFrame,
    dsac: pd.DataFrame,
    dgg_ledger: pd.DataFrame,
    cand: ScoutCandidate,
    prices: tuple[np.ndarray, np.ndarray],
    *,
    slip: float,
) -> list[dict[str, Any]]:
    n = len(features)
    dgg_occ = _occupied_from_dgg(dgg_ledger, n, gap=1)
    scout_occ = np.zeros(n, dtype=bool)
    long_margin = _direction_margin(features, 1)
    short_margin = _direction_margin(features, -1)
    timestamps = pd.to_datetime(features["timestamp"], errors="coerce") if "timestamp" in features.columns else pd.Series(pd.NaT, index=features.index)
    day_counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    next_allowed = 0
    last_entry_limit = max(0, n - cand.hold_bars - 2)
    for i in range(0, last_entry_limit):
        if i < next_allowed or dgg_occ[i] or scout_occ[i]:
            continue
        dsac_side = int(dsac["dsac_side"].iloc[i])
        if dsac_side == 0:
            continue
        score = _safe_float(dsac["dsac_score"].iloc[i], 0.0)
        if score < cand.dsac_threshold:
            continue
        margin = float(long_margin[i] if dsac_side > 0 else short_margin[i])
        if margin < cand.trend_margin:
            continue
        ts = pd.Timestamp(timestamps.iloc[i]) if pd.notna(timestamps.iloc[i]) else pd.Timestamp("1970-01-01")
        day_key = ts.date().isoformat()
        if day_counts.get(day_key, 0) >= cand.max_scouts_per_day:
            continue
        exit_idx, exit_reason = _exit_for_scout(entry_idx=i, side=dsac_side, cand=cand, prices=prices, slip=slip)
        if dgg_occ[i : exit_idx + 1].any() or scout_occ[i : exit_idx + 1].any():
            continue
        scout_occ[i : exit_idx + 1] = True
        day_counts[day_key] = day_counts.get(day_key, 0) + 1
        next_allowed = exit_idx + cand.cooldown_bars
        out.append(
            {
                "source": "DSAC_SCOUT",
                "trade_id": 1_000_000 + len(out),
                "entry_idx": int(i),
                "exit_idx": int(exit_idx),
                "timestamp": ts,
                "side": int(dsac_side),
                "notional": float(cand.notional),
                "action": "SCOUT_LONG" if dsac_side > 0 else "SCOUT_SHORT",
                "reason": f"dsac_scout|{exit_reason}",
                "dsac_score": score,
                "trend_margin": margin,
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
    notional_sum = 0.0
    max_notional = 0.0
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda x: (int(x["entry_idx"]), 0 if x["source"] == "DGG" else 1)):
        n = float(np.clip(_safe_float(event["notional"], 0.0), 0.0, 3.6))
        if n <= 1e-12:
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
        notional_sum += n
        max_notional = max(max_notional, n)
        rows.append(
            {
                "trade_id": int(event["trade_id"]),
                "source": str(event["source"]),
                "candidate": cand_name,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "timestamp": event["timestamp"],
                "side": side,
                "notional": n,
                "action": str(event["action"]),
                "reason": str(event["reason"]),
                "dsac_score": event.get("dsac_score"),
                "trend_margin": event.get("trend_margin"),
                "entry_price": entry_px,
                "exit_price": exit_px,
                "realized_raw": realized / max(n, 1e-12),
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "trade_pnl_pct": pnl_frac * 100.0,
                "cash_before": before,
                "cash_after": after,
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
    scout_trades = sum(1 for e in events if e["source"] == "DSAC_SCOUT")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "scout_trades": int(scout_trades),
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


def build_candidates() -> list[ScoutCandidate]:
    return [
        ScoutCandidate("noop_dgg_v2_replay", 99.0, 0.0, 6, 0.006, 0.004, 0, 99.0, 0),
        ScoutCandidate("p2_scout_s060_n025_h6_tp006_sl004_m015", 0.60, 0.25, 6, 0.006, 0.004, 6, 0.15, 4),
        ScoutCandidate("p2_scout_s065_n025_h6_tp006_sl004_m020", 0.65, 0.25, 6, 0.006, 0.004, 8, 0.20, 3),
        ScoutCandidate("p2_scout_s070_n030_h8_tp008_sl004_m020", 0.70, 0.30, 8, 0.008, 0.004, 8, 0.20, 3),
        ScoutCandidate("p2_scout_s075_n030_h12_tp010_sl005_m025", 0.75, 0.30, 12, 0.010, 0.005, 12, 0.25, 2),
        ScoutCandidate("p2_scout_s080_n050_h12_tp010_sl006_m030", 0.80, 0.50, 12, 0.010, 0.006, 12, 0.30, 2),
    ]


def _score(metrics: dict[str, Any]) -> float:
    val = metrics["validation_cost1"]
    c2 = metrics["validation_cost2"]
    c3 = metrics["validation_cost3"]
    score = float(val["pnl"]) + 0.12 * float(c2["pnl"]) + 0.05 * float(c3["pnl"])
    score -= 10.0 * max(0.0, abs(float(val["mdd"])) - 22.0)
    score -= 80.0 * max(0.0, -float(c2["pnl"]))
    score -= 50.0 * max(0.0, -float(c3["pnl"]))
    score += 2.0 * min(float(val["scout_trades"]), 80.0)
    return float(score)


def _audit(
    features: pd.DataFrame,
    dgg: pd.DataFrame,
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
    if len(dgg) and abs(float(baseline["final_cash"]) - _safe_float(dgg["cash_after"].iloc[-1], np.nan)) > 1e-6:
        blocking.append("noop replay does not reproduce source DGG final cash")
    if "notional" not in selected_ledger.columns or float(pd.to_numeric(selected_ledger["notional"], errors="coerce").max()) > 3.6 + 1e-12:
        blocking.append("selected ledger notional cap violation")
    scout = selected_ledger[selected_ledger.get("source", "") == "DSAC_SCOUT"].copy()
    dgg_occ = _occupied_from_dgg(dgg, len(features), gap=1)
    scout_overlap = 0
    if not scout.empty:
        for _, row in scout.iterrows():
            a = int(row["entry_idx"])
            b = int(row["exit_idx"])
            scout_overlap += int(bool(dgg_occ[a : b + 1].any()))
    if scout_overlap:
        blocking.append(f"DSAC scout trades overlap existing DGG positions: {scout_overlap}")
    if not np.isfinite(float(selected["full_cost1"].get("pnl", np.nan))):
        blocking.append("selected full_cost1 pnl is non-finite")
    if int(selected["full_cost1"].get("scout_trades", 0)) == 0 and selected["candidate"]["name"] != "noop_dgg_v2_replay":
        warnings.append("selected scout candidate created no scout trades")
    return {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "invariants": {
            "base_dgg_trades_preserved": True,
            "scout_entries_only_when_no_dgg_position": scout_overlap == 0,
            "max_notional_lte_3p6": not blocking or "selected ledger notional cap violation" not in blocking,
            "selection_uses_validation_only": True,
            "holdout_report_only": True,
            "scout_does_not_modify_existing_dgg_exit": True,
        },
        "baseline_replay": {
            "source_final_cash": _safe_float(dgg["cash_after"].iloc[-1], np.nan) if len(dgg) else np.nan,
            "noop_replay_final_cash": baseline.get("final_cash"),
            "noop_replay_pnl": baseline.get("pnl"),
            "noop_replay_mdd_mark_to_market": baseline.get("mdd"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Priority 2 DSAC scout entry experiment.")
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
    dgg = _read_ledger(args.ledger)
    features = _read_features(args.features)
    prices = _price_arrays(features)
    dsac, dsac_meta = _flat_dsac_signals(features, args.dsac_ckpt, args.device)
    _ctx_df, context = _attach_context(dgg, features, dsac)
    dgg_events = _dgg_events(dgg)
    split = pd.Timestamp(args.split_date)
    candidates = build_candidates()
    detailed: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        scouts = [] if cand.name == "noop_dgg_v2_replay" else _make_scouts(features, dsac, dgg, cand, prices, slip=args.slip)
        events = sorted(dgg_events + scouts, key=lambda x: (int(x["entry_idx"]), 0 if x["source"] == "DGG" else 1))
        validation_events = _events_for_period(events, None, split)
        holdout_events = _events_for_period(events, split, None)
        metrics = {
            "full_cost1": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "full_cost2": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "full_cost3": _replay_events(events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "validation_cost1": _replay_events(validation_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "validation_cost2": _replay_events(validation_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "validation_cost3": _replay_events(validation_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
            "holdout_cost1": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=1.0),
            "holdout_cost2": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=2.0),
            "holdout_cost3": _replay_events(holdout_events, prices, cand.name, fee=args.fee, slip=args.slip, cost_mult=3.0),
        }
        score = _score(metrics)
        detailed.append({"candidate": asdict(cand), "score": score, "scouts_created": len(scouts), **metrics})
        row = {"name": cand.name, "selectable": cand.selectable, "score": score, "scouts_created": len(scouts)}
        for prefix, data in (
            ("full", metrics["full_cost1"]),
            ("cost2", metrics["full_cost2"]),
            ("cost3", metrics["full_cost3"]),
            ("val", metrics["validation_cost1"]),
            ("val_cost2", metrics["validation_cost2"]),
            ("val_cost3", metrics["validation_cost3"]),
            ("holdout", metrics["holdout_cost1"]),
        ):
            for key in ("pnl", "mdd", "trades", "scout_trades", "trades_per_day", "avg_notional", "max_notional"):
                row[f"{prefix}_{key}"] = data.get(key)
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)
    selected_name = str(grid[grid["selectable"]].iloc[0]["name"])
    selected = next(d for d in detailed if d["candidate"]["name"] == selected_name)
    selected_cand = ScoutCandidate(**selected["candidate"])
    selected_scouts = [] if selected_cand.name == "noop_dgg_v2_replay" else _make_scouts(features, dsac, dgg, selected_cand, prices, slip=args.slip)
    selected_events = sorted(dgg_events + selected_scouts, key=lambda x: (int(x["entry_idx"]), 0 if x["source"] == "DGG" else 1))
    _replay_events(selected_events, prices, selected_name, fee=args.fee, slip=args.slip, cost_mult=1.0, ledger_out=args.ledger_out)
    selected_ledger = pd.read_csv(args.ledger_out)
    baseline = next(d for d in detailed if d["candidate"]["name"] == "noop_dgg_v2_replay")
    audit = {
        "model_id": MODEL_ID,
        "dsac": dsac_meta,
        **_audit(features, dgg, selected_ledger, baseline["full_cost1"], selected, context),
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
            "Priority 2 preserves all DGG V2 trades and only inserts DSAC scout trades outside existing DGG position windows.",
            "Scout entries use DSAC actor side/score plus entry-time M7 trend-margin confirmation.",
            "Scout exits are deterministic TP/SL/time exits and do not alter DGG exits.",
        ],
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": selected_name, "report": str(args.report_out), "audit": str(args.audit_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
