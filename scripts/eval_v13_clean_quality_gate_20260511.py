#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, FullyLearnedGovernorConfig, predict_policy_frame  # noqa: E402


MODEL_ID = "v13_clean_quality_gate_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/v13_clean_quality_gate_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/v13_clean_quality_gate_20260511_audit.json"


@dataclass(frozen=True)
class Gate:
    quality_min: float
    confidence_min: float
    transition_max: float
    risk_off_max: float
    min_notional: float


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)


def _fill_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["open"], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _days(df: pd.DataFrame) -> float:
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _passes(row: pd.Series, dec: pd.Series, gate: Gate) -> bool:
    transition = float(row.get("clean_regime_2024_unsup_v4_transition_risk", 0.0) or 0.0)
    risk_off = float(row.get("clean_regime_2024_unsup_v4_risk_off_prob", 0.0) or 0.0)
    return (
        float(dec.quality_score) >= float(gate.quality_min)
        and float(dec.confidence) >= float(gate.confidence_min)
        and float(dec.notional_exposure) >= float(gate.min_notional)
        and transition <= float(gate.transition_max)
        and risk_off <= float(gate.risk_off_max)
    )


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    gate: Gate,
    *,
    fee: float,
    slip: float,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown_left = 0
    next_cooldown = 0
    peak_unrealized = 0.0
    trades = wins = long_entries = short_entries = blocked = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_time": str(df["timestamp"].iloc[fill_idx]), "exit_reason": reason, "trade_pnl_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "cash_after": float(cash), "peak_unrealized_pct": float(peak_unrealized * 100.0)})
                    ledger.append(out)
                pos = 0
                notional = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                peak_unrealized = 0.0
                open_record = None
                continue

        if pos == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                continue
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            if not _passes(df.iloc[i], dec, gate):
                blocked += 1
                continue
            fill_idx = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(dec.notional_exposure)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
            if record:
                open_record = {
                    "signal_time": str(df["timestamp"].iloc[i]),
                    "entry_time": str(df["timestamp"].iloc[fill_idx]),
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "notional": float(notional),
                    "leverage": float(leverage),
                    "quality_score": float(dec.quality_score),
                    "confidence": float(dec.confidence),
                    "transition_risk": float(df.iloc[i].get("clean_regime_2024_unsup_v4_transition_risk", 0.0) or 0.0),
                    "risk_off_prob": float(df.iloc[i].get("clean_regime_2024_unsup_v4_risk_off_prob", 0.0) or 0.0),
                }
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "blocked_entries": int(blocked),
        "exits": exits,
    }
    if record:
        out["ledger"] = ledger
    return out


def _grid() -> list[Gate]:
    rows: list[Gate] = []
    for q in (0.015, 0.035, 0.060):
        for c in (0.55, 0.68):
            for t in (0.70, 1.00):
                for r in (0.85,):
                    rows.append(Gate(q, c, t, r, 1.20))
    return rows


def _score(r1: dict[str, Any], r2: dict[str, Any]) -> float:
    if r1["trades"] < 20:
        return -1e9 + r1["pnl"]
    return float(r1["pnl"] - 1.10 * abs(r1["mdd"]) - 0.30 * max(0.0, r1["pnl"] - r2["pnl"]) + 0.2 * min(r1["trades_per_day"], 3.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate causal quality gate on v13 clean-regime HF policy.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    cfg = FullyLearnedGovernorConfig(**dict(bundle.get("config", {})))
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_decisions = predict_policy_frame(bundle, val, close=_close(val))
    eval_decisions = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    best_gate = None
    best_score = -1e18
    rows = []
    for gate in _grid():
        r1 = backtest(val, bundle, gate, fee=cfg.fee, slip=cfg.slip, decisions=val_decisions)
        r2 = backtest(val, bundle, gate, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0, decisions=val_decisions)
        s = _score(r1, r2)
        rows.append({"score": float(s), **asdict(gate), "validation_cost1": r1, "validation_cost2": r2})
        if s > best_score:
            best_score = float(s)
            best_gate = gate
    if best_gate is None:
        raise RuntimeError("no selected gate")
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest(eval_df, bundle, best_gate, fee=cfg.fee * mult, slip=cfg.slip * mult, decisions=eval_decisions, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("ledger", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result
    grid_path = args.report_out.with_name(args.report_out.stem + "_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {
        "status": "pass",
        "blocking": [],
        "warnings": [],
        "model": str(args.model),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "selected_on": "2025-10-01..2025-12-31 validation only",
        "oos": "2026 fixed",
        "selected_gate": asdict(best_gate),
        "verdict": "promote_candidate" if metrics["cost1"]["pnl"] >= 100.0 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate",
    }
    report = {
        "model_id": MODEL_ID,
        "design": "v13 clean-regime HF policy plus validation-selected causal quality/confidence/clean-state gate.",
        "base_model": str(args.model),
        "selected_gate": asdict(best_gate),
        "selection_score": best_score,
        "metrics": metrics,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(grid_path), "ledgers": ledgers},
        "audit": audit,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "gate": asdict(best_gate), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
