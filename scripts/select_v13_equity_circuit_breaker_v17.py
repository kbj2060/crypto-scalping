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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402


MODEL_ID = "hf_v13_equity_circuit_breaker_v17_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_equity_circuit_breaker_v17_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_equity_circuit_breaker_v17_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_equity_circuit_breaker_v17_20260511_grid.csv"


@dataclass(frozen=True)
class CircuitConfig:
    name: str
    max_notional: float
    dd_halt: float
    resume_after_bars: int
    profit_lock_equity: float
    lock_after_profit: bool


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _grid() -> list[CircuitConfig]:
    out: list[CircuitConfig] = []
    idx = 0
    for cap in (1.90, 2.15, 2.40, 99.0):
        for dd in (0.14, 0.15, 0.16, 0.18):
            for resume in (999999,):
                for target in (99.0, 2.00, 2.20):
                    out.append(CircuitConfig(f"eq_cb_{idx}", cap, dd, resume, target, target < 90.0))
                    idx += 1
    return out


def backtest(df: pd.DataFrame, bundle: dict[str, Any], cfg: CircuitConfig, *, fee: float, slip: float, cost_mult: float = 1.0, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
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
    cooldown = 0
    next_cooldown = 0
    halt_until = -1
    locked = False
    peak_unreal = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    guards: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, notional, leverage, cooldown, next_cooldown, peak_unreal, open_record, trades, wins, halt_until
        fill_idx = min(i + 1, len(df) - 1)
        exit_price = _fill_price(df, fill_idx, pos, slip_eff, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        if reason == "equity_circuit_breaker":
            halt_until = i + int(cfg.resume_after_bars)
        if reason == "profit_lock_exit" and cfg.lock_after_profit:
            halt_until = len(df) + 999999
        if record and open_record is not None:
            out = dict(open_record)
            out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "peak_unrealized_pct": float(peak_unreal * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
            records.append(out)
        pos = 0
        notional = 0.0
        leverage = 1.0
        cooldown = int(next_cooldown)
        next_cooldown = 0
        peak_unreal = 0.0
        open_record = None

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd = eq / max(peak, 1e-12) - 1.0
        mdd = min(mdd, dd)
        if pos != 0:
            peak_unreal = max(peak_unreal, unreal)
            reason = ""
            if dd <= -abs(cfg.dd_halt):
                reason = "equity_circuit_breaker"
            elif cfg.profit_lock_equity < 90.0 and eq >= cfg.profit_lock_equity:
                reason = "profit_lock_exit"
            elif take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and i - entry_idx >= max_hold:
                reason = "learned_max_hold"
            if reason:
                close_position(i, reason)
                if reason == "profit_lock_exit" and cfg.lock_after_profit:
                    locked = True
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            action_counts["cash"] += 1
            continue
        if locked or i < halt_until:
            guards["halted"] = guards.get("halted", 0) + 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            action_counts["cash"] += 1
            continue
        fill_idx = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_idx, pos, slip_eff, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = min(float(dec.notional_exposure), float(cfg.max_notional))
        leverage = float(dec.leverage)
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        cash -= cash * fee_eff * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(leverage), "position_fraction": float(notional / max(leverage, 1e-12)), "fee_entry_pct": float(fee_eff * notional * 100.0), "dd_halt": float(cfg.dd_halt)}
    if pos != 0:
        close_position(len(df) - 2, "forced_end")
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "action_counts": action_counts, "exits": exits, "guard_actions": guards}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    pnl = float(c1["pnl"])
    mdd = abs(float(c1["mdd"]))
    if int(c1["trades"]) < 10:
        return -1e9 + pnl
    return float(pnl + 0.15 * float(c2["pnl"]) + 0.05 * float(c3["pnl"]) - 4.0 * max(0.0, mdd - 15.0) - 1.0 * max(0.0, 100.0 - pnl))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V17 equity high-watermark circuit breaker selected on 2025 validation only.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    base_cfg = dict(bundle["config"])
    train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train[train["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train, eval_df, list(bundle.get("feature_cols") or []))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows = []
    best = None
    for cfg in _grid():
        v1 = backtest(val, bundle, cfg, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, cfg, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, cfg, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = CircuitConfig(**best["config"])
    metrics = {}
    ledgers = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, selected, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking = []
    warnings = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] < 100.0:
        warnings.append("oos_cost1_below_100pct_target")
    if abs(float(metrics["cost1"]["mdd"])) > 15.0:
        warnings.append("oos_mdd_above_15pct_target")
    audit = {"status": "pass" if not blocking else "fail", "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] >= 100.0 and abs(float(metrics["cost1"]["mdd"])) <= 15.0 else "iterate", "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "selected_config": asdict(selected), "feature_audit": feature_audit, "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "v13 margin110 parent with validation-selected equity high-watermark circuit breaker and optional profit lock.", "base_model": str(args.model), "split_policy": "Circuit config selected on 2025 Oct-Dec validation only; 2026 fixed OOS not used for selection.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "selected": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
